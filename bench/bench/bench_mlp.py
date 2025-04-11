from pathlib import Path
import json
import triton.profiler as proton
import torch
import triton_bench.swiglu
from triton_bench.mxfp import downcast_to_mxfp
from triton_bench.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_bench.numerics import InFlexData
from triton_bench.routing import routing_torch, simulate_expert_sharded_routing
from triton_bench.meta import cuda_capability_geq

# Initialize cuBLAS for performance comparison (when available)
if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def _query_gpu_specs():
    """
    Query GPU specifications for performance analysis.
    Returns theoretical peak performance numbers for the detected GPU.
    """
    import subprocess
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i=0"]
    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    name = output.splitlines()[0]
    return {
        "NVIDIA H100 80GB HBM3": {"MAX_TFLOPS8": 1979, "MAX_TFLOPS16": 989, "MAX_TBPS": 3.35}, "HGX GB200":
        {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0}
    }[name]


# Get GPU specifications for computing utilization metrics
SPECS = _query_gpu_specs()


def quantize(w, dtype, dev, **opt):
    """
    Quantize weights to the specified data type.
    
    Args:
        w: Weight tensor to quantize
        dtype: Target data type (bf16, fp8, or mx4)
        dev: Device to place tensors on
        **opt: Additional options for quantization
        
    Returns:
        Tuple of (quantized_weights, flex_data, microscaling_context)
    """
    if dtype == "bf16":
        # BFloat16 quantization (simple conversion)
        return w.to(torch.bfloat16), InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        # FP8 quantization with scaling
        wq = w.to(torch.float8_e4m3fn).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        # MX4 (microscaling) quantization - 4-bit weights with scaling factors
        swizzle_mx_scale = opt["swizzle_mx_scale"]
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis)
        return w, InFlexData(), MicroscalingCtx(weight_scale=mx_scales, swizzle_mx=swizzle_mx_scale,
                                                actual_weight_scale_shape=weight_scale_shape)


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype,
              # tensor / expert parallelism
              TP=1, EP=1, name=""):
    """
    Benchmark an MoE MLP (Mixture of Experts Multi-Layer Perceptron).
    
    This benchmarks a full MoE MLP with the following components:
    1. Expert routing (if n_expts_tot > 1)
    2. First matrix multiplication with experts (W1)
    3. SwiGLU activation function
    4. Second matrix multiplication with experts (W2)
    
    Expert and tensor parallelism are simulated to evaluate performance.
    
    Args:
        batch: Batch size (number of tokens)
        dim1: Input/intermediate feature dimension
        dim2: Hidden dimension
        n_expts_tot: Total number of experts
        n_expts_act: Number of active experts per token (top-k)
        x_dtype: Input data type
        w_dtype: Weight data type
        TP: Tensor parallelism factor
        EP: Expert parallelism factor
        name: Name for the benchmark
        
    Returns:
        Tuple of (hardware_utilization, tflops, tbps)
    """
    # Validate parallelism parameters
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    dev = "cuda"
    
    # Initialize model parameters
    # -------------------------------
    # wg: Expert router weights (determines which experts to use for each token)
    # w1: First MLP layer weights (separate set for each expert)
    # w2: Second MLP layer weights (separate set for each expert)
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot, ), device=dev)
    b1 = torch.randn((dim2 // TP, ), device=dev)
    b2 = torch.randn((dim1, ), device=dev)

    # Quantize weights for the target precision
    # ----------------------------------------
    # Set up quantization options
    optg = dict()
    opt1 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    opt2 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    
    # Quantize all weights 
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt2)
    
    # Create precision configurations for each component
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))  # Router precision
    pcs = triton_bench.swiglu.PrecisionConfig(limit=1.0)                      # SwiGLU precision
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))  # First MLP layer precision 
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))  # Second MLP layer precision

    # Set up profiling
    # ---------------
    fpath = Path(f"logs/{name}/{batch}-{dim1}-{dim2}-{n_expts_tot}-{n_expts_act}-{x_dtype}-{w_dtype}.hatchet")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    proton.start(str(fpath.with_suffix('')), hook="triton")
    proton.deactivate()
    
    # Run MoE MLP forward pass multiple times for benchmarking
    # ------------------------------------------------------
    x_dtype = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    for i in range(100):
        # Generate random input tokens
        x = torch.randn((batch, dim1), device=dev)
        x = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
        
        # Step 1: Expert routing (if using MoE)
        # ------------------------------------
        if n_expts_tot > 1:
            # Compute expert routing logits using initial matrix multiply
            logits = matmul_ogs(x, wg, bg, precision_config=pcg)
            
            # Generate routing data based on logits (selects top-k experts per token)
            rdata, gather_indx, scatter_indx = routing_torch(logits, n_expts_act)
            
            # If using expert parallelism, simulate sharded experts across devices
            if EP > 1:
                m = logits.shape[0] * EP
                _, rdata, gather_indx, scatter_indx = simulate_expert_sharded_routing(m, rdata, EP, device=dev)
            
            # Convert input to the target precision for MLP computation
            x = x.to(x_dtype)
        else:
            # No MoE - standard MLP
            rdata, gather_indx, scatter_indx = None, None, None
        
        # Start profiling the main MLP computation
        proton.activate()
        
        # Commented out code for cuBLAS baseline 
        # c0 = torch.empty((x.shape[0], w1.shape[-1]), device=dev, dtype=x.dtype)
        # c1 = torch.empty((x.shape[0], w2.shape[-1]), device=dev, dtype=x.dtype)
        # cublas.matmul(x, w1.squeeze(0), c0)
        # cublas.matmul(c0, w2.squeeze(0), c1)
        
        # Step 2: Complete MoE MLP forward pass
        # -----------------------------------
        # First matrix multiplication with expert routing (gather tokens to experts)
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        
        # Apply SwiGLU activation function
        x = triton_bench.swiglu.swiglu(x, 1.0, pcs)
        
        # Second matrix multiplication with result gathering (scatter from experts back to tokens)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
        
        # Stop profiling
        proton.deactivate()
    
    # Finalize profiling
    proton.finalize()

    # Analyze benchmark results
    # ----------------------
    with open(f"{fpath}") as fd:
        data = json.load(fd)
        # Compute useful (matmul) bytes and flops from profiler data
        matmuls = [x for x in data[0]["children"] if "matmul" in x["frame"]["name"]]
        tot_bytes = sum([x["metrics"]["bytes"] for x in matmuls])
        tot_flops = {w: sum([x["metrics"].get(f"flops{w}", 0) for x in matmuls]) for w in [8, 16]}
        
        # Compute total time (including all operations)
        tot_time = sum(x["metrics"].get("time (ns)", 0) for x in data[0]["children"])
        
        # Calculate theoretical minimum execution time based on hw limits
        min_time_flops = sum([tot_flops[w] / SPECS[f"MAX_TFLOPS{w}"] for w in [8, 16]]) * 1e-3
        min_time_bytes = tot_bytes / SPECS["MAX_TBPS"] * 1e-3
        min_time = max(min_time_flops, min_time_bytes)
        
        # Calculate hardware utilization and performance metrics
        util = min_time / tot_time  # Ratio of theoretical min time to actual time
        tflops = sum([tot_flops[w] for w in [8, 16]]) / tot_time * 1e-3  # Achieved TFLOPS
        tbps = tot_bytes / tot_time * 1e-3  # Achieved memory bandwidth in TB/s

    return util, tflops, tbps


if __name__ == "__main__":
    # Check if hardware supports native MX4 operations
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10
    qxdtype = "fp8" if has_native_mx4 else "bf16"
    
    # Benchmark dense MLP (no MoE) with different quantization strategies
    print(bench_mlp(8192, 8192, 8192, 1, 1, "fp8", "fp8", TP=1, EP=1, name="dense"))
    print(bench_mlp(8192, 8192, 8192, 1, 1, qxdtype, "mx4", TP=1, EP=1, name="dense"))
    
    # Benchmark MoE MLP with parallelism (simulates LLaMA-like architecture)
    # 128 experts, 4 active per token, tensor parallelism of 4, expert parallelism of 2
    print(bench_mlp(1024, 5120, 8192, 128, 4, "fp8", "fp8", TP=4, EP=2, name="llama4"))
    print(bench_mlp(1024, 5120, 8192, 128, 4, qxdtype, "mx4", TP=4, EP=2, name="llama4"))