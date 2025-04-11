from dataclasses import dataclass
import itertools
import torch
import triton
# utilities
from triton_bench import meta
from triton_bench.numerics import InFlexData, OutFlexData, should_upcast_indices
from triton_bench.numerics import should_upcast_indices
from triton_bench.routing import GatherIndx, RoutingData, ScatterIndx
# details
from .matmul_ogs_details._matmul_ogs import (
    _compute_writeback_idx,
    _finalize_split_k,
    _matmul_ogs,
    _matmul_postprocess,
)
from .matmul_ogs_details._p_matmul_ogs import _p_matmul_ogs, get_per_device_per_stream_alloc_fn
from .matmul_ogs_details.opt_flags import make_opt_flags

# -----------------------------------------------------------------------------
#                    Matrix Multiplication + Outer Gather/Scatter
# -----------------------------------------------------------------------------
# This file implements efficient matrix multiplication with outer gather/scatter
# operations, which is the primary compute kernel for MoE (Mixture of Experts) models.
#
# The implementation supports:
# 1. Dynamic token routing to experts
# 2. Efficient gather/scatter operations
# 3. Mixed precision and quantization
# 4. Tensor parallelism and expert sharding
#
# This file coordinates two different kernel implementations:
# 1. _matmul_ogs - Standard kernel with better hardware compatibility
#    - Uses standard load/store operations
#    - Works on older GPU hardware
#    - Higher launch overhead but better compatibility
#
# 2. _p_matmul_ogs - Persistent kernel with better performance
#    - Uses Tensor Memory Access (TMA) for efficient memory operations
#    - Requires newer NVIDIA GPUs (Hopper/Blackwell) with TMA support
#    - Maintains thread persistence for reduced launch overhead
#    - Higher performance on supported hardware
#
# The main matmul_ogs function automatically selects the appropriate kernel
# based on hardware capabilities, memory layout, and operation requirements.
# -----------------------------------------------------------------------------

# ---------------------
# Numerics
# ---------------------
# These classes handle the numeric precision and quantization for MoE computation

# fmt: off

@dataclass(frozen=True)
class MicroscalingCtx:
    """
    Context for microscaling quantization.
    
    Microscaling is a technique to quantize weights to very low precision (e.g., 4-bit)
    while maintaining accuracy by using scaling factors.
    """
    # This interprets the scales as E8M0 tensors
    # Packed fp4s (e2m1) are stored as torch.uint8 tensors.
    # Not used for now, inserted here to make space in the APIs.
    act_scale: torch.Tensor | None = None        # Activation scaling factors (not used yet)
    weight_scale: torch.Tensor | None = None     # Weight scaling factors for each block

    swizzle_mx: bool = False  # Whether the weight scales are stored in swizzled 5D layout
    actual_weight_scale_shape: tuple | None = None  # Actual weight scales shape, without padding

    def __post_init__(self):
        """Validate the microscaling context configuration."""
        assert self.act_scale is None, "Activation scale not supported yet"
        if self.weight_scale is None:
            return

        if self.actual_weight_scale_shape is None:
            object.__setattr__(self, "actual_weight_scale_shape", self.weight_scale.shape)

        # Validate the scale tensor data type
        if self.weight_scale.dtype != torch.uint8:
            raise TypeError(f"Weight scale must be uint8. Got {self.weight_scale.dtype}")

        # Validate scale tensor dimensions
        if self.weight_scale.ndim != 3:
            raise ValueError(
                f"Weight scale must be 3D (experts, in_dim // BLOCK_SIZE, out_dim). Got {self.weight_scale.shape}"
            )

    def check_inputs(self, weights: torch.Tensor) -> None:
        """
        Validate that weights are compatible with microscaling settings.
        
        Args:
            weights: Expert weight tensor
        """
        if self.weight_scale is None:
            return

        valid_weight_types = {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}
        # Validate weights data type
        if weights.dtype not in valid_weight_types:
            raise TypeError(f"Weights must be one of {valid_weight_types}. Got {weights.dtype}")

        # Validate weights tensor dimensions
        if weights.ndim != 3:
            raise ValueError(f"Weights must be 3D (experts, in_dim, out_dim). Got {weights.shape}")

        # Validate shapes
        weight_scale_shape = self.actual_weight_scale_shape
        if weights.shape[0] != weight_scale_shape[0] or weights.shape[2] != weight_scale_shape[2]:
            raise ValueError(
                f"Weights and scale must have the same number of experts and output dimensions. "
                f"Got weights experts: {weights.shape[0]}, scale experts: {weight_scale_shape[0]}, "
                f"weights out_dim: {weights.shape[2]}, scale out_dim: {weight_scale_shape[2]}"
            )

        k_dim = self.get_packed_tensor_logical_shape(weights)[1]
        rounded_k_dim = (k_dim + 31) // 32 * 32
        block_size = rounded_k_dim // weight_scale_shape[1]
        if block_size != 32:
            raise ValueError(f"Block size must be 32. Got {block_size}")

    def compute_strides(self):
        """
        Compute memory strides for weight scale tensor.
        
        Returns:
            Tuple of (expert_stride, k_stride, n_stride)
        """
        if self.weight_scale is not None:
            # Check expected properties of the weights.
            if self.swizzle_mx:
                mxE, mxK, mxN = self.weight_scale.shape

                # Compute strides of the 5D swizzled tensor.
                swizzled_shape = (mxE, mxN // 128, mxK // 4, 32, 4, 4)
                s5 = 1
                s4 = swizzled_shape[5] * s5       # 4 * 1 = 4
                s3 = swizzled_shape[4] * s4       # 32 * 4 = 128
                s2 = swizzled_shape[3] * s3       # 4 * 128 = 512
                s1 = swizzled_shape[2] * s2       # (mxK//4) * 512
                s0 = swizzled_shape[1] * s1       # (mxN//128) * ((mxK//4)*512)
                mx_scale_stride_e, mx_scale_stride_n, mx_scale_stride_k = s0, s1, s2
            else:
                mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n = self.weight_scale.stride()
        else:
            mx_scale_stride_e = mx_scale_stride_k = mx_scale_stride_n = 0
        return mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n


    def get_packed_tensor_logical_shape(self, tensor: torch.Tensor):
        """
        Get the logical shape of a packed tensor, accounting for packing.
        
        For example, with 4-bit elements packed into bytes, the logical shape
        will have twice the K dimension.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (experts, logical_k_dim, output_dim)
        """
        k_dim = tensor.shape[1]
        if tensor.dtype == torch.uint8:
            # Assume 2 fp4s packed into a byte
            k_dim *= 2
        return tensor.shape[0], k_dim, tensor.shape[2]

@dataclass(frozen=True)
class FlexCtx:
    """
    Flexpoint context for managing precision in matrix multiplications.
    
    Flexpoint allows adjusting the range and precision of values dynamically
    to maintain accuracy while using lower precision types.
    """
    lhs_data: InFlexData = InFlexData()    # Left-hand side input data settings
    rhs_data: InFlexData = InFlexData()    # Right-hand side input data settings
    out_data: OutFlexData = OutFlexData()  # Output data settings

@dataclass
class PrecisionConfig:
    """
    Configuration for numeric precision in matmul operations.
    
    Controls how computation is performed and what optimizations to apply
    for best precision/performance.
    """
    max_num_imprecise_acc: int = None       # Maximum number of imprecise accumulations
    allow_tf32: bool = True                 # Whether to use TF32 acceleration when available
    flex_ctx: FlexCtx = FlexCtx()           # Flexpoint context for dynamic precision
    acc_scale: int = 1.0                    # Accumulator scaling factor
    flexpoint_saturate_inf: bool = False    # Whether to saturate on overflow
    report_quantization_err_fn: callable = None  # Function to report quantization errors

    mx_ctx: MicroscalingCtx = MicroscalingCtx()  # Microscaling context for quantization
    out_dtype: torch.dtype = None           # Override output data type
    enforce_bitwise_invariance: bool = False  # Force bitwise exact results

    def __post_init__(self):
        """Validate precision configuration."""
        empty_flex = FlexCtx()
        assert self.flex_ctx.rhs_data == empty_flex.rhs_data or self.mx_ctx.weight_scale is None, "flex and mx_ctx cannot be used together"

def mx_can_use_tma(mx_ctx: MicroscalingCtx):
    """
    Check if tensor memory access (TMA) can be used with the given microscaling context.
    
    TMA is an optimization for memory access on NVIDIA GPUs that can accelerate
    certain memory access patterns.
    
    Args:
        mx_ctx: Microscaling context
        
    Returns:
        Boolean indicating if TMA can be used
    """
    mx_scale_stride_e, mx_scale_stride_n, mx_scale_stride_k = mx_ctx.compute_strides()
    if mx_scale_stride_e * mx_ctx.weight_scale.element_size() % 16 != 0:
        return False

    if mx_ctx.swizzle_mx:
        # Check stride in bytes are multiples of 16.
        return mx_scale_stride_n * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_k * mx_ctx.weight_scale.element_size() % 16 == 0
    else:
        # Check MX is either transposed or non-transposed, and with required stride.
        return (
            (mx_scale_stride_n * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_k == 1) or
            (mx_scale_stride_k * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_n == 1)
        )

def can_use_persistent_tma(x, w, gather_indx, precision_config):
    """
    Check if persistent TMA can be used for this computation.
    
    Persistent TMA allows more efficient memory access but has certain requirements.
    
    Args:
        x: Input tensor
        w: Weight tensor
        gather_indx: Gather indices
        precision_config: Precision configuration
        
    Returns:
        Boolean indicating if persistent TMA can be used
    """
    mx_ctx = precision_config.mx_ctx
    return (
        # TMA requires CUDA 9.0, last dim contiguous, and multiple of 16-byte strides otherwise.
        meta.cuda_capability_geq(9, 0) and
        (True if gather_indx is not None else
            # Check strides of X.
            x.stride(1) * x.element_size() % 16 == 0 and x.stride(2) == 1
        ) and (
            # Check W is either transposed or non-transposed, and with required stride.
            (w.stride(1) * w.element_size() % 16 == 0 and w.stride(2) == 1) or
            (w.stride(2) * w.element_size() % 16 == 0 and w.stride(1) == 1)
        ) and (
            mx_ctx.weight_scale is None or mx_can_use_tma(mx_ctx)
        ) and (
            # MFXP4 tma requires 128 elements on the inner dim.
            # MFXP4 is represented as packed uint8.
            w.dtype != torch.uint8 or w.shape[-1] % 128 == 0
        )
        # compiler crash ?
        and (x.dtype.itemsize <= 1 or w.dtype != torch.uint8)
    )

def can_use_fused_scatter(scatter_indx):
    """
    Check if fused scatter operation can be used.
    
    Fused scatter combines the scatter operation with the matmul for better performance.
    
    Args:
        scatter_indx: Scatter indices
        
    Returns:
        Boolean indicating if fused scatter can be used
    """
    return scatter_indx is not None

# ---------------------
# Preprocessing
# ---------------------
# These functions prepare inputs for optimal computation

@dataclass(frozen=True)
class PreprocessingFeatures:
    """
    Features for preprocessing inputs for matmul_ogs.
    
    Controls how tensors are transformed before computation.
    """
    w_want_n_major: bool  # Whether weights should be N-major (output dim major)
    w_want_k_major: bool  # Whether weights should be K-major (input dim major)
    swap_xw: bool         # Whether to swap X and W operands for better performance

    def __post_init__(self):
        """Validate preprocessing features."""
        assert not (self.w_want_k_major and self.w_want_n_major), "Cannot have both K-major and N-major"

def init_preprocessing_features(w, precision_config, opt_flags):
    """
    Initialize preprocessing features based on tensor and hardware characteristics.
    
    Determines how to optimize tensor layout for the current hardware.
    
    Args:
        w: Weight tensor
        precision_config: Precision configuration
        opt_flags: Optimization flags
        
    Returns:
        PreprocessingFeatures object
    """
    mx_ctx = precision_config.mx_ctx
    swap_xw = False  # Whether or not to swap X and W operands to the tl.dot
    w_want_k_major = False
    w_want_n_major = False
    if not meta.cuda_capability_geq(10, 0):
        # Hopper transpose. Reduction dimension must be contiguous.
        if w.stride(1) != 1 and w.dtype.itemsize == 1:
            w_want_k_major = True

    if meta.cuda_capability_geq(10, 0):
        swap_xw = mx_ctx.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
        if swap_xw:
            w_want_k_major = True
        # fp4 padded mode requires the contiguous dim size to be a multiple of 64 bytes. If it is K-major and does not
        # meet the requirement, make the tensor N-major instead.
        # But, don't do this if we're going to swap X and W in which case we would transpose W again.
        if w.stride(1) == 1 and w.dtype == torch.uint8 and w.shape[1] % 64 != 0 and not swap_xw:
            w_want_n_major = True
    if not w_want_k_major and not w_want_n_major:
        w_want_k_major = True
    return PreprocessingFeatures(w_want_n_major, w_want_k_major, swap_xw)

def apply_preprocessing_features(x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features):
    """
    Apply preprocessing features to prepare tensors for matmul.
    
    Transforms tensors according to the preprocessing features and prepares
    routing indices for computation.
    
    Args:
        x: Input tensor
        w: Weight tensor
        gather_indx: Gather indices
        scatter_indx: Scatter indices
        routing_data: Routing data
        opt_flags: Optimization flags
        preprocessing_features: Preprocessing features
        
    Returns:
        Tuple of processed tensors and indices
    """
    # Prepare fused scatter if needed
    has_fused_scatter_scratchpad = opt_flags.fused_scatter and routing_data.n_expts_act > 1
    if has_fused_scatter_scratchpad:
        # Compute writeback indices for fused scatter
        M = scatter_indx.src_indx.shape[0]
        writeback_idxs = torch.empty((M,), dtype=torch.int32, device=x.device)
        writeback_size = writeback_idxs.shape[0]
        BLOCK_M=256
        # Run kernel to compute writeback indices
        _compute_writeback_idx[(triton.cdiv(M, BLOCK_M),)](
            writeback_idxs,
            scatter_indx.dst_indx,
            scatter_indx.src_indx,
            M // routing_data.n_expts_act,
            M,
            BLOCK_M=BLOCK_M,
            N_EXPTS_ACT=routing_data.n_expts_act,
        )
    elif scatter_indx is not None and routing_data.n_expts_act == 1:
        # Simple scatter indices for single expert per token
        writeback_idxs = scatter_indx.dst_indx
        writeback_size = scatter_indx.dst_indx.shape[0]
    else:
        # No scatter needed
        writeback_idxs, writeback_size = None, None
        
    # Apply tensor layout transformations based on preprocessing features
    # some transposition variants aren't supported
    # TODO: this is extremely expensive and we should find
    # a way to surface this to the user
    if preprocessing_features.w_want_n_major:
        w = w.contiguous()
    elif preprocessing_features.w_want_k_major:
        w = w.transpose(-1, -2).contiguous().transpose(-1, -2)
        
    # Compute expert data mapping for the given token count and block size
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    expt_data = routing_data.expt_data(M, opt_flags.block_m)
    
    return x, w, preprocessing_features.swap_xw, writeback_idxs, writeback_size, expt_data

# ---------------------
# Postprocessing
# ---------------------
# These functions process the matmul results

@dataclass(frozen=True)
class PostprocessingFeatures:
    """
    Features for postprocessing matmul_ogs results.
    
    Controls how results are finalized after computation.
    """
    finalize_splitk: bool  # Whether to finalize split-K reduction
    finalize_scatter: bool  # Whether to apply scatter operation to results

    def __post_init__(self):
        """Validate postprocessing features."""
        assert not (self.finalize_splitk and self.finalize_scatter)

def init_postprocessing_features(routing_data, scatter_indx, opt_flags):
    """
    Initialize postprocessing features based on computation requirements.
    
    Args:
        routing_data: Routing data
        scatter_indx: Scatter indices
        opt_flags: Optimization flags
        
    Returns:
        PostprocessingFeatures object
    """
    finalize_scatter = scatter_indx is not None and routing_data.n_expts_act > 1
    # TODO: there should be an assert somewhere!
    finalize_splitk = opt_flags.split_k > 1 and not finalize_scatter
    return PostprocessingFeatures(finalize_splitk=finalize_splitk,
                                  finalize_scatter=finalize_scatter)

def apply_postprocessing_features(scatter_indx, opt_flags, expt_offs, num_indx, precision_config, routing_data,
                           postprocess_features, memory):
    """
    Apply postprocessing to finalize matmul results.
    
    Handles split-K reduction and scatter operations as needed.
    
    Args:
        scatter_indx: Scatter indices
        opt_flags: Optimization flags
        expt_offs: Expert offsets
        num_indx: Number of indices
        precision_config: Precision configuration
        routing_data: Routing data
        postprocess_features: Postprocessing features
        memory: Memory buffers
        
    Returns:
        Finalized output tensor
    """
    out = memory["output"]
    flex_ctx = precision_config.flex_ctx
    
    # finalize split-k reduction if needed
    if postprocess_features.finalize_splitk:
        inp = memory["scratchpad"]["matmul"]
        out_splitk = memory["output"]
        out_splitk_flex = precision_config.flex_ctx.out_data
        assert out_splitk.stride(3) == 1
        flattened_M = inp.shape[1] * inp.shape[2]
        N = inp.shape[3]
        grid = (flattened_M, triton.cdiv(N, opt_flags.block_n))
        # Run kernel to reduce partial results from split-K
        _finalize_split_k[grid](
            inp, inp.stride(0), inp.stride(2),
            flex_ctx.out_data.reinterpret(out_splitk), out_splitk.stride(2),
            *out_splitk_flex,
            flattened_M, N, opt_flags.split_k,
            None if expt_offs is None else expt_offs[-1],
            num_indx,
            1,
            opt_flags.block_n,
            precision_config.flexpoint_saturate_inf,
        )
        out = out_splitk
    
    # finalize scatter operation if needed
    # batched mode not supported.
    if postprocess_features.finalize_scatter:
        has_fused_scatter_scratchpad = opt_flags.fused_scatter and routing_data.n_expts_act > 1
        if has_fused_scatter_scratchpad:
            inp = memory["output"]
        else:
            inp = memory["scratchpad"]["matmul"]
        n_final_rows = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act
        inp_flex = OutFlexData() if inp.dtype == torch.float32 else precision_config.flex_ctx.out_data
        out_scatter = memory["output"]
        out_scatter_flex = precision_config.flex_ctx.out_data
        assert inp.shape[1] == 1
        
        # Configure kernel parameters based on architecture
        if meta.is_hip():
            num_warps = 2
            BLOCK_N = 2048
            warps_per_sm = 32
        else:
            num_warps = 16
            BLOCK_N = 4096
            warps_per_sm = 128
        
        # Calculate grid dimensions for optimal performance
        num_pid = meta.num_sms() * (warps_per_sm // num_warps)
        N = inp.shape[3]
        M = n_final_rows
        # assert M == out_scatter.shape[1], f"{M}, {out_scatter.shape}"
        N_BLOCKS = triton.cdiv(N, BLOCK_N)
        M_BLOCKS = min(M, max(1, triton.cdiv(num_pid, N_BLOCKS)))
        grid = (M_BLOCKS, N_BLOCKS)
        
        # Run kernel to apply scatter operation
        _matmul_postprocess[grid](
            flex_ctx.out_data.reinterpret(out_scatter),
            *out_scatter_flex,
            flex_ctx.out_data.reinterpret(inp), inp.stride(0), inp.stride(2),
            inp_flex.expected_scale,
            scatter_indx.src_indx,
            inp.shape[0], M, N,
            None if expt_offs is None else expt_offs[-1],
            EXPT_PER_TOK=routing_data.n_expts_act,
            BLOCK_N=BLOCK_N,
            M_BLOCKS=M_BLOCKS,
            num_warps=num_warps,
            flexpoint_saturate_inf=precision_config.flexpoint_saturate_inf,
            HAS_FUSED_SCRATCHPAD=has_fused_scatter_scratchpad,
        )
        out = out_scatter
        
        # Trim unnecessary part of output if needed
        if has_fused_scatter_scratchpad:
            # Discard scratchpad part.
            # This still gives a contiguous tensor, because shape[0] > 1 only when
            # batch mode is enabled, in which case this is a no-op (there's no scratchpad).
            out = out[:, :, :n_final_rows, :]
    
    return out


# ---------------------
# Allocation
# ---------------------
# These functions handle memory allocation for the computation

@dataclass
class MatmulAllocation:
    """
    Memory allocation specifications for matmul_ogs.
    
    Specifies the device, output shape/type, and scratchpad buffers needed.
    """
    device: str
    output: tuple[tuple[int], torch.dtype]
    scratchpads: dict[str, tuple]


def init_allocation(x, w, precision_config, routing_data, gather_indx, scatter_indx, opt_flags,
                    preprocessing_features, postprocessing_features):
    """
    Initialize memory allocation for matmul_ogs.
    
    Determines the shapes and types of all buffers needed for computation.
    
    Args:
        x: Input tensor
        w: Weight tensor
        precision_config: Precision configuration
        routing_data: Routing data
        gather_indx: Gather indices
        scatter_indx: Scatter indices
        opt_flags: Optimization flags
        preprocessing_features: Preprocessing features
        postprocessing_features: Postprocessing features
        
    Returns:
        MatmulAllocation object
    """
    # ---- determine output shape ------
    N = precision_config.mx_ctx.get_packed_tensor_logical_shape(w)[-1]
    # by default - M is number of rows in the activations
    M = x.shape[1]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.src_indx.shape[0]
    
    # Determine shape of final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    elif opt_flags.fused_scatter:
        # we need the scratchpad and the output to be contiguous in memory
        Mc = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act # compressed number of rows
        y_rows = M + Mc
    else:
        Mc = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act # compressed number of rows
        y_rows = Mc
    
    y_shape = (x.shape[0], y_rows, N)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (y_shape, out_dtype)
    
    # ---- scratchpad allocation -----
    scratchpad = dict()
    # if we need either standalone scatter or split-k, the matmul output will need post-processing
    if postprocessing_features.finalize_splitk or (postprocessing_features.finalize_scatter and not opt_flags.fused_scatter):
        dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
        scratchpad["matmul"] = ((opt_flags.split_k, x.shape[0], M, N), dtype)
    
    return MatmulAllocation(x.device, output, scratchpad)

def apply_allocation(allocation: MatmulAllocation, output):
    """
    Apply memory allocation for matmul_ogs.
    
    Allocates all buffers needed for computation.
    
    Args:
        allocation: MatmulAllocation object
        output: Pre-allocated output buffer (optional)
        
    Returns:
        Dictionary of allocated memory buffers
    """
    ret = dict()
    if output is None:
        output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
    else:
        assert output.shape == allocation.output[0]
    ret["output"] = output[None, :, :]
    ret["scratchpad"] = {
        k: torch.empty(v[0], device=allocation.device, dtype=v[1])
            for k, v in allocation.scratchpads.items()
    }
    return ret

# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------

def matmul_ogs(x, w, bias,
               routing_data: RoutingData | None = None,
               gather_indx: GatherIndx | None = None,
               scatter_indx: ScatterIndx | None = None,
               precision_config: PrecisionConfig | None = None,
               betas: torch.Tensor | None = None,
               gammas: torch.Tensor | None = None,
               out_alpha: float | None = None,
               y: torch.Tensor | None = None,
               ):
    """
    Matrix multiplication with outer gather/scatter operations.
    
    This is the primary compute kernel for MoE (Mixture of Experts) models.
    It performs the following operation:
    
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    
    Args:
        x: Input tensor (activations)
        w: Weight tensor (expert weights)
        bias: Bias tensor
        routing_data: Token-to-expert routing information
        gather_indx: Indices for gathering tokens to experts
        scatter_indx: Indices for scattering results back
        precision_config: Precision configuration
        betas: Per-token scaling factors for bias
        gammas: Per-token scaling factors for output
        out_alpha: Global output scaling factor
        y: Pre-allocated output tensor (optional)
        
    Returns:
        Output tensor
    """
    # Check if input is batched
    is_input_batched = x.ndim == 3
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    
    # Use default precision config if none provided
    if precision_config is None:
        precision_config = PrecisionConfig()
    
    # Ensure inputs have proper shape
    if w.ndim == 2:
        w = w.view(1, w.shape[-2], w.shape[-1])
    if x.ndim == 2:
        x = x.view(1, x.shape[-2], x.shape[-1])
    assert w.ndim == 3
    assert x.ndim == 3
    
    # Get microscaling context
    mx_ctx = precision_config.mx_ctx
    
    # Determine shapes
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    
    # Create routing data if not provided
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    
    # Determine batch size
    batch_size = w.shape[0] if routing_data.expt_hist is None else 1
    
    # Get packed tensor logical shape
    n_expts_tot, K, N = mx_ctx.get_packed_tensor_logical_shape(w)
    
    # Validate inputs
    mx_ctx.check_inputs(w)
    
    # Compute scale strides
    mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n = mx_ctx.compute_strides()
    
    # Determine output data type
    out_dtype = precision_config.out_dtype or x.dtype
    
    # Compute optimization flags
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, precision_config,
        M, N, K, routing_data,
        can_use_persistent_tma(x, w, gather_indx, precision_config),
        can_use_fused_scatter(scatter_indx),
    )
    
    # Compute grid size
    if not is_input_batched:
        grid_m = routing_data.n_blocks(M, opt_flags.block_m)
    else:
        grid_m = triton.cdiv(M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    
    # Validate inputs
    assert n_expts_tot == routing_data.n_expts_tot
    assert grid_m > 0
    assert x.dtype == w.dtype or mx_ctx.weight_scale is not None
    
    # Determine necessary pre/post processing
    preprocessing_features = init_preprocessing_features(w, precision_config, opt_flags)
    postprocessing_features = init_postprocessing_features(routing_data, scatter_indx, opt_flags)
    
    # Allocate output/scratchpad memory
    allocation = init_allocation(x, w, precision_config, routing_data, gather_indx, scatter_indx, opt_flags,
                                 preprocessing_features, postprocessing_features)
    memory = apply_allocation(allocation, y)
    
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(x.device))
    
    # Determine intermediate tensors and postprocess kernels based on situation
    Mc = M // routing_data.n_expts_act
    out0, out0_flex = memory["output"], precision_config.flex_ctx.out_data
    
    # Update output buffer pointers if postprocessing is needed
    if postprocessing_features.finalize_scatter or postprocessing_features.finalize_splitk:
        if opt_flags.fused_scatter:
            out0 = memory["output"]
        else:
            out0 = memory["scratchpad"]["matmul"]
        out0_flex = OutFlexData() if out0.dtype == torch.float32 else precision_config.flex_ctx.out_data
    
    # Pre-processing
    x, w, swap_xw, writeback_idxs, writeback_size, expt_data = apply_preprocessing_features(
        x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features
    )
    
    # Validate expert data
    if expt_data.buffer is not None:
        assert expt_data.buffer.shape[0] == 3*n_expts_tot + 2 + grid_m, \
            f"invalid expt_data, {expt_data.buffer.shape}, {n_expts_tot=}, {grid_m=}"
    
    # Configure grid for matrix multiplication
    n_cta = batch_size * grid_m * grid_n * opt_flags.split_k
    n_cta = min(meta.num_sms(), n_cta) if opt_flags.is_persistent else n_cta
    
    # Get flex context
    flex = precision_config.flex_ctx
    
    # Prepare bias arguments
    bias_stride = None if bias is None else bias.stride(0)
    
    # Get number of indices
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]
    
    # Launch matrix multiplication kernel (regular or persistent)
    # Select persistent kernel (_p_matmul_ogs) if TMA is supported and memory layout is compatible
    # Otherwise fall back to standard kernel (_matmul_ogs) for better compatibility
    (_p_matmul_ogs if opt_flags.is_persistent else _matmul_ogs)[(n_cta,)](
                   flex.out_data.reinterpret(memory["output"]),
                   flex.out_data.reinterpret(out0), *out0.stride(),
                   *out0_flex,
                   flex.lhs_data.reinterpret(x), x.stride(0), x.stride(1), x.stride(2),
                   flex.lhs_data.scale,
                   flex.rhs_data.reinterpret(w), w.stride(0), w.stride(1), w.stride(2), w.stride(2) != 1,
                   flex.rhs_data.scale,
                   mx_ctx.weight_scale, mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n, mx_scale_stride_n != 1,
                   bias, bias_stride,
                   x.shape[1],
                   x.shape[1] if routing_data.expt_hist is None else None,
                   N, K,
                   betas, gammas,
                   None if gather_indx is None else gather_indx.src_indx,
                   None if scatter_indx is None else scatter_indx.src_indx,
                   num_indx,
                   writeback_idxs, writeback_size,
                   expt_data.hist, expt_data.offs, expt_data.offs_sum, expt_data.blocks,
                   batch_size, grid_m, grid_n,
                   out_alpha,
                   routing_data.n_expts_tot, routing_data.n_expts_act,
                   precision_config.max_num_imprecise_acc,
                   precision_config.allow_tf32,
                   precision_config.flexpoint_saturate_inf,
                   flex.rhs_data.is_per_batch,
                   opt_flags.block_m,
                   opt_flags.block_n,
                   opt_flags.block_k,
                   opt_flags.group_m,
                   XCD_SWIZZLE=opt_flags.xcd_swizzle,
                   SWIZZLE_MX=mx_ctx.swizzle_mx,
                   SPLIT_K=opt_flags.split_k,
                   EVEN_K=K % opt_flags.block_k == 0,
                   W_CACHE_MODIFIER=opt_flags.w_cache_modifier,
                   TOKENS_PER_EXPT_FOR_ANNOTATION=routing_data.expected_tokens_per_expt,
                   num_warps=opt_flags.num_warps,
                   num_stages=opt_flags.num_stages,
                   arch=opt_flags.arch,
                   UPCAST_INDICES=should_upcast_indices(x, w, out0),
                   DISABLE_Y_TMA=out0.stride(-2) * out0.dtype.itemsize % 16 != 0,
                   SWAP_XW=swap_xw,
                   NUM_SMS = n_cta,
                   **opt_flags.target_kernel_kwargs)
    
    # Post-processing
    out = apply_postprocessing_features(scatter_indx, opt_flags, expt_data.offs,
                                num_indx, precision_config, routing_data,
                                postprocessing_features, memory)

    # Reshape output tensor to expected dimensions
    out = out.squeeze(0)
    if not is_input_batched:
        out = out.view(out.shape[-2], out.shape[-1])
    
    return out


# -----------------------------------------------------------------------------
# PyTorch Reference Implementation (for verification)
# -----------------------------------------------------------------------------

def matmul_ogs_torch(x, w, bias,
                 routing_data: RoutingData = None,
                 gather_indx: GatherIndx = None,
                 scatter_indx: ScatterIndx = None,
                 precision_config: PrecisionConfig = None,
                 betas = None,
                 gammas = None,
                 round_x = None, round_y = None,
                 ):
    """
    PyTorch reference implementation of matmul with outer gather/scatter.
    
    This implementation uses standard PyTorch operations and is used for
    verification of the optimized Triton implementation.
    
    Args:
        x: Input tensor
        w: Weight tensor
        bias: Bias tensor
        routing_data: Routing data
        gather_indx: Gather indices
        scatter_indx: Scatter indices
        precision_config: Precision configuration
        betas: Per-token scaling factors for bias
        gammas: Per-token scaling factors for output
        round_x: Function to round input values
        round_y: Function to round output values
        
    Returns:
        Output tensor
    """
    is_input_batched = x.ndim == 3
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    
    # Default rounding functions
    if round_x is None:
        round_x = lambda x: x
    if round_y is None:
        round_y = lambda x: x
    
    # Ensure inputs have proper shape
    if w.ndim == 2:
        w = w.view(1, w.shape[0], w.shape[1])
    if x.ndim == 2:
        x = x.view(1, x.shape[0], x.shape[1])
    
    # Create routing data if not provided
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    n_expts_act = routing_data.n_expts_act
    
    # Compute memory offsets
    if routing_data.n_expts_tot > 1 and not is_input_batched:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[1]] for _ in range(w.shape[0])]
    
    # Allocate output tensor
    n_rows = x.shape[1] if gather_indx is None else gather_indx.dst_indx.shape[0]
    y = torch.zeros((x.shape[0], n_rows, w.shape[-1]), device=x.device, dtype=x.dtype)
    
    # Compute matrix multiplication for each expert
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx.src_indx[lo:hi] // n_expts_act
        
        # Select batch
        batch = i if is_input_batched else 0
        
        # Compute matrix multiplication
        out = torch.matmul(round_x(x[batch, idx, :], torch.arange(lo, hi, device="cuda")).float(),
                           w[i, :, :].float())
        
        # Apply bias if provided
        if bias is not None:
            out += bias[i, :] if betas is None else bias[i, :] * betas[lo:hi, None]
        
        # Apply scaling if provided
        if gammas is not None:
            out *= gammas[lo:hi, None]
        
        # Update output
        y[batch, lo:hi, :] = round_y(out)
    
    # Reshape output
    if not is_input_batched:
        y = y.view(y.shape[1], y.shape[2])
    
    # Return output if no scatter needed
    if scatter_indx is None:
        return y
    
    # Accumulate output from all experts
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    
    # Scatter results back
    for i, (lo, hi) in enumerate(offs):
        dst_idx = scatter_indx.dst_indx[lo:hi] // n_expts_act
        msk = dst_idx != -1
        out[dst_idx[msk], :] += y[lo:hi, :][msk, :].float()
    
    return out