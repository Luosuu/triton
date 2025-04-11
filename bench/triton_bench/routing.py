import torch
import triton
from dataclasses import dataclass, field
import triton.language as tl


@dataclass
class GatherIndx:
    """
    Indices for an operation that performs:
    Y = X[src_idx, :] - gather operation from source to destination
    
    This operation selects rows from X based on src_indx and places them in Y.
    Often used to gather token features before routing them to experts.
    """
    # src_indx: indices into the source tensor X
    src_indx: torch.Tensor
    # dst_indx: indices into the destination tensor Y
    # Note: `dst_idx[src_idx] = arange(0, N)` - reconstructs original order
    dst_indx: torch.Tensor


@dataclass
class ScatterIndx:
    """
    Indices for an operation that performs:
    Y[dst_idx, :] = X - scatter operation from source to destination
    
    This operation places rows from X into specific positions in Y based on dst_indx.
    Often used to scatter token features after expert processing, back to their original positions.
    """
    # src_indx: indices into the source tensor X
    src_indx: torch.Tensor
    # dst_indx: indices into the destination tensor Y
    # Note: `dst_idx[src_idx] = arange(0, N)` - reconstructs original order
    dst_indx: torch.Tensor


@dataclass
class ExptData:
    """
    Container for expert routing data used during MoE computation.
    
    This structure manages the mapping between tokens and experts.
    """
    # hist: histogram of token counts per expert
    hist: torch.Tensor
    # offs: offsets into token arrays for each expert (start position of each expert's tokens)
    offs: torch.Tensor
    # offs_sum: cumulative sum of offsets (total token count)
    offs_sum: torch.Tensor
    # blocks: mapping of computational blocks to experts
    blocks: torch.Tensor
    # buffer: raw buffer containing all of the above data
    buffer: torch.Tensor


# Expert data kernel - fills in expert data structures for efficient computation
@triton.jit
def _fill_expt_data(
    ExpertData,  # Buffer to store expert routing information
    n_blocks,    # Total number of computational blocks
    n_experts: tl.constexpr,  # Number of experts in the MoE model
    BLOCK_M: tl.constexpr,    # Block size in the M dimension (token dimension)
    BLOCK_N: tl.constexpr,    # Block size in the N dimension (feature dimension)
) -> None:
    # Each program (thread group) handles one expert
    expert_id = tl.program_id(0)
    # Create offsets for accessing data
    offs = tl.arange(0, BLOCK_N)

    # Calculate offsets for loading expert histogram data
    # We offset by 1 to avoid out-of-bounds accesses
    ranges_offs = tl.where(offs > 0, offs - 1, 0)
    # Create mask for valid expert accesses
    expert_mask = offs > 0 and ranges_offs < n_experts
    
    # Load the number of tokens assigned to each expert
    tokens_per_expert = tl.load(ExpertData + ranges_offs, mask=expert_mask, other=0)
    # Calculate how many blocks each expert needs based on its token count
    blocks_per_expert = tl.cdiv(tokens_per_expert, BLOCK_M)  # Ceiling division
    # Calculate the starting block index for each expert using cumulative sum
    block_starts = tl.cumsum(blocks_per_expert, axis=0)

    # Only the first program (thread group) stores the global metadata
    if expert_id == 0:
        # Calculate token start positions for each expert
        token_starts = tl.cumsum(tokens_per_expert, axis=0)

        # Create mask for valid storage positions
        store_mask = offs < (n_experts + 1)
        # Store token start positions to the buffer
        tl.store(ExpertData + n_experts + offs, token_starts, mask=store_mask)
        # Store block start positions to the buffer
        tl.store(ExpertData + 2 * n_experts + 1 + offs, block_starts, mask=store_mask)

    # Get the first block index for the current expert
    block_starts = tl.where(offs == expert_id, block_starts, 0)
    first_block = tl.sum(block_starts, axis=0)

    # Load the number of tokens for the current expert
    n_tokens = tl.load(ExpertData + expert_id)
    # Calculate the number of blocks needed for this expert
    n_blocks = tl.cdiv(n_tokens, BLOCK_M)
    # Calculate the starting position in the buffer for block mapping data
    ExpertData += 3 * n_experts + 2 + first_block
    
    # Populate the block mapping data: each block stores the expert ID it belongs to
    # and its local block index within that expert
    for block_off in range(0, n_blocks, BLOCK_N):
        block_offs = block_off + tl.arange(0, BLOCK_N)
        # Pack block index and expert ID into a single integer:
        # Upper 16 bits: block index, Lower 16 bits: expert ID
        data = (block_offs << 16) + expert_id
        # Store the packed data, with a mask to handle boundary conditions
        tl.store(ExpertData + block_offs, data, mask=block_offs < n_blocks)


@dataclass
class RoutingData:
    """
    Container for all routing-related data in MoE computation.
    
    This class manages how tokens are routed to experts, including scaling factors,
    expert assignments, and block mappings.
    """
    # Scaling factors from router gate, applied to each token's expert output
    gate_scal: torch.Tensor = field()
    # Histogram of tokens per expert
    expt_hist: torch.Tensor = field()
    # Total number of experts in the model
    n_expts_tot: int = field()
    # Number of active experts per token (top-k selection)
    n_expts_act: int = field()
    # Cache of computed expert data mappings (key: (n_rows, block_m))
    expt_data_map: dict[int, torch.Tensor] = field(default_factory=dict, init=False)

    # Expected tokens per expert - used for performance optimization when sharding
    # This helps provide consistent annotations for benchmarking
    expected_tokens_per_expt: int = field(default=None)

    def n_blocks(self, n_rows, block_m):
        """
        Calculate the number of computational blocks needed for n_rows tokens.
        
        Args:
            n_rows: Number of token rows to process
            block_m: Block size in the token dimension
            
        Returns:
            Total number of blocks needed
        """
        if n_rows <= self.n_expts_tot:
            # Special case: fewer tokens than experts
            return n_rows
        else:
            # Calculate blocks needed with special handling for block boundaries
            return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m) + self.n_expts_tot - 1

    def _compute_expt_data(self, n_rows, block_m):
        """
        Compute expert data mapping for the given token count and block size.
        
        This method fills the ExptData structure by running the Triton kernel.
        
        Args:
            n_rows: Number of token rows
            block_m: Block size in the token dimension
            
        Returns:
            ExptData containing the computed mapping
        """
        routing_matrix = None
        expt_histogram = self.expt_hist
        assert routing_matrix is not None or expt_histogram is not None, ("Must pass routing_matrix or expt_histogram")
        
        # Determine number of experts from inputs
        n_experts = routing_matrix.shape[1] if routing_matrix is not None else expt_histogram.numel()
        device = routing_matrix.device if routing_matrix is not None else expt_histogram.device
        
        # Calculate the number of blocks needed
        if n_rows < n_experts:
            n_blocks = n_rows
        else:
            n_blocks = triton.cdiv(n_rows - n_experts + 1, block_m) + n_experts - 1

        # Create buffer to hold all expert mapping data
        shape = n_experts * 3 + 2 + n_blocks
        expt_data = torch.full((shape, ), -1, dtype=torch.int32, device=device)
        
        # Fill the expert histogram data
        if expt_histogram is not None:
            expt_data[:n_experts] = expt_histogram
        else:
            torch.sum(routing_matrix, dim=0, out=expt_data[:n_experts])

        # Determine BLOCK_N size as power of 2 for efficiency
        BLOCK_N = triton.next_power_of_2(n_experts + 1)
        # Set up the grid for parallel execution (one program per expert)
        grid = (n_experts, )
        
        # Run the Triton kernel to compute the expert data mapping
        _fill_expt_data[grid](
            expt_data,
            n_blocks,
            n_experts,
            block_m,
            BLOCK_N,
        )
        
        # Extract different components from the buffer
        n_expts_tot = self.n_expts_tot
        hist = expt_data[:n_expts_tot]  # Expert histogram
        offs = expt_data[n_expts_tot:2 * n_expts_tot + 1]  # Token offsets
        offs_sum = expt_data[3 * n_expts_tot + 2 - 1]  # Total tokens
        blocks = expt_data[n_expts_tot + 2 * (n_expts_tot + 1):]  # Block mapping
        
        # Return structured expert data
        return ExptData(hist, offs, offs_sum, blocks, expt_data)

    def expt_data(self, n_rows, block_m):
        """
        Get expert data mapping, computing it if not cached.
        
        Args:
            n_rows: Number of token rows
            block_m: Block size in the token dimension
            
        Returns:
            ExptData containing the mapping
        """
        if self.expt_hist is None:
            # Return empty expert data if no histogram available
            return ExptData(None, None, None, None, None)
        
        # Use cached data if available, otherwise compute it
        key = (n_rows, block_m)
        if key not in self.expt_data_map:
            self.expt_data_map[key] = self._compute_expt_data(*key)
        return self.expt_data_map[key]


def routing_torch(logits, n_expts_act, expt_indx=None):
    """
    Compute token-to-expert routing using PyTorch operations.
    
    This function routes each token to its top-k experts based on router logits.
    
    Args:
        logits: Routing logits of shape [n_tokens, n_experts]
        n_expts_act: Number of active experts per token (top-k)
        expt_indx: Optional pre-computed expert indices (skips topk computation)
        
    Returns:
        RoutingData: Routing information
        GatherIndx: Indices for gathering token features before expert computation
        ScatterIndx: Indices for scattering token features after expert computation
    """
    def topk(vals, k, expt_indx):
        """
        Select top-k experts for each token.
        
        Args:
            vals: Values to select from
            k: Number of experts to select
            expt_indx: Optional pre-computed expert indices
            
        Returns:
            Selected values and their indices
        """
        # Use provided indices or compute top-k
        if expt_indx is None:
            tk_idx = torch.argsort(-vals, dim=1, stable=True)[:, :k]  # Descending order
        else:
            tk_idx = expt_indx
        # Get the values at the selected indices
        tk_val = torch.take_along_dim(vals, tk_idx, dim=1)
        return tk_val, tk_idx

    # Get number of experts from logits shape
    _, n_expts_tot = logits.shape
    
    # Compute expert selection and scaling factors
    expt_scal, expt_indx = topk(torch.softmax(logits, dim=-1), n_expts_act, expt_indx)
    
    # Sort each token's experts for more efficient processing
    expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
    expt_scal = torch.gather(expt_scal, 1, sort_indices)
    
    # Flatten top-k data
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    
    # Sort by expert_id to make experts contiguous for efficient matrix multiplication
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx)
    gate_scal = expt_scal[topk_indx]
    
    # Compute histogram of tokens per expert
    hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)
    
    # Create gather/scatter indices for token routing
    gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
    scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
    
    # Return routing data and indices
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act), gather_indx, scatter_indx


def simulate_expert_sharded_routing(n_global_rows, routing_data, n_expt_shards, row_align=1, device="cuda", cache=None):
    """
    Simulate routing for a sharded MoE setup (experts distributed across devices).
    
    This function creates realistic routing data for benchmarking sharded MoE models
    without requiring actual multi-device setup.
    
    Args:
        n_global_rows: Total number of token rows across all shards
        routing_data: Base routing data
        n_expt_shards: Number of expert shards (devices)
        row_align: Alignment factor for row count
        device: Device to create tensors on
        cache: Optional cache for routing data
        
    Returns:
        n_local_rows: Number of rows assigned to local experts
        RoutingData: Updated routing data for local experts
        GatherIndx: Indices for gathering token features
        ScatterIndx: Indices for scattering token features
    """
    # Calculate number of experts per shard
    n_expts_local = routing_data.n_expts_tot // n_expt_shards
    
    # Create or retrieve cached routing simulation data
    if cache is None or n_global_rows not in cache:
        # Create a uniform routing matrix (all experts equally likely)
        weights = torch.ones(n_global_rows, routing_data.n_expts_tot, device=device)
        # Randomly select experts for each token (without replacement)
        expt_indx = torch.multinomial(weights, num_samples=routing_data.n_expts_act, replacement=False)

        # Sort each token's expert selections
        expt_indx, _ = expt_indx.sort(dim=1)

        # Compute histogram for local experts only
        hist = torch.histc(expt_indx, bins=routing_data.n_expts_tot, max=routing_data.n_expts_tot - 1)[:n_expts_local]

        # Count how many of each token's experts are local
        num_local_expts = (expt_indx < n_expts_local).sum(dim=1)
        # Count tokens that have at least one local expert, with alignment
        n_local_rows = (num_local_expts != 0).sum()
        n_local_rows = ((n_local_rows + row_align - 1) // row_align) * row_align

        # Sort tokens to prioritize those with local experts
        is_active = torch.argsort((num_local_expts == 0).to(torch.int8), stable=True)
        expt_indx = expt_indx[is_active].view(-1)
        
        # Create mapping indices for gather/scatter operations
        topk_indx = torch.argsort(expt_indx.view(-1), stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        
        # Filter out non-local experts from the mapping
        expt_indx = torch.where(expt_indx < n_expts_local, expt_indx, -1)
        gate_indx = torch.where(expt_indx == -1, -1, gate_indx)
        topk_indx = torch.where(gate_indx[topk_indx] == -1, -1, topk_indx)

        # Cache the computed data if a cache is provided
        if cache is not None:
            cache[n_global_rows] = hist, gate_indx, topk_indx, n_local_rows
    else:
        # Use cached data
        hist, gate_indx, topk_indx, n_local_rows = cache[n_global_rows]

    # Calculate expected tokens per expert for balanced sharding
    tokens_per_expt = ((n_global_rows // n_expt_shards) * routing_data.n_expts_act) // n_expts_local

    # Expand gate scaling factors for global rows
    # TODO: This currently adds a bogus "elementwise" kernel to the profile.
    gate_scal = routing_data.gate_scal.repeat_interleave(n_expt_shards, dim=0)

    # Return local row count, routing data, and gather/scatter indices
    return (
        n_local_rows,
        RoutingData(
            gate_scal,
            hist,
            n_expts_local,
            routing_data.n_expts_act,
            expected_tokens_per_expt=tokens_per_expt,
        ),
        GatherIndx(src_indx=topk_indx, dst_indx=gate_indx),
        ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx),
    )