# Memory Transfer Profiling in Proton

## Overview

This enhancement adds memory transfer (cudaMemcpy and variants) profiling to Proton, Triton's GPU profiler. With this feature, Proton can now track and profile memory operations between host and device, providing valuable insights into data transfer patterns and potential bottlenecks.

## Key Features

- **Track all memory transfers**: Host-to-Device, Device-to-Host, and Device-to-Device
- **Capture relevant metrics**: Timing, bytes transferred, transfer direction
- **Compatible with existing profiling**: Works alongside kernel execution profiling
- **Minimal overhead**: Leverages the same CUPTI mechanisms already in use

## Implementation

This enhancement required several changes to the CUPTI profiler implementation:

1. Enabling CUPTI memory copy activity tracking
2. Adding callbacks for CUDA memory transfer API calls
3. Implementing processing logic for memory copy activities
4. Extending metric conversion to include memory transfer data

For a detailed explanation of the implementation, see [memory_profiling.md](memory_profiling.md).

## Testing

A test script is provided in `/test/test_memcpy_profiling.py` to verify the functionality of memory transfer profiling. This script:

1. Performs various memory transfer operations (H2D, D2H, D2D)
2. Generates a profiling report
3. Verifies that the memory operations are properly captured

To run the test:

```bash
cd /path/to/triton
python -m third_party.proton.test.test_memcpy_profiling
```

## Example Usage

To use memory transfer profiling in your own code:

```python
import triton.profiler as proton

# Start profiling
session_id = proton.start("my_profile")

# Perform memory operations
host_data = torch.ones(1024, 1024)
device_data = host_data.cuda()  # This H2D transfer will be profiled
result = device_data * 2
host_result = result.cpu()  # This D2H transfer will be profiled

# End profiling
proton.finalize(session_id)

# View results (shows both kernel execution and memory transfers)
# proton-viewer -m time/ns,bytes/MB my_profile.hatchet
```

## Visualization

When viewing the profiling results, memory transfers will appear alongside kernel executions, with additional metrics:

- `time`: Duration of the memory transfer
- `bytes`: Amount of data transferred
- `direction`: Type of transfer (HostToDevice, DeviceToHost, etc.)

This allows you to see the complete picture of GPU application performance, including both computation and data movement.

## Future Enhancements

Potential future improvements include:

1. Driver API support for memory operations
2. Unified memory transfer tracking
3. Automatic bandwidth calculations
4. Transfer pattern analysis
5. Asynchronous transfer overlap visualization