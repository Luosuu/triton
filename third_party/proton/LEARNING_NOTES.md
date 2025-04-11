# Understanding Proton: A GPU Profiler for Triton

## Introduction

Proton is a lightweight GPU profiler designed specifically for Triton, a language and compiler focused on GPU programming. This document explores how Proton works internally, focusing on its CUPTI-based profiling capabilities and how it integrates with Triton.

## Architecture Overview

Proton's architecture is built around several key components:

1. **Python Interface Layer**: User-facing API for controlling profiling sessions
2. **Core Profiler Interface**: Abstract interface implemented by specific backends
3. **GPU-Specific Backends**: 
   - CUPTI-based implementation for NVIDIA GPUs
   - ROCTracer-based implementation for AMD GPUs
4. **Data Collection and Analysis**: Components for collecting and organizing profiling data
5. **Visualization Tools**: For interpreting and displaying the collected data

## Core Profiling Components

### Profiler Base Class

The `Profiler` class (`Profiler.h`) defines the basic interface for all profiler implementations:

- `start()`: Begins profiling session
- `flush()`: Flushes profiling data from device to host
- `stop()`: Stops profiling session
- `registerData()/unregisterData()`: Manages data collection targets

This base class follows the template method pattern, with subclasses implementing the abstract methods:
- `doStart()`
- `doFlush()`
- `doStop()`

### GPU Profiler Specialization

For GPU profiling, Proton provides the `GPUProfiler<T>` template class, which is specialized for different GPU backends:

- `CuptiProfiler`: For NVIDIA GPUs using CUPTI
- `RoctracerProfiler`: For AMD GPUs using ROCTracer

### CUPTI Profiler Implementation

The NVIDIA CUPTI implementation (`CuptiProfiler.cpp`) demonstrates how Proton leverages NVIDIA's profiling APIs:

1. **Profiler Initialization**:
   - Subscribes to CUPTI callbacks
   - Enables activity tracking
   - Sets up callback functions

2. **Callback Functions**:
   - Tracks CUDA API calls (kernel launches)
   - Correlates CUDA driver actions with user code
   - Handles GPU graph operations

3. **Activity Processing**:
   - Collects kernel execution data
   - Associates kernels with calling contexts
   - Tracks execution time and other metrics

4. **PC Sampling Support** (`CuptiPCSampling.h`):
   - Enables instruction-level sampling
   - Associates samples with source code
   - Provides deeper insights into kernel execution

## Integration with Triton

Proton integrates with Triton through a hook mechanism, allowing it to intercept kernel launches and collect metrics:

1. **Hook Registration**:
   - `register_triton_hook()` inserts hooks into Triton's kernel launch process
   - Hooks are installed in `CompiledKernel.launch_enter_hook` and `CompiledKernel.launch_exit_hook`

2. **Metadata Collection**:
   - Kernel launches are annotated with metadata through the `launch_metadata` function
   - Metadata includes operation counts (FLOPs) and memory transfer size (bytes)

3. **Scope and State Management**:
   - Proton tracks execution context using `scope` and `state` constructs
   - These provide hierarchical view of operations

## Data Flow in Proton

Proton's data flow follows these general steps:

1. **User Code Execution**:
   - Application executes Triton kernels within profiled regions
   - Hooks intercept kernel launches and create correlation points

2. **CUPTI Callbacks**:
   - CUDA API calls trigger CUPTI callbacks
   - Kernel execution activities are recorded
   - PC samples are collected (if enabled)

3. **Data Collection**:
   - Profiling data is buffered on GPU
   - Periodically flushed to host memory
   - Correlated with program context information

4. **Data Organization**:
   - Metrics are organized into a call tree
   - Data is aggregated by context (e.g., function, line number)
   - Additional metrics (like FLOP counts) are associated with operations

5. **Output Generation**:
   - Collected data is written to output files (usually JSON format)
   - Compatible with Hatchet for visualization and analysis

## CUPTI Integration Details

Proton's integration with NVIDIA CUPTI is particularly interesting:

1. **Kernel Activity Tracking**:
   - Uses `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` for regular profiling
   - Uses `CUPTI_ACTIVITY_KIND_KERNEL` with PC sampling

2. **Correlation System**:
   - Maps CUPTI correlation IDs to Proton's internal IDs
   - Handles asynchronous kernel execution
   - Properly attributes metrics to the correct calling context

3. **CUDA Graph Support**:
   - Tracks graph creation and execution
   - Maps graph executions back to creation contexts
   - Handles repeated executions of the same graph

4. **PC Sampling**:
   - Sets up PC sampling configuration
   - Processes sampled program counter values
   - Associates samples with source code using CUDA debugging information

## Advanced Features

### PC Sampling

PC sampling provides instruction-level insights by sampling the program counter during kernel execution:

1. **Setup**:
   - Configures sampling period, buffer sizes
   - Enables stall reason collection

2. **Module Tracking**:
   - Loads debugging information from CUDA modules
   - Maps PC values to source code locations

3. **Sample Processing**:
   - Collects samples during kernel execution
   - Aggregates them by instruction and stall reason
   - Attributes them to source code locations

### Scope and State Management

Proton provides two methods for annotating execution context:

1. **Scope**:
   - Hierarchical call tree structure
   - Nests operations within parent scopes
   - Allows custom metrics to be attached

2. **State**:
   - Non-hierarchical context annotation
   - Overrides rather than nests
   - Useful for tagging operations with additional metadata

## Comparison with Other Profilers

Proton offers some advantages over tools like NVIDIA Nsight:

1. **Lower Overhead**:
   - Targets specific profiling needs for Triton
   - Less overhead than full-featured profilers

2. **Triton Integration**:
   - Direct integration with Triton's execution model
   - Access to Triton-specific metadata

3. **Portability**:
   - Works on both NVIDIA and AMD GPUs
   - Consistent interface across platforms

4. **File Size**:
   - More compact output files
   - Aggregates metrics rather than recording every event

## Performance Considerations

Proton is designed to minimize profiling overhead:

1. **Selective Callbacks**:
   - Only enables necessary CUPTI callbacks
   - Focuses on kernel execution events

2. **Efficient Buffering**:
   - Uses large buffers to reduce host-device synchronization
   - Avoids unnecessary data transfers

3. **Correlation System**:
   - Efficiently tracks relationships between API calls and kernel executions
   - Avoids expensive lookups during critical sections

4. **PC Sampling Trade-offs**:
   - PC sampling has higher overhead (~20x mentioned in docs)
   - Not enabled by default, only when explicitly requested

## Conclusion

Proton represents a specialized profiling solution tailored to the needs of Triton programming. Its integration with CUPTI provides detailed insights into GPU kernel execution while maintaining reasonable overhead.

The design allows it to collect comprehensive performance data, attribute it correctly to source code locations, and present it in a way that helps developers optimize their Triton kernels for maximum performance.

By understanding how Proton works internally, developers can better leverage its capabilities to diagnose performance issues and optimize GPU code written with Triton.