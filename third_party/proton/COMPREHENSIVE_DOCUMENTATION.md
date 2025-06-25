# Triton Proton Profiler: Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [Comprehensive API Reference](#comprehensive-api-reference)
4. [Command Line API](#command-line-api)
5. [Profile Comparison and Analysis](#profile-comparison-and-analysis)
6. [Implementation Details](#implementation-details)
7. [Getting Started](#getting-started)
8. [Advanced Usage](#advanced-usage)
   - [Custom Metrics](#custom-metrics)
   - [Multi-Session Profiling](#multi-session-profiling)
   - [State-Based Profiling](#state-based-profiling)
   - [Multi-GPU Profiling](#multi-gpu-profiling)
   - [Time-Based Profiling](#time-based-profiling)
   - [PC Sampling Analysis](#pc-sampling-analysis)
9. [Development Guide](#development-guide)
10. [Troubleshooting](#troubleshooting)
11. [Limitations and Future Features](#limitations-and-future-features)

## Overview

The Triton Proton profiler is a high-performance, low-overhead GPU profiling system specifically designed for deep learning workloads. It provides comprehensive profiling capabilities for both NVIDIA (CUDA/CUPTI) and AMD (HIP/ROCtracer) GPUs, with particular strength in Triton kernel analysis.

### Key Features

- **Low Overhead**: ~1.5x overhead compared to NSight Systems' higher overhead
- **Multi-Backend Support**: NVIDIA CUPTI and AMD ROCtracer backends
- **Context-Aware Profiling**: Python and shadow context tracking
- **Hierarchical Data**: Tree-based aggregation for smaller profile sizes
- **Triton Integration**: Deep integration with Triton kernels including metadata collection
- **Flexible API**: Multiple profiling modes (scope, hook, instrumentation)
- **Rich Metrics**: Hardware counters, PC sampling, custom metrics
- **Visualization**: Hatchet-based profile visualization and analysis
- **Time-Based Profiling**: Support for periodic profiling windows ideal for long-running services

### Design Philosophy

Proton follows several key design principles:

1. **Performance First**: Minimal overhead profiling suitable for production workloads
2. **Modularity**: Plugin-based architecture supporting multiple backends
3. **Correctness**: Thread-safe design with proper synchronization
4. **Usability**: Simple Python API with powerful underlying capabilities
5. **Extensibility**: Template-based design allowing easy addition of new features

## Architecture and Design

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python API    │    │   MLIR Dialect  │    │   CLI Tools     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ profile.py      │    │ ProtonOps.td    │    │ proton          │
│ scope.py        │    │ Passes          │    │ proton-viewer   │
│ state.py        │    │ Conversion      │    │                 │
│ hook.py         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    C++ Core Layer                                  │
├─────────────────────────────────┼─────────────────────────────────┤
│                                 │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │
│  │  Session    │  │  Context    │  │         Data                │ │
│  │ Management  │  │  System     │  │     Management              │ │
│  │             │  │             │  │                             │ │
│  │ - Manager   │  │ - Sources   │  │ - TreeData                  │ │
│  │ - Sessions  │  │ - Shadow    │  │ - Metrics                   │ │
│  │ - Lifecycle │  │ - Python    │  │ - Aggregation               │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Profiler System                         │   │
│  │                                                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │   │
│  │  │    CUPTI    │  │ ROCtracer   │  │   GPU Profiler      │ │   │
│  │  │   Backend   │  │   Backend   │  │     Template        │ │   │
│  │  │             │  │             │  │                     │ │   │
│  │  │ - Kernel    │  │ - HIP API   │  │ - Correlation       │ │   │
│  │  │ - PC Sample │  │ - Timing    │  │ - Threading         │ │   │
│  │  │ - Metrics   │  │ - Metrics   │  │ - Interface         │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### Session Management System

The session management system is built around the `SessionManager` singleton that coordinates multiple profiling sessions. Each session encapsulates:

- **Profiler Backend**: CUPTI, ROCtracer, or custom profiler
- **Context Source**: Python stack or shadow context tracking
- **Data Storage**: Tree or trace-based data structures
- **Hook Integration**: Optional Triton or custom hooks

**Key Design Patterns**:
- **Singleton Pattern**: Global session coordination
- **RAII**: Automatic resource management
- **Template Metaprogramming**: Compile-time interface selection

#### Context System

The context system provides call stack tracking through two main implementations:

**Shadow Context Source**:
- Thread-local context stacks with main thread inheritance
- Designed for multi-threaded scenarios (PyTorch backward passes)
- Manual scope management with balanced entry/exit

**Python Context Source**:
- Automatic Python call stack unwinding
- Uses Python C API for frame inspection
- Captures file, function, and line information

#### Data Management

**TreeData Structure**:
- Hierarchical tree with path-based node construction
- Context-based deduplication
- Metric aggregation with inclusive/exclusive semantics

**Metric System**:
- **Flexible Metrics**: User-defined with configurable aggregation
- **Kernel Metrics**: GPU execution timing and device info
- **PC Sampling Metrics**: Detailed stall reason analysis

#### GPU Backend Integration

**CUPTI Backend**:
- Kernel timing and hardware counters
- PC sampling for detailed stall analysis
- Activity buffer management for asynchronous data collection

**ROCtracer Backend**:
- HIP API tracing and kernel timing
- Activity buffer processing
- Device visibility management

### Threading Model

Proton uses a sophisticated threading model designed for minimal overhead:

1. **Thread-Local Storage**: Per-thread state for contexts and operations
2. **Shared-Read Locks**: Reader-writer locks for read-heavy data structures
3. **Lock-Free Paths**: Atomic operations for correlation tracking
4. **Coarse-Grained Locking**: Single session manager mutex for simplicity

## Comprehensive API Reference

### Python Public API

#### Core Profiling Functions

##### `proton.start(name=None, *, context="shadow", data="tree", backend=None, hook=None)`

Start a profiling session with specified configuration.

**Parameters**:
- `name` (str, optional): Profile name/path (default: "proton")
- `context` (str): Context source type - "shadow" | "python" (default: "shadow")
- `data` (str): Data structure type - "tree" (default: "tree")
- `backend` (str): Profiling backend - "cupti" | "cupti_pcsampling" | "roctracer" | None (auto-detect)
- `hook` (str): Hook type - "triton" | None

**Returns**: Session ID (int) - Unique identifier for this profiling session

**Example**:
```python
session_id = proton.start("my_profile", backend="cupti", hook="triton")
```

**Multi-GPU Example**:
```python
# Each GPU/rank gets its own session
rank = int(os.environ.get('RANK', 0))
session_id = proton.start(f"profile_rank_{rank}", backend="cupti")
# session_id is unique per start() call, allowing independent control
```

##### `proton.activate(session=None)`

Activate profiling session to start data recording.

**Parameters**: `session` (int, optional): Session ID (None = all sessions)

##### `proton.deactivate(session=None)`

Deactivate profiling session to stop data recording.

**Parameters**: `session` (int, optional): Session ID (None = all sessions)

##### `proton.finalize(session=None, output_format="hatchet")`

Finalize session and write profiling data to file.

**Parameters**:
- `session` (int, optional): Session ID (None = all sessions)
- `output_format` (str): Output format - "hatchet" (default)

##### `@proton.profile(func=None, *, name=None, context="shadow", data="tree", backend=None, hook=None)`

Decorator for automatic profiling of functions.

**Example**:
```python
@proton.profile
def my_function():
    pass

@proton.profile(name="custom_profile", backend="cupti")
def my_function():
    pass
```

#### Scope Management

##### `proton.scope(name, metrics=None)`

Context manager/decorator for named profiling scopes.

**Parameters**:
- `name` (str): Scope name
- `metrics` (dict[str, Union[int, float]], optional): Custom metrics to record at scope entry

**Examples**:
```python
# Context manager
with proton.scope("compute_phase", {"custom_metric": 42}):
    compute()

# Decorator
@proton.scope("compute_phase")
def compute():
    pass
```

**Note**: The `scope` context manager records metrics at entry time. For metrics that are only available at the end of a scope (like percentiles or final counts), use `enter_scope()`/`exit_scope()` with exit metrics instead.

##### `proton.cpu_timed_scope(name, metrics=None)`

Scope that automatically measures CPU elapsed time.

**Parameters**: Same as `scope()` plus automatic "cpu_time (ns)(exc)" metric

**Constraint**: Cannot use "cpu_time" as custom metric name

##### `proton.enter_scope(name, *, triton_op=False, metrics=None)`

Manually enter a named scope.

**Parameters**:
- `name` (str): Scope name
- `triton_op` (bool): Whether this is a Triton operation (default: False)
- `metrics` (dict, optional): Custom metrics

**Returns**: Scope ID (int)

##### `proton.exit_scope(triton_op=False, metrics=None)`

Exit the most recent scope with optional end-of-scope metrics.

**Parameters**:
- `triton_op` (bool): Must match `enter_scope()` call
- `metrics` (dict, optional): Metrics to record at scope exit (e.g., percentiles, throughput)

**Returns**: Scope ID (int)

**Important**: Exit metrics are perfect for recording values that are only known at the end of a measurement period, such as:
- Percentile latencies (p50, p90, p99)
- Total throughput/goodput
- Success/error rates
- Final aggregated statistics

**Example**:
```python
import proton
import numpy as np

# Track request latencies during a time window
latencies = []
scope_id = proton.enter_scope("service_window")

# Process requests...
for request in requests:
    start = time.time()
    process_request(request)
    latencies.append(time.time() - start)

# Calculate end-of-window metrics
proton.exit_scope(metrics={
    "p50_latency_ms": np.percentile(latencies, 50) * 1000,
    "p90_latency_ms": np.percentile(latencies, 90) * 1000,
    "p99_latency_ms": np.percentile(latencies, 99) * 1000,
    "throughput_rps": len(latencies) / window_duration,
    "success_rate": success_count / len(latencies)
})
```

#### State Management

##### `proton.state(name)`

Context manager/decorator for profiling states.

**Parameters**: `name` (str): State name

**Examples**:
```python
# Context manager
with proton.state("inference"):
    model(input)

# Decorator
@proton.state("training")
def train_step():
    pass
```

##### `proton.enter_state(name)` / `proton.exit_state()`

Manual state entry/exit functions.

#### Hook System

The hook system provides automatic integration with frameworks like Triton.

##### Triton Hook

**`register_triton_hook()` / `unregister_triton_hook()`**: Register/unregister Triton kernel launch hooks

**Automatic Metrics**: `flops8`, `flops16`, `flops32`, `flops64`, `bytes`, `flops`

**Usage**: Automatically called when `hook="triton"` in `start()`

#### Performance Specifications

##### `proton.max_flops(device_type, arch, width, num_sms, clock_rate)`

Calculate theoretical maximum FLOPS for device.

**Parameters**:
- `device_type` (str): "CUDA" | "HIP"
- `arch` (str): Architecture ("80", "89", "90", "100", "gfx90a", "gfx942", "gfx950")
- `width` (int): Bit width (8, 16, 32, 64)
- `num_sms` (int): Number of streaming multiprocessors
- `clock_rate` (float): Clock rate in GHz

**Returns**: Maximum FLOPS (float)

##### `proton.max_bps(device_type, arch, bus_width, memory_clock_rate)`

Calculate theoretical maximum memory bandwidth.

**Parameters**:
- `device_type` (str): "CUDA" | "HIP"
- `arch` (str): Architecture string
- `bus_width` (int): Memory bus width in bits
- `memory_clock_rate` (float): Memory clock rate in GHz

**Returns**: Maximum bytes per second (float)

#### Viewer/Analysis

##### `proton.read(filename)`

Read and parse profile data file.

**Parameters**: `filename` (str): Path to .hatchet file

**Returns**: `(gf, inclusive_metrics, exclusive_metrics, device_info)` tuple

##### `proton.parse(metrics, filename, include=None, exclude=None, threshold=None)`

Parse profile data with filtering and metric derivation.

**Parameters**:
- `metrics` (list[str]): Metrics to analyze
- `filename` (str): Profile file path
- `include/exclude` (str, optional): Regex patterns for frame filtering
- `threshold` (float, optional): Minimum metric threshold

**Returns**: `(gf, derived_metrics)` tuple

**Derived Metrics Available**:
- Time metrics: `time/s`, `time/ms`, `time/us`, `time/ns`, `avg_time/*`
- FLOPS metrics: `flop/s`, `gflop/s`, `tflop/s`, `flop{8,16,32,64}/s`
- Bandwidth: `byte/s`, `gbyte/s`, `tbyte/s`
- Utilization: `util` (combined FLOPS and memory utilization)
- Percentage: `<metric>/%` (percentage of total for metric)

#### Context Management

##### `proton.depth(session=0)`

Get the current context depth for debugging and validation.

**Parameters**: `session` (int, optional): Session ID (default: 0)

**Returns**: `int` or `None` - Current context depth, or None if profiling is off

**Usage**:
```python
import proton

# Check context depth during profiling
depth = proton.depth()
if depth is not None:
    print(f"Current context depth: {depth}")
```

### C++ API

#### Core Context System

##### Context Classes

**`struct Context`**:
- **Members**: `std::string name`
- **Methods**: Comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`)

**`class ContextSource`** (Abstract):
- **`std::vector<Context> getContexts()`**: Get current context stack
- **`void setState(std::optional<Context>)`**: Set current state
- **`virtual size_t getDepth() = 0`**: Get context depth

**`struct Scope : public Context`**:
- **Members**: `size_t scopeId`, `static std::atomic<size_t> scopeIdCounter`
- **`static size_t getNewScopeId()`**: Generate new scope ID

##### Interface Classes

**`class ScopeInterface`** (Abstract):
- **`virtual void enterScope(const Scope &scope) = 0`**
- **`virtual void exitScope(const Scope &scope) = 0`**

**`class OpInterface`** (Abstract):
- **`void enterOp(const Scope &scope)`**: Thread-safe operation entry
- **`void exitOp(const Scope &scope)`**: Thread-safe operation exit

#### Data Management

##### Data Interface

**`class Data : public ScopeInterface`** (Abstract):
- **Constructor**: `Data(const std::string &path, ContextSource *contextSource)`
- **`virtual size_t addOp(size_t scopeId, const std::string &opName = {}) = 0`**
- **`virtual void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) = 0`**
- **`virtual void addMetrics(size_t scopeId, const std::map<std::string, MetricValueType> &metrics) = 0`**
- **`void dump(OutputFormat format)`**: Write data to file

##### Metric System

**`using MetricValueType = std::variant<uint64_t, int64_t, double, std::string>`**

**`class Metric`** (Abstract Base):
- **Constructor**: `Metric(MetricKind kind, size_t size)`
- **`virtual const std::string getName() const = 0`**
- **`virtual const std::string getValueName(int valueId) const = 0`**
- **`virtual bool isProperty(int valueId) const = 0`**: Non-aggregatable values
- **`virtual bool isExclusive(int valueId) const = 0`**: Non-propagatable values
- **`void updateValue(int valueId, MetricValueType value)`**: Update specific value
- **`void updateMetric(Metric &other)`**: Merge with another metric

#### Session Management

**`class Session`**:
- **`void activate()`**: Start data collection
- **`void deactivate()`**: Stop data collection
- **`void finalize(OutputFormat format)`**: Write results
- **`size_t getContextDepth()`**: Get current depth

**`class SessionManager : public Singleton<SessionManager>`**:
- **`size_t addSession(...)`**: Create new session
- **`void finalizeSession(size_t sessionId, OutputFormat format)`**
- **`void activateSession(size_t sessionId)` / `deactivateSession(size_t sessionId)`**
- **`void enterScope(const Scope &scope)` / `exitScope(const Scope &scope)`**: Global scope management

#### Profiler System

**`class Profiler`** (Abstract):
- **`Profiler *start()`**: Start profiling
- **`Profiler *flush()`**: Flush device data to host
- **`Profiler *stop()`**: Stop profiling
- **`Profiler *registerData(Data *data)` / `unregisterData(Data *data)`**: Manage data connections

**`template <typename ConcreteProfilerT> class GPUProfiler : public Profiler, public ThreadLocalOpInterface, public Singleton<ConcreteProfilerT>`**:
- **`ConcreteProfilerT &enablePCSampling()` / `disablePCSampling()`**
- **`ConcreteProfilerT &setLibPath(const std::string &libPath)`**: Set profiler library path

### MLIR Dialect API

#### Proton Dialect

**Dialect Definition**: `mlir::triton::proton`

##### Operations

**`proton.record`** (`TT_RecordOp`): Record GPU hardware events from performance counters

**Attributes**:
- `isStart` (BoolAttr): Whether starting or ending measurement
- `regionId` (I32Attr): Region identifier (non-negative)
- `metric` (MetricAttr): Metric type (default: "cycle")
- `granularity` (GranularityAttr): Measurement granularity (default: "warpgroup")

**Assembly Format**: `proton.record() {isStart = true, regionId = 4 : i32}`

##### Attributes

**`MetricAttr`**: I32 Enum - Values: `CYCLE` (0) → "cycle"

**`GranularityAttr`**: I32 Enum - Values: `WARPGROUP` (0) → "warpgroup", `WARP` (1) → "warp"

### Command Line API

#### Proton Profiler CLI

**Command**: `proton [options] script.py [script_args]`

**Global Options**:
- `-n, --name NAME`: Profile session name
- `-b, --backend {cupti,cupti_pcsampling,roctracer}`: Profiling backend
- `-c, --context {shadow,python}`: Context source (default: shadow)
- `-d, --data {tree}`: Data structure (default: tree)
- `-k, --hook {triton}`: Profiling hook
- `-i, --instrument {print-mem-spaces}`: Instrumentation analysis

**Examples**:
```bash
proton -n my_profile -b cupti script.py
proton -k triton --instrument print-mem-spaces test.py
```

#### Proton Viewer CLI

**Command**: `proton-viewer [options] profile.hatchet`

**Options**:
- `-m, --metrics METRICS`: Comma-separated metrics to display
- `-i, --include REGEX`: Include frames matching regex
- `-e, --exclude REGEX`: Exclude frames matching regex
- `-t, --threshold VALUE`: Exclude frames below threshold
- `-d, --depth VALUE`: The depth of the tree to display (default: 100)
- `-f, --format FORMAT`: Frame name formatting (full, file_function_line, function_line, file_function)
- `--print-sorted`: Sort output by metric value instead of chronologically
- `--diff-profile FILE`: Compare two profiles
- `-l, --list`: List available metrics

**Examples**:
```bash
# Basic usage
proton-viewer -m time,flops,util profile.hatchet

# Compare two profiles
proton-viewer -m time --diff-profile baseline.hatchet optimized.hatchet

# Filter and sort results
proton-viewer -m time/ms -t 0.01 --print-sorted profile.hatchet

# Format frame names
proton-viewer -m time -f function_line profile.hatchet
```

## Profile Comparison and Analysis

### Comparing Profiles (diff)

The Proton viewer supports profile comparison through the `--diff-profile` option, which computes the difference between two profiles:

```bash
proton-viewer -m time --diff-profile file1.hatchet file2.hatchet
```

This command computes: `file2[metric] - file1[metric]` for the specified metrics.

**Use Cases**:
- **Performance Regression Testing**: Compare before/after optimization profiles
- **A/B Testing**: Compare different implementation approaches
- **Hardware Comparison**: Compare performance across different GPUs

**Example Workflow**:
```bash
# 1. Capture baseline profile
proton -n baseline script.py

# 2. Make optimizations and capture new profile
proton -n optimized script.py

# 3. Compare the profiles
proton-viewer -m time/ms --diff-profile baseline.hatchet optimized.hatchet

# Positive values indicate the optimized version takes more time
# Negative values indicate the optimized version is faster
```

### Advanced Filtering and Analysis

#### Include/Exclude Patterns

Use regular expressions to focus on specific parts of your code:

```bash
# Include only kernel-related frames
proton-viewer -i ".*kernel.*" profile.hatchet

# Exclude internal/system functions
proton-viewer -e ".*internal.*" profile.hatchet
```

#### Threshold Filtering

Filter out noise by excluding frames below a performance threshold:

```bash
# Show only frames taking > 1% of total time
proton-viewer -m time/% -t 1.0 profile.hatchet

# Show only kernels with > 100 GFLOPS
proton-viewer -m gflop/s -t 100 profile.hatchet
```

#### Sorted Output

View top performance bottlenecks:

```bash
# Show kernels sorted by execution time
proton-viewer -m time/ms --print-sorted profile.hatchet

# Show kernels sorted by FLOPS utilization
proton-viewer -m util --print-sorted profile.hatchet
```

## Implementation Details

### Session Management Implementation

The session management system uses a singleton pattern with careful thread safety considerations:

```cpp
class SessionManager : public Singleton<SessionManager> {
private:
  mutable std::mutex mutex;  // Protects all session state
  std::map<size_t, std::unique_ptr<Session>> sessions;
  std::map<std::string, size_t> sessionPaths;
  std::map<size_t, bool> sessionActive;
};
```

**Key Features**:
- **RAII Session Management**: Sessions managed via `std::unique_ptr`
- **Coarse-Grained Locking**: Single mutex for simplicity and correctness
- **Interface Counting**: Reference counting for shared interfaces
- **Batch Operations**: Coordinated session activation/deactivation

### Context System Implementation

#### Shadow Context Source

Designed for multi-threaded scenarios with thread-local inheritance:

```cpp
class ShadowContextSource : public ContextSource, public ScopeInterface {
private:
  std::vector<Context> *mainContextStack{};
  static thread_local std::map<ShadowContextSource *, bool> threadContextInitialized;
  static thread_local std::map<ShadowContextSource *, std::vector<Context>> threadContextStack;
};
```

**Algorithm**:
1. Main thread initializes master context stack
2. Worker threads copy main thread's stack on first access
3. Each thread maintains shadow stack for local scopes
4. Context inheritance preserves parent-child relationships

#### Python Context Source

Uses Python C API for automatic stack unwinding:
- `PyEval_GetFrame()` to capture current execution frame
- `PyFrame_GetBack()` to walk the call stack
- Extracts file, function, and line information
- Handles Python version compatibility

### Data Collection and Aggregation

#### TreeData Structure

Core data structure for hierarchical profiling:

```cpp
struct TreeNode : public Context {
  size_t id, parentId;
  std::map<Context, size_t> children;
  std::map<MetricKind, std::shared_ptr<Metric>> metrics;
  std::map<std::string, FlexibleMetric> flexibleMetrics;
};
```

**Tree Construction Algorithm**:
1. Root node always created with ID 0
2. Path-based insertion creates hierarchical structure
3. Deduplication via existing node lookup
4. Metric aggregation at specific tree nodes

#### Metric Aggregation

Three types of metrics with different aggregation semantics:

1. **Inclusive**: Aggregated by addition, propagated to parents
2. **Exclusive**: Aggregated locally, not propagated (marked with "(exc)")
3. **Property**: Not aggregated, represents metadata (marked with "(pty)")

### GPU Backend Implementation

#### Template-Based Architecture

```cpp
template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler, public ThreadLocalOpInterface, 
                    public Singleton<ConcreteProfilerT> {
  std::unique_ptr<GPUProfilerPimplInterface> pImpl;
};
```

**Benefits**:
- **Pimpl Idiom**: Hides backend-specific headers
- **CRTP Pattern**: Compile-time polymorphism
- **Clean Abstraction**: Common interface for different backends

#### Correlation Tracking

Multi-level correlation system for GPU event attribution:

```cpp
struct Correlation {
  std::atomic<uint64_t> maxSubmittedCorrelationId{0};
  std::atomic<uint64_t> maxCompletedCorrelationId{0};
  CorrIdToExternIdMap corrIdToExternId;
  static thread_local std::deque<size_t> externIdQueue;
};
```

**Algorithm**:
1. **Submit Phase**: Track launched operations
2. **Context Mapping**: Map correlation IDs to external IDs
3. **Completion Tracking**: Mark finished operations
4. **Flush Synchronization**: Wait for completion with retry

### Threading and Synchronization

#### Thread Safety Mechanisms

1. **Coarse-Grained Locking**: SessionManager uses single mutex
2. **Reader-Writer Locks**: `std::shared_mutex` for read-heavy workloads
3. **Lock-Free Atomics**: Correlation tracking uses `std::atomic`
4. **Thread-Local Storage**: Eliminates synchronization overhead

#### Double-Checked Locking

```cpp
template <typename Condition, typename Function>
void doubleCheckedLock(Condition enterCondition, std::mutex &lock, Function function) {
  if (!enterCondition()) return;          // Fast path
  std::unique_lock<std::mutex> guard(lock);
  if (!enterCondition()) return;          // Recheck under lock
  function();                             // Critical section
}
```

### Performance Optimizations

#### Low Overhead Techniques

1. **Thread-Local Storage**: Eliminates synchronization overhead
2. **Template Metaprogramming**: Zero-cost abstractions
3. **Lock-Free Fast Paths**: Atomic operations for hot paths
4. **Cache-Friendly Structures**: Compact representations

#### Batching and Buffering

1. **Event Batching**: GPU events processed in batches
2. **Metric Aggregation**: Local accumulation before global updates
3. **Deferred Operations**: Expensive operations delayed until necessary

## Getting Started

### Installation

Proton is included as part of the Triton installation. To use it:

```bash
pip install triton
```

### Basic Usage

#### Simple Function Profiling

```python
import triton
import proton

@proton.profile
def my_function():
    # Your GPU code here
    pass

my_function()
```

#### Manual Session Management

```python
import proton

# Start profiling session
session_id = proton.start("my_profile", backend="cupti", hook="triton")
proton.activate(session_id)

# Your code here
with proton.scope("computation"):
    # GPU kernels and operations
    pass

# Finalize profiling
proton.deactivate(session_id)
proton.finalize(session_id)
```

#### Scope-Based Profiling

```python
import proton

with proton.profile("detailed_profile"):
    with proton.scope("preprocessing"):
        preprocess_data()
    
    with proton.scope("inference", {"batch_size": 32}):
        model(input)
    
    with proton.scope("postprocessing"):
        postprocess_results()
```

#### Profiling with Exit Metrics

```python
import proton
import time

# Start profiling
session = proton.start("service_profile")
proton.activate(session)

# Track metrics for a batch processing window
batch_latencies = []
batch_sizes = []
scope_id = proton.enter_scope("batch_processing_window")

# Process batches
for batch in batches:
    start = time.time()
    results = process_batch(batch)
    batch_latencies.append(time.time() - start)
    batch_sizes.append(len(batch))

# Record comprehensive metrics at window end
proton.exit_scope(metrics={
    "total_batches": len(batch_latencies),
    "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
    "p95_latency_sec": np.percentile(batch_latencies, 95),
    "p99_latency_sec": np.percentile(batch_latencies, 99),
    "throughput_items_per_sec": sum(batch_sizes) / sum(batch_latencies)
})

proton.finalize(session)
```

### Command Line Usage

#### Profile a Script

```bash
proton -n gpu_profile script.py
```

#### Profile with Triton Hook

```bash
proton -k triton -n triton_profile training_script.py
```

#### View Profile Results

```bash
proton-viewer -m time,flops,util gpu_profile.hatchet
```

## Advanced Usage

### Custom Metrics

```python
import proton

with proton.scope("custom_computation", {
    "input_size": 1024,
    "algorithm": "optimized",
    "memory_usage": 2048
}):
    # Your computation
    pass
```

### Multi-Session Profiling

```python
import proton

# Start multiple sessions
session1 = proton.start("detailed", backend="cupti_pcsampling")
session2 = proton.start("overview", backend="cupti")

# Activate both
proton.activate()  # Activates all sessions

# Your code with dual profiling
compute()

# Finalize separately
proton.finalize(session1)
proton.finalize(session2)
```

### State-Based Profiling

```python
import proton

@proton.state("training")
def train_epoch():
    for batch in dataloader:
        with proton.state("forward"):
            loss = model(batch)
        
        with proton.state("backward"):
            loss.backward()
        
        with proton.state("optimizer"):
            optimizer.step()
```

### Multi-GPU Profiling

Proton supports profiling multi-GPU applications with separate sessions for each GPU/rank. This allows you to:
- Create independent profiling sessions per GPU
- Save profiles with rank-based naming
- Analyze performance across different GPUs
- Debug GPU-specific performance issues

#### Basic Multi-GPU Profiling

```python
import proton
import torch
import os

# Get rank information (example with PyTorch distributed)
rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))

# Set the device for this rank
torch.cuda.set_device(local_rank)

# Create a profiling session with rank-based naming
session_id = proton.start(f"profile_rank_{rank}", backend="cupti")

# Activate profiling for this specific session
proton.activate(session_id)

# Your GPU computation
model = MyModel().cuda(local_rank)
output = model(input_data)

# Finalize the session
proton.finalize(session_id)
```

#### Advanced Multi-GPU with Multiple Sessions

```python
import proton
import torch.distributed as dist

class MultiGPUProfiler:
    def __init__(self, base_name="profile"):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.sessions = {}
        self.base_name = base_name
        
    def create_session(self, phase_name, backend="cupti"):
        """Create a profiling session for a specific phase"""
        session_name = f"{self.base_name}_rank{self.rank}_{phase_name}"
        session_id = proton.start(session_name, backend=backend)
        self.sessions[phase_name] = session_id
        return session_id
    
    def profile_phase(self, phase_name):
        """Context manager for profiling a specific phase"""
        class PhaseProfiler:
            def __init__(self, profiler, phase):
                self.profiler = profiler
                self.phase = phase
                self.session_id = None
                
            def __enter__(self):
                self.session_id = self.profiler.create_session(self.phase)
                proton.activate(self.session_id)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                proton.deactivate(self.session_id)
                proton.finalize(self.session_id)
                
        return PhaseProfiler(self, phase_name)

# Usage
profiler = MultiGPUProfiler("training_run")

# Profile different phases separately
with profiler.profile_phase("data_loading"):
    data = load_data()

with profiler.profile_phase("forward_pass"):
    output = model(data)

with profiler.profile_phase("backward_pass"):
    loss.backward()
```

#### Multi-GPU Profiling with Horovod

```python
import proton
import horovod.torch as hvd

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# Create rank-specific profile
session_id = proton.start(
    f"horovod_profile_rank{hvd.rank()}_size{hvd.size()}",
    backend="cupti",
    hook="triton"
)

proton.activate(session_id)

# Your distributed training code
model = MyModel().cuda()
optimizer = hvd.DistributedOptimizer(optimizer)

for epoch in range(num_epochs):
    with proton.scope(f"epoch_{epoch}"):
        for batch_idx, (data, target) in enumerate(train_loader):
            with proton.scope(f"batch_{batch_idx}"):
                # Training step
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

proton.finalize(session_id)
```

#### Comparing Multi-GPU Profiles

After collecting profiles from multiple GPUs, you can compare them:

```bash
# Compare GPU 0 and GPU 1 performance
proton-viewer -m time/ms --diff-profile profile_rank_0.hatchet profile_rank_1.hatchet

# View all GPU profiles side by side
for rank in {0..7}; do
    echo "=== Rank $rank ==="
    proton-viewer -m time/ms,util --print-sorted profile_rank_$rank.hatchet | head -20
done
```

#### Best Practices for Multi-GPU Profiling

1. **Synchronization Points**: Add explicit synchronization before profiling critical sections:
   ```python
   torch.cuda.synchronize()
   with proton.scope("critical_kernel"):
       kernel_computation()
   torch.cuda.synchronize()
   ```

2. **Memory Profiling**: Track memory usage per GPU:
   ```python
   with proton.scope("memory_intensive_op", {
       "allocated_memory_mb": torch.cuda.memory_allocated() / 1e6,
       "reserved_memory_mb": torch.cuda.memory_reserved() / 1e6,
       "gpu_id": local_rank
   }):
       large_tensor_operation()
   ```

3. **Collective Operations**: Profile communication overhead:
   ```python
   # Profile all-reduce operations
   with proton.scope("all_reduce", {"size_mb": tensor.numel() * 4 / 1e6}):
       dist.all_reduce(tensor)
   ```

4. **Load Balancing Analysis**: Use session IDs to track work distribution:
   ```python
   # Track work items per GPU
   work_items = get_work_for_rank(rank)
   session_id = proton.start(f"workload_rank{rank}_items{len(work_items)}")
   
   for item in work_items:
       with proton.scope(f"process_item_{item.id}"):
           process_item(item)
   ```

#### Automated Multi-GPU Profiling Script

```python
#!/usr/bin/env python
import proton
import torch
import argparse
from datetime import datetime

def profile_multi_gpu_job(args):
    # Setup
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    
    # Create detailed profile name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_name = f"{args.profile_dir}/{args.job_name}_rank{rank}_of_{world_size}_{timestamp}"
    
    # Configure profiling based on rank
    if rank == 0 and args.detailed_on_rank_0:
        # More detailed profiling on rank 0
        backend = "cupti_pcsampling"
    else:
        # Standard profiling on other ranks
        backend = "cupti"
    
    # Start profiling
    session_id = proton.start(profile_name, backend=backend, hook="triton")
    
    try:
        proton.activate(session_id)
        
        # Run the actual job
        run_distributed_job(args)
        
    finally:
        proton.deactivate(session_id)
        proton.finalize(session_id)
        
        if rank == 0:
            print(f"Profiles saved to {args.profile_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", default="distributed_training")
    parser.add_argument("--profile-dir", default="./profiles")
    parser.add_argument("--detailed-on-rank-0", action="store_true")
    args = parser.parse_args()
    
    profile_multi_gpu_job(args)
```

### Time-Based Profiling

For long-running services like LLM inference servers, you often need to collect profiles periodically to monitor performance over time windows. Proton supports this through session management, allowing you to create separate sessions for each time window.

#### Basic Time-Window Profiling

```python
import proton
import time
import threading
from datetime import datetime

class TimeWindowProfiler:
    def __init__(self, window_duration_seconds=300, output_dir="./profiles"):
        self.window_duration = window_duration_seconds
        self.output_dir = output_dir
        self.current_session = None
        self.window_count = 0
        self.profiling = False
        
    def start_window(self):
        """Start a new profiling window"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_name = f"{self.output_dir}/profile_window_{self.window_count}_{timestamp}"
        
        # Create new session for this time window
        self.current_session = proton.start(profile_name, backend="cupti", hook="triton")
        proton.activate(self.current_session)
        self.window_count += 1
        self.profiling = True
        
    def end_window(self):
        """End current profiling window and save data"""
        if self.profiling and self.current_session is not None:
            proton.deactivate(self.current_session)
            proton.finalize(self.current_session)
            self.profiling = False
            self.current_session = None
            
    def rotate_profile(self):
        """Rotate to a new profiling window"""
        self.end_window()
        self.start_window()

# Usage example
profiler = TimeWindowProfiler(window_duration_seconds=300)  # 5-minute windows
profiler.start_window()

# Your service code
while service_running:
    # Check if it's time to rotate
    if time_to_rotate():
        profiler.rotate_profile()
    
    # Process requests
    process_request()
```

#### Advanced Periodic Profiling for LLM Inference

```python
import proton
import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

class LLMInferenceProfiler:
    def __init__(self, 
                 window_minutes: int = 5,
                 profile_dir: str = "./inference_profiles",
                 detailed_sampling_interval: int = 60):  # Detailed profiling every N windows
        self.window_duration = timedelta(minutes=window_minutes)
        self.profile_dir = profile_dir
        self.detailed_interval = detailed_sampling_interval
        
        self.window_number = 0
        self.current_session: Optional[int] = None
        self.window_start_time: Optional[datetime] = None
        self.window_metrics: Dict[str, Any] = {}
        
        # Create profile directory
        os.makedirs(profile_dir, exist_ok=True)
        
    def start_profiling_window(self):
        """Start a new profiling window"""
        self.window_start_time = datetime.now()
        self.window_number += 1
        
        # Use detailed profiling periodically
        if self.window_number % self.detailed_interval == 0:
            backend = "cupti_pcsampling"
            suffix = "detailed"
        else:
            backend = "cupti"
            suffix = "standard"
            
        timestamp = self.window_start_time.strftime("%Y%m%d_%H%M%S")
        profile_name = f"{self.profile_dir}/inference_{suffix}_w{self.window_number}_{timestamp}"
        
        self.current_session = proton.start(profile_name, backend=backend, hook="triton")
        proton.activate(self.current_session)
        
        # Reset window metrics
        self.window_metrics = {
            "window_number": self.window_number,
            "start_time": timestamp,
            "requests_processed": 0,
            "tokens_generated": 0,
            "errors": 0
        }
        
    def end_profiling_window(self):
        """End current window and save profile with metadata"""
        if self.current_session is None:
            return
            
        # Record final metrics
        self.window_metrics["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.window_metrics["duration_seconds"] = (datetime.now() - self.window_start_time).total_seconds()
        
        # Save metadata
        metadata_path = f"{self.profile_dir}/metadata_w{self.window_number}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.window_metrics, f, indent=2)
        
        # Finalize profiling
        proton.deactivate(self.current_session)
        proton.finalize(self.current_session)
        self.current_session = None
        
    async def periodic_rotation(self):
        """Async task to rotate profiles periodically"""
        while True:
            await asyncio.sleep(self.window_duration.total_seconds())
            self.end_profiling_window()
            self.start_profiling_window()
            
    def record_request(self, request_id: str, prompt_tokens: int, generated_tokens: int):
        """Record metrics for a processed request"""
        with proton.scope(f"request_{request_id}", {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens
        }):
            self.window_metrics["requests_processed"] += 1
            self.window_metrics["tokens_generated"] += generated_tokens

# LLM Inference Service Example
class LLMInferenceService:
    def __init__(self):
        self.profiler = LLMInferenceProfiler(
            window_minutes=5,
            detailed_sampling_interval=12  # Detailed profiling every hour
        )
        self.model = load_model()
        
    async def start(self):
        """Start the service with profiling"""
        self.profiler.start_profiling_window()
        
        # Start periodic rotation task
        asyncio.create_task(self.profiler.periodic_rotation())
        
        # Start serving requests
        await self.serve_requests()
        
    async def process_request(self, request):
        """Process an inference request with profiling"""
        request_id = request.id
        
        with proton.scope("tokenization"):
            tokens = tokenize(request.prompt)
            
        with proton.scope("model_forward"):
            generated = await self.model.generate(tokens)
            
        with proton.scope("detokenization"):
            response = detokenize(generated)
            
        # Record metrics
        self.profiler.record_request(
            request_id,
            len(tokens),
            len(generated)
        )
        
        return response
```

#### Rolling Window Profiling with Buffer Management

```python
import proton
from collections import deque
from threading import Lock
import shutil

class RollingWindowProfiler:
    """
    Maintains a rolling window of profiles with automatic cleanup
    """
    def __init__(self, 
                 window_size_minutes: int = 10,
                 max_windows_kept: int = 6,  # Keep last hour of profiles
                 profile_dir: str = "./rolling_profiles"):
        self.window_size = window_size_minutes * 60  # Convert to seconds
        self.max_windows = max_windows_kept
        self.profile_dir = profile_dir
        self.profile_queue = deque(maxlen=max_windows_kept)
        self.lock = Lock()
        self.current_session = None
        
    def start_new_window(self):
        """Start a new profiling window"""
        with self.lock:
            # End previous window if exists
            if self.current_session is not None:
                proton.finalize(self.current_session)
                
            # Create new profile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_path = f"{self.profile_dir}/rolling_{timestamp}"
            
            self.current_session = proton.start(profile_path, backend="cupti")
            proton.activate(self.current_session)
            
            # Add to queue (automatic cleanup via maxlen)
            self.profile_queue.append({
                "path": profile_path,
                "session": self.current_session,
                "start_time": datetime.now()
            })
            
            # Clean up old files if queue is full
            if len(self.profile_queue) == self.max_windows:
                self._cleanup_old_profiles()
                
    def _cleanup_old_profiles(self):
        """Remove profiles older than the window"""
        # The deque automatically removes old entries
        # This method can be used for additional cleanup if needed
        pass
        
    def get_recent_profiles(self, n: int = None):
        """Get paths to recent profile files"""
        with self.lock:
            if n is None:
                n = len(self.profile_queue)
            return [p["path"] + ".hatchet" for p in list(self.profile_queue)[-n:]]
            
    def analyze_trends(self):
        """Analyze performance trends across windows"""
        recent_profiles = self.get_recent_profiles()
        
        for profile_path in recent_profiles:
            if os.path.exists(profile_path):
                gf, metrics = proton.parse(["time/ms", "util"], profile_path)
                # Perform trend analysis
                print(f"Profile: {profile_path}")
                print(f"  Total time: {gf.dataframe['time/ms (inc)'].iloc[0]:.2f} ms")
                print(f"  Average utilization: {gf.dataframe['util'].mean():.2%}")
```

#### Service Window Profiling with Exit Metrics

```python
import proton
import numpy as np
from dataclasses import dataclass
from typing import List
import time

@dataclass
class RequestMetrics:
    latency_ms: float
    tokens_generated: int
    success: bool

class ServiceWindowProfiler:
    """
    Profile service windows with comprehensive end-of-window metrics
    """
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.current_requests: List[RequestMetrics] = []
        self.window_scope_id = None
        
    def start_window(self):
        """Start a new profiling window"""
        self.current_requests = []
        self.window_start = time.time()
        self.window_scope_id = proton.enter_scope("service_window", {
            "window_start_timestamp": self.window_start,
            "window_duration_seconds": self.window_seconds
        })
        
    def record_request(self, request_metrics: RequestMetrics):
        """Record metrics for a completed request"""
        self.current_requests.append(request_metrics)
        
    def end_window(self):
        """End window and record comprehensive metrics"""
        if not self.current_requests:
            proton.exit_scope(metrics={"requests_count": 0})
            return
            
        # Calculate latency percentiles
        latencies = [r.latency_ms for r in self.current_requests]
        latencies_array = np.array(latencies)
        
        # Calculate throughput metrics
        window_duration = time.time() - self.window_start
        total_tokens = sum(r.tokens_generated for r in self.current_requests)
        successful_requests = sum(1 for r in self.current_requests if r.success)
        
        # Record comprehensive end-of-window metrics
        proton.exit_scope(metrics={
            # Latency percentiles
            "p50_latency_ms": np.percentile(latencies_array, 50),
            "p90_latency_ms": np.percentile(latencies_array, 90),
            "p95_latency_ms": np.percentile(latencies_array, 95),
            "p99_latency_ms": np.percentile(latencies_array, 99),
            "max_latency_ms": np.max(latencies_array),
            "min_latency_ms": np.min(latencies_array),
            "avg_latency_ms": np.mean(latencies_array),
            
            # Throughput metrics
            "requests_per_second": len(self.current_requests) / window_duration,
            "tokens_per_second": total_tokens / window_duration,
            "success_rate": successful_requests / len(self.current_requests),
            
            # Volume metrics
            "total_requests": len(self.current_requests),
            "total_tokens_generated": total_tokens,
            "failed_requests": len(self.current_requests) - successful_requests
        })

# Usage in LLM service
profiler = ServiceWindowProfiler(window_seconds=300)
profiler.start_window()

# Process requests
for request in incoming_requests:
    start_time = time.time()
    try:
        tokens = process_llm_request(request)
        profiler.record_request(RequestMetrics(
            latency_ms=(time.time() - start_time) * 1000,
            tokens_generated=len(tokens),
            success=True
        ))
    except Exception as e:
        profiler.record_request(RequestMetrics(
            latency_ms=(time.time() - start_time) * 1000,
            tokens_generated=0,
            success=False
        ))

# End window with all metrics
profiler.end_window()
```

#### Continuous Profiling with Alerts

```python
import proton
from typing import Callable
import statistics

class ContinuousProfiler:
    """
    Continuous profiling with performance monitoring and alerts
    """
    def __init__(self,
                 window_seconds: int = 300,
                 alert_callback: Callable = None):
        self.window_seconds = window_seconds
        self.alert_callback = alert_callback
        self.performance_history = deque(maxlen=20)  # Keep last 20 windows
        self.baseline_metrics = None
        
    def profile_with_monitoring(self):
        """Profile and monitor for performance anomalies"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_name = f"continuous_{timestamp}"
        
        session = proton.start(profile_name, backend="cupti")
        proton.activate(session)
        
        # Let it run for the window duration
        time.sleep(self.window_seconds)
        
        proton.deactivate(session)
        proton.finalize(session)
        
        # Analyze the profile
        metrics = self._analyze_profile(f"{profile_name}.hatchet")
        self.performance_history.append(metrics)
        
        # Check for anomalies
        if self._detect_anomaly(metrics):
            if self.alert_callback:
                self.alert_callback(metrics)
                
    def _analyze_profile(self, profile_path):
        """Extract key metrics from profile"""
        gf, _ = proton.parse(["time/ms", "util"], profile_path)
        
        return {
            "timestamp": datetime.now(),
            "avg_kernel_time_ms": gf.dataframe['time/ms (inc)'].mean(),
            "max_kernel_time_ms": gf.dataframe['time/ms (inc)'].max(),
            "avg_utilization": gf.dataframe['util'].mean(),
            "profile_path": profile_path
        }
        
    def _detect_anomaly(self, current_metrics):
        """Detect performance anomalies"""
        if len(self.performance_history) < 5:
            return False  # Not enough history
            
        # Calculate statistics from history
        avg_times = [m["avg_kernel_time_ms"] for m in self.performance_history[-5:]]
        historical_avg = statistics.mean(avg_times)
        historical_std = statistics.stdev(avg_times)
        
        # Alert if current performance is 2 std devs away from mean
        if abs(current_metrics["avg_kernel_time_ms"] - historical_avg) > 2 * historical_std:
            return True
            
        return False

# Usage
def performance_alert(metrics):
    print(f"⚠️ Performance anomaly detected at {metrics['timestamp']}")
    print(f"  Average kernel time: {metrics['avg_kernel_time_ms']:.2f} ms")
    print(f"  Profile saved at: {metrics['profile_path']}")
    # Send alert to monitoring system
    
profiler = ContinuousProfiler(
    window_seconds=300,
    alert_callback=performance_alert
)

# Run continuous profiling
while True:
    profiler.profile_with_monitoring()
```

### PC Sampling Analysis

```python
import proton

# Enable PC sampling for detailed analysis
session = proton.start("detailed", backend="cupti_pcsampling")
proton.activate(session)

# Your GPU-intensive code
kernel_computation()

proton.finalize(session)

# Analyze with viewer
# proton-viewer -m "StalledBranchResolving,StalledMemoryDependency" detailed.hatchet
```

### Profile Analysis and Visualization

```python
import proton

# Read profile data
gf, inclusive, exclusive, device_info = proton.read("profile.hatchet")

# Parse with derived metrics
gf, derived = proton.parse(
    ["time/ms", "flop/s", "util"],
    "profile.hatchet",
    include=".*kernel.*",
    threshold=0.01
)

# Print analysis
print(gf.dataframe)
```

## Development Guide

### Building from Source

```bash
cd triton/third_party/proton
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Adding New Backends

1. Create backend implementation in `csrc/lib/Profiler/YourBackend/`
2. Implement `GPUProfilerPimplInterface`
3. Add backend selection logic in Python API
4. Update CMake configuration for dependencies

### Adding New Metrics

1. Define metric class inheriting from `Metric`
2. Implement virtual methods for naming and properties
3. Add metric collection in profiler backend
4. Update aggregation logic if needed

### Testing

```bash
cd test
python -m pytest test_api.py -v
python -m pytest test_profile.py -v
```

## Troubleshooting

### Common Issues

#### Backend Detection Fails

**Symptoms**: `ValueError: No backend available`

**Solutions**:
- Ensure CUDA/HIP runtime is properly installed
- Check `triton.runtime.driver.active.get_current_target().backend`
- Manually specify backend: `proton.start(backend="cupti")`

#### Library Path Issues

**Symptoms**: `RuntimeError: Failed to load profiler library`

**Solutions**:
- Set `CUPTI_ROOT_PATH` environment variable
- Use `proton-viewer --list-backends` to check available backends
- Manually set library path via knobs

#### Permission Errors (Linux)

**Symptoms**: `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`

**Solutions**:
```bash
# Temporary (until reboot)
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Permanent
echo 'kernel.perf_event_paranoid = 0' | sudo tee -a /etc/sysctl.conf
```

#### AMD GPU Issues

**Symptoms**: Device visibility errors

**Solutions**:
- Use `ROCR_VISIBLE_DEVICES` instead of `HIP_VISIBLE_DEVICES`
- Avoid setting `CUDA_VISIBLE_DEVICES` on AMD systems
- Check ROCm installation and permissions

### Performance Issues

#### High Overhead

**Symptoms**: Significant slowdown during profiling

**Solutions**:
- Use `context="shadow"` instead of `context="python"`
- Disable PC sampling if not needed
- Reduce scope granularity
- Use `cupti` instead of `cupti_pcsampling`

#### Large Profile Files

**Symptoms**: Profile files are too large

**Solutions**:
- Use threshold filtering: `proton-viewer -t 0.01`
- Filter irrelevant frames: `proton-viewer -e ".*internal.*"`
- Use context aggregation effectively

### Debug Tips

#### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Session Status

```python
import proton
print(f"Context depth: {proton.depth()}")
```

#### Validate Context Balance

Ensure proper scope entry/exit balance:

```python
try:
    with proton.scope("test"):
        # Your code
        pass
except RuntimeError as e:
    print(f"Context error: {e}")
```

## Limitations and Future Features

### Current Limitations

1. **Output Formats**: Currently only supports Hatchet JSON format for output. Other formats may be added in future releases.

2. **Chrome Trace Export**: While the codebase includes a `TraceData` class infrastructure, Chrome trace timeline export is not currently implemented. All methods in `TraceData` throw `NotImplemented()` exceptions.

3. **Trace-based Data Collection**: The `data="trace"` option exists in the API but is not functional due to the unimplemented `TraceData` class.

4. **Limited Derived Metrics**: While many derived metrics are available, some advanced metrics may require manual calculation.

### Future Features

Based on the existing infrastructure, potential future enhancements include:

1. **Chrome Trace Format Export**: 
   - Implementation of the `TraceData` class
   - Support for Chrome's trace event format (JSON)
   - Timeline visualization in Chrome DevTools

2. **Additional Output Formats**:
   - Support for other profiling formats (e.g., pprof, speedscope)
   - Custom binary formats for reduced file sizes

3. **Enhanced Analysis Tools**:
   - Built-in statistical analysis
   - Automated performance regression detection
   - Machine learning-based optimization suggestions

4. **Extended Platform Support**:
   - Additional GPU architectures
   - CPU profiling integration
   - Distributed profiling for multi-node systems

### API Stability

The Proton profiler API is actively developed. While the core functionality is stable:
- New features may be added in minor releases
- Breaking changes will be documented in major releases
- Experimental features are marked in the documentation

This comprehensive documentation provides complete coverage of the Triton Proton profiler, from basic usage to advanced implementation details. The profiler represents a sophisticated, production-ready solution for GPU performance analysis with particular strength in Triton kernel profiling.