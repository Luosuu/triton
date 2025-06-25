# Triton Proton Profiler: Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [Comprehensive API Reference](#comprehensive-api-reference)
4. [Implementation Details](#implementation-details)
5. [Getting Started](#getting-started)
6. [Advanced Usage](#advanced-usage)
7. [Development Guide](#development-guide)
8. [Troubleshooting](#troubleshooting)

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

**Returns**: Session ID (int)

**Example**:
```python
session_id = proton.start("my_profile", backend="cupti", hook="triton")
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
- `metrics` (dict[str, Union[int, float]], optional): Custom metrics

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

Exit the most recent scope.

**Parameters**:
- `triton_op` (bool): Must match `enter_scope()` call
- `metrics` (dict, optional): Additional metrics

**Returns**: Scope ID (int)

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
- `--diff-profile FILE`: Compare two profiles

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

This comprehensive documentation provides complete coverage of the Triton Proton profiler, from basic usage to advanced implementation details. The profiler represents a sophisticated, production-ready solution for GPU performance analysis with particular strength in Triton kernel profiling.