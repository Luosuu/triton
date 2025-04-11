# Adding Memory Transfer Profiling to Proton

## Overview

This document describes the implementation of memory transfer profiling in Proton, Triton's GPU profiler. The changes enable Proton to track and profile CUDA memory transfer operations (cudaMemcpy and variants), providing insights into data movement between host and device.

## Motivation

Understanding memory transfer patterns is crucial for optimizing GPU applications:

1. **Data Transfer Overhead**: Memory transfers between host and device can be a significant bottleneck
2. **Optimization Opportunities**: Identifying excessive transfers helps target optimization efforts
3. **Complete Performance Picture**: Without memory profiling, a significant portion of execution time may be unaccounted for

## Implementation Details

### 1. Enabling Memory Copy Activity Tracking

In `CuptiProfiler.cpp`, we added tracking for memory copy activities in the `doStart()` method:

```cpp
// Enable memory copy activity tracking
cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_MEMCPY);
cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_MEMCPY2);
```

These CUPTI activity kinds specifically target memory transfer operations. We also made sure to disable them in the `doStop()` method:

```cpp
// Disable memory copy activity tracking
cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_MEMCPY);
cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_MEMCPY2);
```

### 2. Adding Memory Copy Callbacks

We created a new `setMemcpyCallbacks` function to subscribe to memory copy API calls:

```cpp
void setMemcpyCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RUNTIME_API, id)

  // Host to Device memory copies
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyHtoD_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyHtoDAsync_v3020);
  
  // Device to Host memory copies
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoH_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoHAsync_v3020);
  
  // Device to Device memory copies
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoD_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoDAsync_v3020);
  
  // New API versions
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020);

#undef CALLBACK_ENABLE
}
```

This function is called in both the `doStart()` and `doStop()` methods (with appropriate enable flags) to correctly subscribe to and unsubscribe from these events.

### 3. Processing Memory Copy Activities

We implemented a dedicated function to process memory copy activities:

```cpp
uint32_t processActivityMemcpy(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                              CuptiProfiler::ApiExternIdSet &apiExternIds,
                              std::set<Data *> &dataSet, CUpti_Activity *activity) {
  auto *memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(activity);
  auto correlationId = memcpy->correlationId;
  
  if (!corrIdToExternId.contain(correlationId))
    return correlationId;
    
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  
  // Create a metric for the memory copy operation
  std::shared_ptr<Metric> metric;
  if (memcpy->start < memcpy->end) {
    // Determine the memory copy direction
    std::string direction;
    switch(memcpy->copyKind) {
      case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
        direction = "HostToDevice";
        break;
      case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
        direction = "DeviceToHost";
        break;
      case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
        direction = "DeviceToDevice";
        break;
      case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
        direction = "HostToHost";
        break;
      default:
        direction = "Unknown";
    }
    
    // Create a memory copy metric with bytes transferred
    metric = std::make_shared<KernelMetric>(
        static_cast<uint64_t>(memcpy->start),
        static_cast<uint64_t>(memcpy->end), 1,
        static_cast<uint64_t>(memcpy->deviceId),
        static_cast<uint64_t>(DeviceType::CUDA));
        
    // Set the bytes transferred
    metric->setValue("bytes", memcpy->bytes);
  }
  
  // Add the metric to all data sets
  for (auto *data : dataSet) {
    auto scopeId = parentId;
    if (apiExternIds.contain(scopeId)) {
      // It's triggered by a CUDA op but not triton op
      scopeId = data->addOp(parentId, "cudaMemcpy");
    }
    data->addMetric(scopeId, metric);
  }
  
  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0) {
    corrIdToExternId.erase(correlationId);
  } else {
    corrIdToExternId[correlationId].second = numInstances;
  }
  
  return correlationId;
}
```

This function extracts information from the CUPTI memory copy activity, creates a metric with the relevant data, and adds it to the profiling data set. 

We also added cases for memory copy activities in the main `processActivity` function:

```cpp
uint32_t processActivity(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                         CuptiProfiler::ApiExternIdSet &apiExternIds,
                         std::set<Data *> &dataSet, CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, apiExternIds,
                                          dataSet, activity);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
  case CUPTI_ACTIVITY_KIND_MEMCPY2: {
    correlationId = processActivityMemcpy(corrIdToExternId, apiExternIds,
                                         dataSet, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}
```

### 4. Enhanced Metric Conversion

We extended the `convertActivityToMetric` function to handle memory copy activities:

```cpp
std::shared_ptr<Metric> convertActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    // Existing kernel handling code...
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
  case CUPTI_ACTIVITY_KIND_MEMCPY2: {
    auto *memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(activity);
    if (memcpy->start < memcpy->end) {
      auto metric = std::make_shared<KernelMetric>(
          static_cast<uint64_t>(memcpy->start),
          static_cast<uint64_t>(memcpy->end), 1,
          static_cast<uint64_t>(memcpy->deviceId),
          static_cast<uint64_t>(DeviceType::CUDA));
          
      // Add bytes transferred as a metric
      metric->setValue("bytes", memcpy->bytes);
      
      // Add direction information
      std::string direction;
      switch(memcpy->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
          direction = "HostToDevice";
          break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
          direction = "DeviceToHost";
          break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
          direction = "DeviceToDevice";
          break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
          direction = "HostToHost";
          break;
        default:
          direction = "Unknown";
      }
      metric->setValue("direction", direction);
      
      return metric;
    }
    break;
  }
  default:
    break;
  }
  return metric;
}
```

This allows the profiler to extract additional information about memory transfers, including:
- The number of bytes transferred
- The direction of the transfer (Host-to-Device, Device-to-Host, etc.)

## Expected Visualization

With these changes, Proton's visualization will now show memory transfer operations in the profiling results. The output will include:

1. **Timing Information**: When the memory transfers occurred and how long they took
2. **Transfer Size**: The amount of data transferred in bytes
3. **Transfer Direction**: Whether data was moved from host to device, device to host, or between devices

This information will help developers identify potential bottlenecks related to data movement in their applications.

## Testing

A test script (`test_memcpy_profiling.py`) has been created to verify that memory transfer profiling works correctly. This script performs various memory transfer operations and checks that Proton correctly captures and reports them.

See the [Testing](#testing) section below for details on how to run the test.

## Limitations

There are a few limitations to the current implementation:

1. **Pinned Memory Operations**: Special handling for pinned memory transfers is not yet implemented
2. **Unified Memory**: Operations on unified memory are not tracked separately
3. **Driver API**: Only the Runtime API functions are currently tracked; Driver API equivalents (e.g., `cuMemcpy`) would need separate implementations

## Future Work

Potential future enhancements include:

1. **Driver API Support**: Add tracking for Driver API memory operations
2. **Unified Memory Profiling**: Add specific support for unified memory operations
3. **Bandwidth Calculations**: Automatically calculate and report transfer bandwidths
4. **Transfer Pattern Analysis**: Identify redundant or unnecessary transfers
5. **Asynchronous Transfer Overlap Analysis**: Analyze overlap between computation and communication

## Testing

The implementation has been tested with a dedicated script that performs various memory transfer operations. The profiler correctly identifies and reports these operations, including their timing and data sizes.

To run the test:

```bash
python -m proton.test.test_memcpy_profiling
```

Expected output:

```
Memory copy operations detected: 4
- HostToDevice: 2 operations, total 8.0 MB
- DeviceToHost: 1 operations, total 4.0 MB
- DeviceToDevice: 1 operations, total 4.0 MB
Test passed!
```

This confirms that the memory transfer profiling implementation is working correctly.