#include "Profiler/Cupti/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Utility/Map.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<CuptiProfiler>::ThreadState
    GPUProfiler<CuptiProfiler>::threadState(CuptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<CuptiProfiler>::Correlation::externIdQueue{};

namespace {

/**
 * Converts a CUPTI activity record to a Proton metric.
 * 
 * This function extracts data from CUPTI activity records and converts them
 * into Proton metrics that can be processed and visualized.
 * 
 * @param activity Pointer to the CUPTI activity record
 * @return A shared pointer to a Metric object, or nullptr if not supported
 */
std::shared_ptr<Metric> convertActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    // Process kernel execution activity
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
    if (kernel->start < kernel->end) {
      // Create a kernel metric with timestamp, device ID, etc.
      metric = std::make_shared<KernelMetric>(
          static_cast<uint64_t>(kernel->start),
          static_cast<uint64_t>(kernel->end), 1,
          static_cast<uint64_t>(kernel->deviceId),
          static_cast<uint64_t>(DeviceType::CUDA));
    } // else: not a valid kernel activity (end time <= start time)
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
  case CUPTI_ACTIVITY_KIND_MEMCPY2: {
    // Process memory copy activity
    auto *memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(activity);
    if (memcpy->start < memcpy->end) {
      // Create a metric for the memory transfer operation
      auto metric = std::make_shared<KernelMetric>(
          static_cast<uint64_t>(memcpy->start),
          static_cast<uint64_t>(memcpy->end), 1,
          static_cast<uint64_t>(memcpy->deviceId),
          static_cast<uint64_t>(DeviceType::CUDA));
          
      // Add bytes transferred as a metric - useful for bandwidth calculations
      metric->setValue("bytes", memcpy->bytes);
      
      // Add direction information to distinguish between H2D, D2H, etc.
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
    } // else: not a valid memcpy activity (end time <= start time)
    break;
  }
  default:
    // Unsupported activity type
    break;
  }
  return metric;
}

uint32_t
processActivityKernel(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                      CuptiProfiler::ApiExternIdSet &apiExternIds,
                      std::set<Data *> &dataSet, CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  if (/*Not a valid context*/ !corrIdToExternId.contain(correlationId))
    return correlationId;
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  if (kernel->graphId == 0) {
    // Non-graph kernels
    for (auto *data : dataSet) {
      auto scopeId = parentId;
      if (apiExternIds.contain(scopeId)) {
        // It's triggered by a CUDA op but not triton op
        scopeId = data->addOp(parentId, kernel->name);
      }
      data->addMetric(scopeId, convertActivityToMetric(activity));
    }
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // 1. graphId -> numKernels
    // 2. graphExecId -> graphId
    // --- CUPTI thread ---
    // 3. corrId -> numKernels
    for (auto *data : dataSet) {
      auto externId = data->addOp(parentId, kernel->name);
      data->addMetric(externId, convertActivityToMetric(activity));
    }
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

/**
 * Process a memory copy activity record from CUPTI.
 * 
 * This function extracts information from a CUPTI memory copy activity record,
 * creates a metric, and adds it to the appropriate data set based on the correlation ID.
 * It handles different types of memory copies (H2D, D2H, D2D, etc.) and records
 * important metrics such as bytes transferred and operation duration.
 * 
 * @param corrIdToExternId Map from CUPTI correlation IDs to Proton external IDs
 * @param apiExternIds Set of API external IDs
 * @param dataSet Set of data objects to add metrics to
 * @param activity CUPTI activity record for the memory copy
 * @return Correlation ID of the processed activity
 */
uint32_t processActivityMemcpy(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                              CuptiProfiler::ApiExternIdSet &apiExternIds,
                              std::set<Data *> &dataSet, CUpti_Activity *activity) {
  // Cast the generic activity to a memory copy activity
  auto *memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(activity);
  auto correlationId = memcpy->correlationId;
  
  // Skip if we can't find the correlation ID in our map
  // This can happen if profiling was started after the API call was made
  if (!corrIdToExternId.contain(correlationId))
    return correlationId;
    
  // Get the parent operation ID and number of instances from our map
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  
  // Create a metric for the memory copy operation
  std::shared_ptr<Metric> metric;
  if (memcpy->start < memcpy->end) {
    // Determine the memory copy direction for more informative profiling
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
    
    // Create a memory copy metric with timing and device information
    metric = std::make_shared<KernelMetric>(
        static_cast<uint64_t>(memcpy->start),
        static_cast<uint64_t>(memcpy->end), 1,
        static_cast<uint64_t>(memcpy->deviceId),
        static_cast<uint64_t>(DeviceType::CUDA));
        
    // Add the number of bytes transferred to enable bandwidth calculations
    metric->setValue("bytes", memcpy->bytes);
    
    // Add the direction as a string value for easier filtering in visualization
    metric->setValue("direction", direction);
  }
  
  // Add the metric to all registered data sets
  for (auto *data : dataSet) {
    auto scopeId = parentId;
    if (apiExternIds.contain(scopeId)) {
      // If this was triggered by a CUDA API call (not a Triton op),
      // add a specific operation name for better context
      scopeId = data->addOp(parentId, "cudaMemcpy");
    }
    // Add the metric to the data set with the appropriate scope ID
    data->addMetric(scopeId, metric);
  }
  
  // Update bookkeeping structures
  apiExternIds.erase(parentId);
  --numInstances;
  
  // Clean up the correlation map if this was the last instance
  if (numInstances == 0) {
    corrIdToExternId.erase(correlationId);
  } else {
    // Otherwise update the instance count
    corrIdToExternId[correlationId].second = numInstances;
  }
  
  return correlationId;
}

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

void setRuntimeCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RUNTIME_API, id)

  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000);
  CALLBACK_ENABLE(
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000);
  CALLBACK_ENABLE(
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);

#undef CALLBACK_ENABLE
}

void setDriverCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_DRIVER_API, id)

  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunch);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
#undef CALLBACK_ENABLE
}

/**
 * Enable or disable callbacks for memory copy operations.
 * 
 * This function configures CUPTI to track CUDA memory transfer operations.
 * It covers various memory transfer APIs including synchronous and asynchronous
 * transfers in different directions (host-to-device, device-to-host, etc.)
 * 
 * @param subscriber CUPTI subscriber handle
 * @param enable Whether to enable (true) or disable (false) callbacks
 */
void setMemcpyCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RUNTIME_API, id)

  // Host to Device memory copies
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);       // cudaMemcpy with H2D direction
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020);  // Async version
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyHtoD_v3020);   // Explicit H2D API
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyHtoDAsync_v3020); // Async H2D API
  
  // Device to Host memory copies
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoH_v3020);    // Explicit D2H API
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoHAsync_v3020); // Async D2H API
  
  // Device to Device memory copies
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoD_v3020);    // Device to device copy
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyDtoDAsync_v3020); // Async D2D copy
  
  // New API versions (CUDA 7.0+)
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v7000);        // CUDA 7.0 API version
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v7000);   // CUDA 7.0 async API
  
  // Symbol-based copies (for device constant and texture memory)
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020);    // Copy to symbol
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020);  // Copy from symbol

#undef CALLBACK_ENABLE
}

void setGraphCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {

#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RESOURCE, id)

  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING);
#undef CALLBACK_ENABLE
}

void setResourceCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RESOURCE, id)

  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_MODULE_LOADED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);
#undef CALLBACK_ENABLE
}

bool isDriverAPILaunch(CUpti_CallbackId cbId) {
  return cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
}

} // namespace

/**
 * PIMPL (Pointer to Implementation) for CuptiProfiler
 * 
 * This struct contains the implementation details of the CuptiProfiler.
 * It handles the interaction with CUPTI for profiling CUDA operations,
 * including kernel launches and memory transfers.
 */
struct CuptiProfiler::CuptiProfilerPimpl
    : public GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface {
  
  /**
   * Constructor for the PIMPL implementation.
   * @param profiler Reference to the parent CuptiProfiler
   */
  CuptiProfilerPimpl(CuptiProfiler &profiler)
      : GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  
  /**
   * Virtual destructor.
   */
  virtual ~CuptiProfilerPimpl() = default;

  /**
   * Sets the path to the CUPTI library.
   * @param libPath Path to the CUPTI library
   */
  void setLibPath(const std::string &libPath) override {
    cupti::setLibPath(libPath);
  }
  
  /**
   * Starts the profiler. 
   * Enables CUPTI callbacks and activity tracking.
   */
  void doStart() override;
  
  /**
   * Flushes profiling data from the device to the host.
   */
  void doFlush() override;
  
  /**
   * Stops the profiler.
   * Disables CUPTI callbacks and activity tracking.
   */
  void doStop() override;

  /**
   * Allocates a buffer for CUPTI activity records.
   * This is a callback function used by CUPTI.
   * 
   * @param buffer Pointer to the allocated buffer
   * @param bufferSize Size of the allocated buffer
   * @param maxNumRecords Maximum number of records the buffer can hold
   */
  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  
  /**
   * Processes a completed buffer of CUPTI activity records.
   * This is a callback function used by CUPTI.
   * 
   * @param context CUDA context
   * @param streamId CUDA stream ID
   * @param buffer Buffer containing activity records
   * @param size Total size of the buffer
   * @param validSize Size of valid data in the buffer
   */
  static void completeBuffer(CUcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  
  /**
   * Callback function for CUPTI events.
   * Handles CUDA API calls, kernel launches, and other events.
   * 
   * @param userData User data pointer
   * @param domain CUPTI callback domain
   * @param cbId CUPTI callback ID
   * @param cbData CUPTI callback data
   */
  static void callbackFn(void *userData, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbId, const void *cbData);

  // Constants for buffer allocation and management
  static constexpr size_t AlignSize = 8;            ///< Alignment size for buffer allocation
  static constexpr size_t BufferSize = 64 * 1024 * 1024;  ///< 64MB default buffer size
  static constexpr size_t AttributeSize = sizeof(size_t);  ///< Size of attribute values

  CUpti_SubscriberHandle subscriber{};  ///< CUPTI subscriber handle
  CuptiPCSampling pcSampling;           ///< PC sampling functionality

  // Maps for tracking CUDA graph operations
  /** Maps graph IDs to the number of instances (nodes) in the graph */
  ThreadSafeMap<uint32_t, size_t, std::unordered_map<uint32_t, size_t>>
      graphIdToNumInstances;
      
  /** Maps graph execution IDs to their corresponding graph IDs */
  ThreadSafeMap<uint32_t, uint32_t, std::unordered_map<uint32_t, uint32_t>>
      graphExecIdToGraphId;
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("[PROTON] aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler = threadState.profiler;
  auto dataSet = profiler.getDataSet();
  uint32_t maxCorrelationId = 0;
  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      auto correlationId =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("[PROTON] cupti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
}

void CuptiProfiler::CuptiProfilerPimpl::callbackFn(void *userData,
                                                   CUpti_CallbackDomain domain,
                                                   CUpti_CallbackId cbId,
                                                   const void *cbData) {
  CuptiProfiler &profiler = threadState.profiler;
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    auto *resourceData =
        static_cast<CUpti_ResourceData *>(const_cast<void *>(cbData));
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (cbId == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.loadModule(moduleResource->pCubin,
                                     moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.unloadModule(moduleResource->pCubin,
                                       moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.initialize(resourceData->context);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.finalize(resourceData->context);
      }
    } else {
      auto *graphData =
          static_cast<CUpti_GraphData *>(resourceData->resourceDescriptor);
      uint32_t graphId = 0;
      uint32_t graphExecId = 0;
      if (graphData->graph)
        cupti::getGraphId<true>(graphData->graph, &graphId);
      if (graphData->graphExec)
        cupti::getGraphExecId<true>(graphData->graphExec, &graphExecId);
      if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED ||
          cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
        if (!pImpl->graphIdToNumInstances.contain(graphId))
          pImpl->graphIdToNumInstances[graphId] = 1;
        else
          pImpl->graphIdToNumInstances[graphId]++;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING) {
        pImpl->graphIdToNumInstances[graphId]--;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED) {
        pImpl->graphExecIdToGraphId[graphExecId] = graphId;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING) {
        pImpl->graphExecIdToGraphId.erase(graphExecId);
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING) {
        pImpl->graphIdToNumInstances.erase(graphId);
      }
    }
  } else {
    const CUpti_CallbackData *callbackData =
        static_cast<const CUpti_CallbackData *>(cbData);
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (callbackData->callbackSite == CUPTI_API_ENTER) {
      threadState.enterOp();
      size_t numInstances = 1;
      if (cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
        auto graphExec = static_cast<const cuGraphLaunch_params *>(
                             callbackData->functionParams)
                             ->hGraph;
        uint32_t graphExecId = 0;
        cupti::getGraphExecId<true>(graphExec, &graphExecId);
        numInstances = std::numeric_limits<size_t>::max();
        auto findGraph = false;
        if (pImpl->graphExecIdToGraphId.contain(graphExecId)) {
          auto graphId = pImpl->graphExecIdToGraphId[graphExecId];
          if (pImpl->graphIdToNumInstances.contain(graphId)) {
            numInstances = pImpl->graphIdToNumInstances[graphId];
            findGraph = true;
          }
        }
        if (!findGraph)
          std::cerr << "[PROTON] Cannot find graph for graphExecId: "
                    << graphExecId
                    << ", and t may cause memory leak. To avoid this problem, "
                       "please start profiling before the graph is created."
                    << std::endl;
      }
      profiler.correlation.correlate(callbackData->correlationId, numInstances);
      if (profiler.isPCSamplingEnabled() && isDriverAPILaunch(cbId)) {
        pImpl->pcSampling.start(callbackData->context);
      }
    } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
      if (profiler.isPCSamplingEnabled() && isDriverAPILaunch(cbId)) {
        // XXX: Conservatively stop every GPU kernel for now
        auto scopeId = profiler.correlation.externIdQueue.back();
        pImpl->pcSampling.stop(
            callbackData->context, scopeId,
            profiler.correlation.apiExternIds.contain(scopeId));
      }
      threadState.exitOp();
      profiler.correlation.submit(callbackData->correlationId);
    }
  }
}

/**
 * Start the CUPTI profiler.
 * 
 * This method initializes the CUPTI profiler, subscribes to callbacks,
 * and enables activity tracking for kernels and memory operations.
 * It sets up all the necessary infrastructure for profiling.
 */
void CuptiProfiler::CuptiProfilerPimpl::doStart() {
  // Subscribe to CUPTI callbacks to capture events
  cupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  
  // Configure profiling based on the mode
  if (profiler.isPCSamplingEnabled()) {
    // Enable resource callbacks for PC sampling
    setResourceCallbacks(subscriber, /*enable=*/true);
    
    // For PC sampling, we use non-concurrent kernel tracking
    // Note: Continuous PC sampling is not compatible with concurrent kernel profiling
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    // For regular profiling, use concurrent kernel tracking
    // This is more efficient when not doing PC sampling
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  
  // Enable memory copy activity tracking for both synchronous and asynchronous transfers
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_MEMCPY);  // For CUDA 4.0+
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_MEMCPY2); // For CUDA 7.0+
  
  // Register callbacks for buffer allocation and processing
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  
  // Enable various callback domains
  setGraphCallbacks(subscriber, /*enable=*/true);      // CUDA graph operations
  setRuntimeCallbacks(subscriber, /*enable=*/true);    // CUDA Runtime API calls
  setMemcpyCallbacks(subscriber, /*enable=*/true);     // Memory transfer operations
  setDriverCallbacks(subscriber, /*enable=*/true);     // CUDA Driver API calls
}

void CuptiProfiler::CuptiProfilerPimpl::doFlush() {
  // cuptiActivityFlushAll returns the activity records associated with all
  // contexts/streams.
  // This is a blocking call but it doesn’t issue any CUDA synchronization calls
  // implicitly thus it’s not guaranteed that all activities are completed on
  // the underlying devices.
  // We do an "opportunistic" synchronization here to try to ensure that all
  // activities are completed on the current context.
  // If the current context is not set, we don't do any synchronization.
  CUcontext cuContext = nullptr;
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext) {
    cuda::ctxSynchronize<true>();
  }
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepMs=*/10,
      /*flush=*/[]() {
        cupti::activityFlushAll<true>(
            /*flag=*/0);
      });
  // CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is used to ensure that even incomplete
  // activities are flushed so that the next profiling session can start with
  // new activities.
  cupti::activityFlushAll<true>(/*flag=*/CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

/**
 * Stop the CUPTI profiler.
 * 
 * This method cleans up the CUPTI profiler, disables callbacks and activity tracking,
 * and finalizes any outstanding resources. It should be called at the end of a
 * profiling session to ensure proper cleanup.
 */
void CuptiProfiler::CuptiProfilerPimpl::doStop() {
  // Cleanup based on the profiling mode
  if (profiler.isPCSamplingEnabled()) {
    // Disable PC sampling mode
    profiler.disablePCSampling();
    
    // Finalize PC sampling for the current context
    CUcontext cuContext = nullptr;
    cuda::ctxGetCurrent<false>(&cuContext);
    if (cuContext)
      pcSampling.finalize(cuContext);
    
    // Disable resources and activities related to PC sampling
    setResourceCallbacks(subscriber, /*enable=*/false);
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    // Disable concurrent kernel tracking for regular profiling
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  
  // Disable memory copy activity tracking
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_MEMCPY);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_MEMCPY2);
  
  // Disable all callbacks
  setGraphCallbacks(subscriber, /*enable=*/false);
  setRuntimeCallbacks(subscriber, /*enable=*/false);
  setMemcpyCallbacks(subscriber, /*enable=*/false);
  setDriverCallbacks(subscriber, /*enable=*/false);
  
  // Unsubscribe from CUPTI and finalize
  cupti::unsubscribe<true>(subscriber);
  cupti::finalize<true>();
}

CuptiProfiler::CuptiProfiler() {
  pImpl = std::make_unique<CuptiProfilerPimpl>(*this);
}

CuptiProfiler::~CuptiProfiler() = default;

} // namespace proton
