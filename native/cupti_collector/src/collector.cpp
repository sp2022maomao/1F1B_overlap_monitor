#include "overlap_cupti.h"

#include <cuda.h>
#include <cupti.h>
#include <cupti_activity.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr size_t kBufferSize = 8 * 1024 * 1024;
constexpr size_t kBufferAlignment = 8;
constexpr size_t kDefaultMaxRecords = 2 * 1024 * 1024;
constexpr int kClientError = -1;

#if CUDA_VERSION >= 13000
using KernelActivity = CUpti_ActivityKernel10;
#else
using KernelActivity = CUpti_ActivityKernel9;
#endif

struct KernelRecord {
  uint64_t start_ns;
  uint64_t end_ns;
  uint32_t device_id;
  uint32_t stream_id;
  uint32_t correlation_id;
  std::string name;
};

struct ExternalCorrelationRecord {
  uint32_t correlation_id;
  uint64_t external_id;
  uint32_t external_kind;
};

std::mutex g_mutex;
std::vector<KernelRecord> g_kernels;
std::vector<ExternalCorrelationRecord> g_external_correlations;
std::atomic<size_t> g_cupti_dropped{0};
std::atomic<size_t> g_client_dropped{0};
std::atomic<int> g_callback_error{0};
std::atomic<bool> g_started{false};
bool g_callbacks_registered = false;
size_t g_max_records = kDefaultMaxRecords;
std::string g_output_path;
int g_rank = -1;

int CuptiStatus(CUptiResult result) {
  if (result == CUPTI_SUCCESS) {
    return 0;
  }
  const char* message = nullptr;
  cuptiGetResultString(result, &message);
  std::fprintf(stderr, "overlap-cupti: %s\n", message ? message : "unknown CUPTI error");
  return static_cast<int>(result);
}

size_t ReadMaxRecords() {
  const char* value = std::getenv("OVERLAP_CUPTI_MAX_RECORDS");
  if (value == nullptr || *value == '\0') {
    return kDefaultMaxRecords;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
    std::fprintf(stderr, "overlap-cupti: invalid OVERLAP_CUPTI_MAX_RECORDS=%s\n", value);
    return kDefaultMaxRecords;
  }
  return static_cast<size_t>(parsed);
}

void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* max_records) {
  void* allocation = nullptr;
  if (posix_memalign(&allocation, kBufferAlignment, kBufferSize) != 0) {
    *buffer = nullptr;
    *size = 0;
    *max_records = 0;
    g_callback_error.store(kClientError);
    return;
  }
  *buffer = static_cast<uint8_t*>(allocation);
  *size = kBufferSize;
  *max_records = 0;
}

void StoreKernel(const CUpti_Activity* activity) {
  const auto* kernel = reinterpret_cast<const KernelActivity*>(activity);
  if (kernel->start == 0 && kernel->end == 0) {
    g_client_dropped.fetch_add(1);
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_kernels.size() + g_external_correlations.size() >= g_max_records) {
    g_client_dropped.fetch_add(1);
    return;
  }
  g_kernels.push_back(KernelRecord{
      kernel->start,
      kernel->end,
      kernel->deviceId,
      kernel->streamId,
      kernel->correlationId,
      kernel->name ? kernel->name : "",
  });
}

void StoreExternalCorrelation(const CUpti_Activity* activity) {
  const auto* correlation =
      reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(activity);
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_kernels.size() + g_external_correlations.size() >= g_max_records) {
    g_client_dropped.fetch_add(1);
    return;
  }
  g_external_correlations.push_back(ExternalCorrelationRecord{
      correlation->correlationId,
      correlation->externalId,
      static_cast<uint32_t>(correlation->externalKind),
  });
}

void CUPTIAPI BufferCompleted(
    CUcontext context,
    uint32_t stream_id,
    uint8_t* buffer,
    size_t,
    size_t valid_size) {
  if (valid_size > 0) {
    CUpti_Activity* activity = nullptr;
    while (true) {
      const CUptiResult result =
          cuptiActivityGetNextRecord(buffer, valid_size, &activity);
      if (result == CUPTI_SUCCESS) {
        if (activity->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
          StoreKernel(activity);
        } else if (activity->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
          StoreExternalCorrelation(activity);
        }
      } else if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        g_callback_error.store(static_cast<int>(result));
        break;
      }
    }
  }

  size_t dropped = 0;
  const CUptiResult dropped_result =
      cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);
  if (dropped_result == CUPTI_SUCCESS) {
    g_cupti_dropped.fetch_add(dropped);
  } else {
    g_callback_error.store(static_cast<int>(dropped_result));
  }
  std::free(buffer);
}

void WriteJsonString(FILE* output, const std::string& value) {
  std::fputc('"', output);
  for (const unsigned char character : value) {
    switch (character) {
      case '\\':
        std::fputs("\\\\", output);
        break;
      case '"':
        std::fputs("\\\"", output);
        break;
      case '\n':
        std::fputs("\\n", output);
        break;
      case '\r':
        std::fputs("\\r", output);
        break;
      case '\t':
        std::fputs("\\t", output);
        break;
      default:
        if (character < 0x20) {
          std::fprintf(output, "\\u%04x", character);
        } else {
          std::fputc(character, output);
        }
    }
  }
  std::fputc('"', output);
}

const char* ExternalKindName(uint32_t kind) {
  if (kind == static_cast<uint32_t>(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0)) {
    return "custom0";
  }
  return "unknown";
}

int WriteTrace() {
  FILE* output = std::fopen(g_output_path.c_str(), "w");
  if (output == nullptr) {
    std::fprintf(stderr, "overlap-cupti: cannot open %s: %s\n", g_output_path.c_str(), std::strerror(errno));
    return kClientError;
  }

  const int process_id = static_cast<int>(getpid());
  std::fprintf(
      output,
      "{\"process_id\":%d,\"rank\":%d,\"record_kind\":\"trace_metadata\",\"schema_version\":1}\n",
      process_id,
      g_rank);
  for (const auto& correlation : g_external_correlations) {
    std::fprintf(
        output,
        "{\"correlation_id\":%u,\"external_id\":%llu,\"external_kind\":\"%s\",\"external_kind_id\":%u,\"record_kind\":\"external_correlation\",\"schema_version\":1}\n",
        correlation.correlation_id,
        static_cast<unsigned long long>(correlation.external_id),
        ExternalKindName(correlation.external_kind),
        correlation.external_kind);
  }
  for (const auto& kernel : g_kernels) {
    std::fprintf(
        output,
        "{\"correlation_id\":%u,\"device_id\":%u,\"end_ns\":%llu,\"name\":",
        kernel.correlation_id,
        kernel.device_id,
        static_cast<unsigned long long>(kernel.end_ns));
    WriteJsonString(output, kernel.name);
    std::fprintf(
        output,
        ",\"process_id\":%d,\"rank\":%d,\"record_kind\":\"kernel\",\"schema_version\":1,\"start_ns\":%llu,\"stream_id\":%u}\n",
        process_id,
        g_rank,
        static_cast<unsigned long long>(kernel.start_ns),
        kernel.stream_id);
  }
  const size_t dropped = g_cupti_dropped.load() + g_client_dropped.load();
  std::fprintf(
      output,
      "{\"client_dropped_records\":%llu,\"cupti_dropped_records\":%llu,\"dropped_records\":%llu,\"record_kind\":\"collector_summary\",\"schema_version\":1}\n",
      static_cast<unsigned long long>(g_client_dropped.load()),
      static_cast<unsigned long long>(g_cupti_dropped.load()),
      static_cast<unsigned long long>(dropped));
  const int close_result = std::fclose(output);
  return close_result == 0 ? 0 : kClientError;
}

int EnableActivities() {
  const CUpti_ActivityKind kinds[] = {
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_RUNTIME,
      CUPTI_ACTIVITY_KIND_DRIVER,
      CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
  };
  for (const auto kind : kinds) {
    const int result = CuptiStatus(cuptiActivityEnable(kind));
    if (result != 0) {
      return result;
    }
  }
  return 0;
}

void DisableActivities() {
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

}  // namespace

extern "C" int overlap_cupti_start(const char* output_path, int rank) {
  if (output_path == nullptr || *output_path == '\0' || g_started.exchange(true)) {
    return kClientError;
  }
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_kernels.clear();
    g_external_correlations.clear();
    g_kernels.reserve(65536);
    g_external_correlations.reserve(4096);
  }
  g_output_path = output_path;
  g_rank = rank;
  g_max_records = ReadMaxRecords();
  g_cupti_dropped.store(0);
  g_client_dropped.store(0);
  g_callback_error.store(0);

  if (!g_callbacks_registered) {
    const int result = CuptiStatus(
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted));
    if (result != 0) {
      g_started.store(false);
      return result;
    }
    g_callbacks_registered = true;
  }
  const int result = EnableActivities();
  if (result != 0) {
    DisableActivities();
    g_started.store(false);
    return result;
  }
  return 0;
}

extern "C" int overlap_cupti_stop(void) {
  if (!g_started.exchange(false)) {
    return 0;
  }
  const int flush_result = CuptiStatus(cuptiActivityFlushAll(0));
  DisableActivities();
  const int callback_result = g_callback_error.load();
  const int write_result = WriteTrace();
  if (flush_result != 0) {
    return flush_result;
  }
  if (callback_result != 0) {
    return callback_result;
  }
  return write_result;
}

extern "C" int overlap_cupti_push_external_id(uint64_t external_id) {
  if (!g_started.load()) {
    return kClientError;
  }
  return CuptiStatus(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, external_id));
}

extern "C" int overlap_cupti_pop_external_id(uint64_t* external_id) {
  if (!g_started.load() || external_id == nullptr) {
    return kClientError;
  }
  return CuptiStatus(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, external_id));
}
