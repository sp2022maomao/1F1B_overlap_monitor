#ifndef OVERLAP_CUPTI_H_
#define OVERLAP_CUPTI_H_

#include <stdint.h>

#if defined(_WIN32)
#define OVERLAP_CUPTI_EXPORT __declspec(dllexport)
#else
#define OVERLAP_CUPTI_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Returns zero on success. The caller must ensure outstanding GPU work is
// complete before stop() if a complete trace is required.
OVERLAP_CUPTI_EXPORT int overlap_cupti_start(const char* output_path, int rank);
OVERLAP_CUPTI_EXPORT int overlap_cupti_stop(void);

// External IDs connect application regions to CUDA API and kernel records.
// Calls must be balanced on each calling thread.
OVERLAP_CUPTI_EXPORT int overlap_cupti_push_external_id(uint64_t external_id);
OVERLAP_CUPTI_EXPORT int overlap_cupti_pop_external_id(uint64_t* external_id);

#ifdef __cplusplus
}
#endif

#endif  // OVERLAP_CUPTI_H_
