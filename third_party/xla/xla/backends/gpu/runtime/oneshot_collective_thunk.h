/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_ONESHOT_COLLECTIVE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_ONESHOT_COLLECTIVE_THUNK_H_

#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// A thunk that runs single-host collective operations in a single shot.
// Assumes multiple devices are present, but all on the same host.
// Basic mode of operation is as follows:
// Prepare:
// - Acquire clique key.
// Initialize:
// - Allocated local buffers and create start/end events for each device
//    (aka per stream).
// ExecuteOnStream:
// - Sync streams on all devices.
// - Run one-shot kernel.
// - Wait for all streams to complete.
class OneshotCollectiveThunk : public Thunk {
 public:
  OneshotCollectiveThunk(Thunk::Kind kind, ThunkInfo info,
                         const CollectiveConfig& collective_config,
                         CollectiveThunk::Buffer buffer)
      : Thunk{kind, info},
        collective_config_(collective_config),
        buffer_(buffer) {}

  // The single host collective thunk actually requires a clique key.
  // But it expects it to be acquired by someone up in the stack.
  // (See class CollectiveThunk::Prepare).
  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) final;

  // Allocate buffers and events as needed for cross device communication.
  absl::Status Initialize(const InitializeParams& params) final;

  // Execute the one-shot kernel on all devices.
  absl::Status ExecuteOnStream(const ExecuteParams& params) final;

  // Returns the clique key for the GPU clique involved in the collective op.
  const GpuCliqueKey& GetCliqueKey() const { return *clique_key_; }

 private:
  // Unique key for the GPU clique involved in the collective op.
  // Initialized during Prepare
  std::optional<GpuCliqueKey> clique_key_{std::nullopt};
  // Reference to collective config being used.
  CollectiveConfig const& collective_config_;

  // Reference to the buffer related information required for the collective.
  CollectiveThunk::Buffer buffer_;
  // Guard access to the local buffer allocations and events across
  // different threads (which control different streams).
  absl::Mutex mutex_;
  // Local buffer allocations to copy input data for the one-shot kernel.
  absl::flat_hash_map<se::StreamExecutor*, se::DeviceMemoryHandle>
      local_buffer_allocs_ ABSL_GUARDED_BY(mutex_);

  // Allocation for signal flags to synchronize blocks on different devices.
  absl::flat_hash_map<se::StreamExecutor*, se::DeviceMemoryHandle>
      signal_flags_allocs_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ONESHOT_COLLECTIVE_THUNK_H_
