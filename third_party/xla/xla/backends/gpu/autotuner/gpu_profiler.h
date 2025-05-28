/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"

namespace xla {
namespace gpu {

namespace se = stream_executor;

class GpuProfiler : public Profiler {
 public:
  explicit GpuProfiler(se::StreamExecutor* stream_executor,
                       ProfileOptions options)
      : stream_executor_(stream_executor),
        allocator_(std::make_unique<se::StreamExecutorMemoryAllocator>(
            stream_executor_)),
        stream_(
            allocator_->GetStream(stream_executor_->device_ordinal()).value()),
        options_(options) {}

  absl::StatusOr<std::vector<ProfileResult>> ProfileWithSharedBuffers(
      std::vector<std::unique_ptr<Executable>> executables) override;

 private:
  absl::StatusOr<ExecutionOutput> Execute(Executable* executable,
                                          std::vector<ExecutionInput> inputs,
                                          ExecutionProfile* profile);

  absl::StatusOr<RedzoneBuffers> CreateInputBuffers(
      const Executable* executable);

  absl::StatusOr<ProfileResult> ProfileInternal(Executable* executable,
                                                RedzoneBuffers& buffers);

  stream_executor::StreamExecutor* stream_executor_;
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
  se::Stream* stream_;
  ProfileOptions options_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_
