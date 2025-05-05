/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class GpuPerformanceModel : public GpuPerformanceModelBase {
 public:
  /// Constructor.
  ///
  /// Lifetime to all references to this constructor must live at least as long
  ///
  /// @param[in] device_info Device description.
  /// @param[in] fusion_analysis_cache Cache for fusion analysis.
  /// @param[in] gpu_performance_model_cache Cache for the model.
  /// as the model.
  GpuPerformanceModel(const se::DeviceDescription& device_info,
                      HloFusionAnalysisCache& fusion_analysis_cache,
                      GpuPerformanceModelCache& gpu_performance_model_cache);

  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis);
  /// Static version of EstimateRunTimeForInstruction.
  /// This version does not use the caches so is only useful for one shot
  /// estimations.
  static EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr, const se::DeviceDescription& device_info,
      const GpuHloCostAnalysis* cost_analysis);

  EstimateRunTimeData EstimateRunTimeForInstructionCached(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis);

  // TODO(shyshkov): Unify interface with EstimateRunTimeForInstruction.
  absl::Duration EstimateRunTimeForFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      bool producer_writes_side_output = false);

  absl::Duration EstimateRunTimeForFusionCached(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis);

  RunTimes EstimateRunTimes(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      absl::Span<const HloInstruction* const> fused_consumers = {});

  RunTimes EstimateRunTimesForMultiOutputFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const GpuHloCostAnalysis* cost_analysis);

  // Static version of EstimateRunTimesForMultiOutputFusion.
  // Since runtime for multi output fusion does not depend on the fusion
  // analysis cache this function can be invoked without it.
  static RunTimes EstimateRunTimesForMultiOutputFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const se::DeviceDescription& device_info,
      const GpuHloCostAnalysis* cost_analysis);

  // Writes estimated execution time to FusionBackendConfig.reification_cost.
  void RecordEstimatedRunTime(HloInstruction* instruction,
                              const GpuHloCostAnalysis* cost_analysis);

 private:
  const se::DeviceDescription& device_info_;
  HloFusionAnalysisCache& fusion_analysis_cache_;
  GpuPerformanceModelCache& gpu_performance_model_cache_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_
