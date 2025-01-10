/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_ANALYSIS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_ANALYSIS_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Duty cycle results for all TPUs.
struct DutyCycleAnalysis {
  uint64_t busy_time_ps = 0;
  uint64_t idle_time_ps = 0;
  double duty_cycle = 0.0;
};

// Computes the TPU duty cycle analysis for the given XSpace.
inline DutyCycleAnalysis ComputeTpuDutyCycleAnalysis(const XSpace& space) {
  DutyCycleAnalysis analysis;
  const std::vector<const XPlane*> device_planes =
      tsl::profiler::FindPlanesWithPrefix(space, kTpuPlanePrefix);
  if (device_planes.empty()) {
    LOG(WARNING) << "No TPU device planes found.";
    return analysis;
  }

  absl::flat_hash_map<uint32_t, DutyCycleTracker> chip_duty_cycle_trackers;
  for (const XPlane* device_plane : device_planes) {
    tsl::profiler::XPlaneVisitor visitor =
        tsl::profiler::CreateTfXPlaneVisitor(device_plane);
    CoreDetails core_details;
    if (auto stat = visitor.GetStat(StatType::kCoreDetails)) {
      absl::string_view core_details_bytes = stat->BytesValue();
      if (!core_details.ParseFromArray(core_details_bytes.data(),
                                       core_details_bytes.size())) {
        LOG(WARNING) << "Failed to parse core details for device plane: "
                     << device_plane->name();
        continue;
      }
    } else {
      LOG(WARNING) << "Failed to get core details for device plane: "
                   << device_plane->name();
      continue;
    }
    DutyCycleTracker& duty_cycle_tracker =
        chip_duty_cycle_trackers[core_details.local_chip_id()];
    visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
      if (line.Name() == tsl::profiler::kXlaOpLineName) {
        line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
          auto hlo_category_stat = event.GetStat(StatType::kHloCategory);
          duty_cycle_tracker.AddInterval(
              tsl::profiler::Timespan(event.OffsetPs(), event.DurationPs()),
              !(hlo_category_stat && tsl::profiler::IsOffDutyOp(
                                         hlo_category_stat->StrOrRefValue())));
        });
      } else if (line.Name() == tsl::profiler::kXlaModuleLineName) {
        line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
          duty_cycle_tracker.AddInterval(
              tsl::profiler::Timespan(event.OffsetPs(), event.DurationPs()),
              /*is_active=*/false);
        });
      }
    });
  }

  double avg_duty_cycle = 0.0;
  for (const auto& [chip_id, duty_cycle_tracker] : chip_duty_cycle_trackers) {
    analysis.busy_time_ps += duty_cycle_tracker.GetActiveTimePs();
    analysis.idle_time_ps += duty_cycle_tracker.GetIdleTimePs();
    avg_duty_cycle += duty_cycle_tracker.DutyCycle();
  }
  analysis.duty_cycle = avg_duty_cycle / chip_duty_cycle_trackers.size();
  return analysis;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_ANALYSIS_H_
