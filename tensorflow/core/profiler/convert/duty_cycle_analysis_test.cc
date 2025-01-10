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
#include "tensorflow/core/profiler/convert/duty_cycle_analysis.h"

#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::XSpace;

TEST(DutyCycleAnalysisTest, ComputeTpuDutyCycleAnalysis) {
  XSpace space;
  XPlaneBuilder tpu0(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0));
  CoreDetails tpu0_core_details;
  tpu0_core_details.set_local_chip_id(0);
  tpu0.AddStatValue(
      *tpu0.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCoreDetails)),
      tpu0_core_details);
  XLineBuilder tpu0_op_line = tpu0.GetOrCreateLine(0);
  tpu0_op_line.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&tpu0, &tpu0_op_line, "event.0", /*offset_ps=*/0,
               /*duration_ps=*/100);

  XPlaneBuilder tpu1(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/1, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0));
  CoreDetails tpu1_core_details;
  tpu1_core_details.set_local_chip_id(1);
  tpu1.AddStatValue(
      *tpu1.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCoreDetails)),
      tpu1_core_details);
  XLineBuilder tpu1_op_line = tpu1.GetOrCreateLine(0);
  tpu1_op_line.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&tpu1, &tpu1_op_line, "event.0", /*offset_ps=*/100,
               /*duration_ps=*/100);

  DutyCycleAnalysis results = ComputeTpuDutyCycleAnalysis(space);
  EXPECT_EQ(results.busy_time_ps, 200);
  EXPECT_EQ(results.idle_time_ps, 0);
  EXPECT_EQ(results.duty_cycle, 1.0);
}

TEST(DutyCycleAnalysisTest, ComputeTpuDutyCycleAnalysisWithHloModule) {
  XSpace space;
  XPlaneBuilder tpu(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0));
  CoreDetails core_details;
  core_details.set_local_chip_id(0);
  tpu.AddStatValue(
      *tpu.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCoreDetails)),
      core_details);
  XLineBuilder op_line = tpu.GetOrCreateLine(0);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&tpu, &op_line, "call.0", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  XLineBuilder tpu_module_line = tpu.GetOrCreateLine(1);
  tpu_module_line.SetName(tsl::profiler::kXlaModuleLineName);
  CreateXEvent(&tpu, &tpu_module_line, "event.0", /*offset_ps=*/0,
               /*duration_ps=*/30);

  DutyCycleAnalysis results = ComputeTpuDutyCycleAnalysis(space);
  EXPECT_EQ(results.busy_time_ps, 10);
  EXPECT_EQ(results.idle_time_ps, 20);
  EXPECT_NEAR(results.duty_cycle, 0.3333, 0.0001);
}

TEST(DutyCycleAnalysisTest, ComputeTpuDutyCycleAnalysisWithOffDutyOp) {
  XSpace space;
  XPlaneBuilder tpu(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0));
  CoreDetails core_details;
  core_details.set_local_chip_id(0);
  tpu.AddStatValue(
      *tpu.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCoreDetails)),
      core_details);
  XLineBuilder op_line = tpu.GetOrCreateLine(0);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&tpu, &op_line, "event.0", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed}});
  CreateXEvent(&tpu, &op_line, "event.1", /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  CreateXEvent(&tpu, &op_line, "event.2", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&tpu, &op_line, "event.3", /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloOutfeed}});

  DutyCycleAnalysis results = ComputeTpuDutyCycleAnalysis(space);
  EXPECT_EQ(results.busy_time_ps, 20);
  EXPECT_EQ(results.idle_time_ps, 20);
  EXPECT_NEAR(results.duty_cycle, 0.5, 0.0001);
}

TEST(DutyCycleAnalysisTest, ComputeTpuDutyCycleAnalysisWithMultipleCores) {
  XSpace space;
  XPlaneBuilder core0(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0));
  CoreDetails core0_details;
  core0_details.set_local_chip_id(0);
  core0.AddStatValue(
      *core0.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCoreDetails)),
      core0_details);
  XLineBuilder core0_op_line = core0.GetOrCreateLine(0);
  core0_op_line.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&core0, &core0_op_line, "event.0", /*offset_ps=*/10,
               /*duration_ps=*/10);
  CreateXEvent(&core0, &core0_op_line, "event.2", /*offset_ps=*/30,
               /*duration_ps=*/10);

  XPlaneBuilder core1(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/1, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0));
  CoreDetails core1_details;
  core1_details.set_local_chip_id(0);
  core1.AddStatValue(
      *core1.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCoreDetails)),
      core1_details);
  XLineBuilder core1_op_line = core1.GetOrCreateLine(0);
  core1_op_line.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&core1, &core1_op_line, "event.1", /*offset_ps=*/20,
               /*duration_ps=*/10);
  CreateXEvent(&core1, &core1_op_line, "event.2", /*offset_ps=*/30,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloOutfeed}});

  DutyCycleAnalysis results = ComputeTpuDutyCycleAnalysis(space);
  EXPECT_EQ(results.busy_time_ps, 30);
  EXPECT_EQ(results.idle_time_ps, 0);
  EXPECT_EQ(results.duty_cycle, 1.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
