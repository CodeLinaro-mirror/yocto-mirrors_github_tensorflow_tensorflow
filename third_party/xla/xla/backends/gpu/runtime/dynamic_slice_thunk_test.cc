/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.pb.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class DummyThunk : public Thunk {
 public:
  explicit DummyThunk(Kind kind, const Thunk::ThunkInfo& info)
      : Thunk(kind, info) {}
  ~DummyThunk() override = default;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

using DynamicSliceThunkTest = HloHardwareIndependentTestBase;
using ::testing::NotNull;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

std::string GetPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

se::StreamExecutor* GpuExecutor() {
  stream_executor::Platform* platform =
      se::PlatformManager::PlatformWithName(GetPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}
void CheckProtoRoundTrip(const DynamicSliceThunk& thunk,
                         const DynamicSliceThunkProto& proto) {
  std::vector<BufferAllocation> buffer_allocations;
  for (int i = 0; i < 10; ++i) {
    buffer_allocations.push_back(BufferAllocation(
        /*index=*/i, /*size=*/1024, /*color=*/0));
  }

  std::vector<BufferAllocation> fake_allocations_span;
  const auto& arguments = thunk.get_arguments();
  for (int i = 0; i < arguments.size(); ++i) {
    if (arguments[i].has_value()) {
      fake_allocations_span.push_back(
          BufferAllocation(i, arguments[i].value().allocation()->size(), 0));
    }
  }

  Thunk::DeserializerWithCustomAllocations deserializer =
      [](const ThunkProto& thunk_proto,
         absl::Span<const BufferAllocation> fake_allocations_span)
      -> absl::StatusOr<std::unique_ptr<Thunk>> {
    ThunkSequenceProto thunk_sequence_proto;
    *thunk_sequence_proto.add_thunks() = thunk_proto;
    TF_ASSIGN_OR_RETURN(ThunkSequence sequence,
                        DeserializeThunkSequenceProto(
                            thunk_sequence_proto, fake_allocations_span,
                            /*hlo_module=*/nullptr,
                            /*platform_name=*/"TEST_PLATFORM",
                            /*gpu_compute_capability=*/{}));
    return std::move(sequence.front());
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk_from_proto,
      DynamicSliceThunk::FromProto(Thunk::ThunkInfo(), proto,
                                   /*buffer_allocations=*/buffer_allocations,
                                   deserializer));
  TF_ASSERT_OK_AND_ASSIGN(auto proto_roundtrip, thunk_from_proto->ToProto());
  auto dynamic_slice_thunk_proto_roundtrip =
      proto_roundtrip.dynamic_slice_thunk();
  auto proto_no_ids = proto;
  // Hlo ids are expected to be different after roundtrip, thus we drop them
  // from comparison.
  proto_no_ids.mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_init()
      ->mutable_hlo_module()
      ->clear_id();
  proto_no_ids.mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_update()
      ->mutable_hlo_module()
      ->clear_id();
  for (auto& module_with_config :
       *proto_no_ids.mutable_offset_as_function_of_indvar_modules_metadata()
            ->mutable_extracted_offset_modules()) {
    module_with_config.mutable_hlo_module()->clear_id();
  }

  dynamic_slice_thunk_proto_roundtrip
      .mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_init()
      ->mutable_hlo_module()
      ->clear_id();
  dynamic_slice_thunk_proto_roundtrip
      .mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_update()
      ->mutable_hlo_module()
      ->clear_id();
  for (auto& module_with_config :
       *dynamic_slice_thunk_proto_roundtrip
            .mutable_offset_as_function_of_indvar_modules_metadata()
            ->mutable_extracted_offset_modules()) {
    module_with_config.mutable_hlo_module()->clear_id();
  }

  EXPECT_THAT(dynamic_slice_thunk_proto_roundtrip, EqualsProto(proto_no_ids));
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Ret<ffi::AnyBuffer>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "CUDA",
                         kMemcpy);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "ROCM",
                         kMemcpy);

TEST_F(DynamicSliceThunkTest, SlicedMemcpy) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 8;
  int64_t dst_count = 8 * 8;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * dst_count;

  // Step 1:
  // Prepare embedded and dynamic slice thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_src_fake(&fake_allocations.back(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  fake_allocations.emplace_back(/*index=*/1, dst_length, /*color=*/0);
  BufferAllocation::Slice slice_dst(&fake_allocations.back(), 0, dst_length);

  BufferAllocation alloc_offset_0(/*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_0(&alloc_offset_0, 0, offset_length);

  BufferAllocation alloc_offset_1(/*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_1(&alloc_offset_1, 0, offset_length);

  BufferAllocation alloc_offset_2(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_2(&alloc_offset_2, 0, offset_length);

  BufferAllocation alloc_offset_3(/*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_3(&alloc_offset_3, 0, offset_length);

  // Preparing custom call thunk: setting up call target and operands + results
  // buffers.
  TF_ASSERT_OK_AND_ASSIGN(
      auto registration,
      xla::ffi::FindHandler("__xla_test$$memcpy", GetPlatformName()));

  std::vector<NullableShapedSlice> operands{ShapedSlice{
      slice_src_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8})}};
  std::vector<NullableShapedSlice> results{
      ShapedSlice{slice_dst, ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  TF_ASSERT_OK_AND_ASSIGN(
      seq.emplace_back(),
      CustomCallThunk::Create(Thunk::ThunkInfo(), "__xla_test$$memcpy",
                              registration.bundle, operands, results,
                              /*attributes=*/ffi::AttributesMap(),
                              /*called_computation=*/nullptr,
                              /*gpu_compute_capability=*/{}));

  // Wrapping dynamic slice thunk around the custom call thunk.
  std::vector<DynamicSliceThunk::Offset> slice_offsets{
      slice_offset_0, slice_offset_1, slice_offset_2, slice_offset_3};
  DynamicSliceThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_offsets, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 8}), std::nullopt},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 8, 8}), std::nullopt},
      {S64, std::nullopt});

  // Step 2:
  // Execute dynamic slice thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,8]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,8,8]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:10], [0:8]}

  // Preparing memory for thunk arguments.
  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) {
    src_arr[i] = i;
  }
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceAddress<int64_t> offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> offset_arr{3, 5, 2, 0};
  TF_ASSERT_OK(stream->Memcpy(&offset_0, &offset_arr[0], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_1, &offset_arr[1], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_2, &offset_arr[2], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_3, &offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, offset_0, offset_1, offset_2, offset_3}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `dst` data back to host for verification.
  std::vector<int32_t> out(dst_count, 0);
  TF_ASSERT_OK(stream->Memcpy(out.data(), dst, dst_length));

  // Verifying that the right slice of `src` was copied to `dst`.
  std::vector<int32_t> ref(dst_count, 0);
  int64_t offset_val =
      offset_arr[3] +
      8 * (offset_arr[2] + 10 * (offset_arr[1] + 8 * offset_arr[0]));
  std::copy(src_arr.begin() + offset_val,
            src_arr.begin() + offset_val + dst_count, ref.begin());
  ASSERT_EQ(out, ref);
}

TEST_F(DynamicSliceThunkTest, SlicedOutputMemcpy) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 2;
  int64_t dst_count = 2 * 2 * 2 * 2;
  int64_t slice_count = 2 * 2;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * slice_count;

  // Step 1:
  // Prepare embedded and dynamic slice thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_src_fake(&fake_allocations.back(), 0,
                                         slice_length);

  fake_allocations.emplace_back(/*index=*/1, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_fake(&fake_allocations.back(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  BufferAllocation alloc_dst(/*index=*/1, dst_length, /*color=*/0);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, dst_length);

  BufferAllocation alloc_src_offset_0(/*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_0(&alloc_src_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_1(/*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_1(&alloc_src_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_2(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_2(&alloc_src_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_3(/*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_3(&alloc_src_offset_3, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_0(/*index=*/6, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_0(&alloc_dst_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_1(/*index=*/7, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_1(&alloc_dst_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_2(/*index=*/8, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_2(&alloc_dst_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_3(/*index=*/9, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_3(&alloc_dst_offset_3, 0,
                                             offset_length);

  // Preparing custom call thunk: setting up call target and operands + results
  // buffers.
  TF_ASSERT_OK_AND_ASSIGN(
      auto registration,
      xla::ffi::FindHandler("__xla_test$$memcpy", GetPlatformName()));

  std::vector<NullableShapedSlice> operands{ShapedSlice{
      slice_src_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};
  std::vector<NullableShapedSlice> results{ShapedSlice{
      slice_dst_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  TF_ASSERT_OK_AND_ASSIGN(
      seq.emplace_back(),
      CustomCallThunk::Create(Thunk::ThunkInfo(), "__xla_test$$memcpy",
                              registration.bundle, operands, results,
                              /*attributes=*/ffi::AttributesMap(),
                              /*called_computation=*/nullptr,
                              /*gpu_compute_capability=*/{}));

  // Wrapping dynamic slice thunk around the custom call thunk.
  std::vector<DynamicSliceThunk::Offset> slice_src_offsets{
      slice_src_offset_0, slice_src_offset_1, slice_src_offset_2,
      slice_src_offset_3};
  std::vector<DynamicSliceThunk::Offset> slice_dst_offsets{
      slice_dst_offset_0, slice_dst_offset_1, slice_dst_offset_2,
      slice_dst_offset_3};
  DynamicSliceThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_src_offsets, slice_dst_offsets},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2, 2, 2})},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2})},
      {S64, S64});

  // Step 2:
  // Execute dynamic slice thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,2]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:4], [0:2]}
  //
  // Given a `dst` tensor of shape s32[2,2,2,2]{3,2,1,0}
  // The `dst` slice that we want to copy into will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(dst), slice={[1:2], [1:2], [0:2], [0:2]}

  // Preparing memory for thunk arguments.
  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) {
    src_arr[i] = i;
  }
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceAddress<int64_t> src_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> src_offset_arr{3, 5, 2, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_0, &src_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_1, &src_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_2, &src_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_3, &src_offset_arr[3], offset_length));

  se::DeviceAddress<int64_t> dst_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> dst_offset_arr{1, 1, 0, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_0, &dst_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_1, &dst_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_2, &dst_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_3, &dst_offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, src_offset_0, src_offset_1, src_offset_2, src_offset_3,
       dst_offset_0, dst_offset_1, dst_offset_2, dst_offset_3},
      0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `dst` data back to host for verification.
  std::vector<int32_t> out(dst_count, 0);
  TF_ASSERT_OK(stream->Memcpy(out.data(), dst, dst_length));

  // Verifying that the right slice of `src` was copied to `dst`.
  std::vector<int32_t> ref(dst_count, 0);
  int64_t src_offset_val =
      src_offset_arr[3] +
      2 * (src_offset_arr[2] +
           10 * (src_offset_arr[1] + 8 * src_offset_arr[0]));
  int64_t dst_offset_val =
      dst_offset_arr[3] +
      2 * (dst_offset_arr[2] + 2 * (dst_offset_arr[1] + 2 * dst_offset_arr[0]));
  std::copy(src_arr.begin() + src_offset_val,
            src_arr.begin() + src_offset_val + slice_count,
            ref.begin() + dst_offset_val);
  ASSERT_EQ(out, ref);
}

TEST_F(DynamicSliceThunkTest, SlicedMemcpyOOB) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 2;
  int64_t dst_count = 2 * 2 * 2 * 2;
  int64_t slice_count = 2 * 2;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * slice_count;

  // Step 1:
  // Prepare embedded and dynamic slice thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_src_fake(&fake_allocations.back(), 0,
                                         slice_length);

  fake_allocations.emplace_back(/*index=*/1, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_fake(&fake_allocations.back(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  BufferAllocation alloc_dst(/*index=*/1, dst_length, /*color=*/0);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, dst_length);

  BufferAllocation alloc_src_offset_0(/*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_0(&alloc_src_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_1(/*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_1(&alloc_src_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_2(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_2(&alloc_src_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_3(/*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_3(&alloc_src_offset_3, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_0(/*index=*/6, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_0(&alloc_dst_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_1(/*index=*/7, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_1(&alloc_dst_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_2(/*index=*/8, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_2(&alloc_dst_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_3(/*index=*/9, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_3(&alloc_dst_offset_3, 0,
                                             offset_length);

  // Preparing custom call thunk: setting up call target and operands + results
  // buffers.
  TF_ASSERT_OK_AND_ASSIGN(
      auto registration,
      xla::ffi::FindHandler("__xla_test$$memcpy", GetPlatformName()));

  std::vector<NullableShapedSlice> operands{ShapedSlice{
      slice_src_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};
  std::vector<NullableShapedSlice> results{ShapedSlice{
      slice_dst_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  TF_ASSERT_OK_AND_ASSIGN(
      seq.emplace_back(),
      CustomCallThunk::Create(Thunk::ThunkInfo(), "__xla_test$$memcpy",
                              registration.bundle, operands, results,
                              /*attributes=*/ffi::AttributesMap(),
                              /*called_computation=*/nullptr,
                              /*gpu_compute_capability=*/{}));

  // Wrapping dynamic slice thunk around the custom call thunk.
  std::vector<DynamicSliceThunk::Offset> slice_src_offsets{
      slice_src_offset_0, slice_src_offset_1, slice_src_offset_2,
      slice_src_offset_3};
  std::vector<DynamicSliceThunk::Offset> slice_dst_offsets{
      slice_dst_offset_0, slice_dst_offset_1, slice_dst_offset_2,
      slice_dst_offset_3};
  DynamicSliceThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_src_offsets, slice_dst_offsets},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2, 2, 2})},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2})},
      {S64, S64});

  // Step 2:
  // Execute dynamic slice thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,2]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:4], [0:2]}
  //
  // Given a `dst` tensor of shape s32[2,2,2,2]{3,2,1,0}
  // The `dst` slice that we want to copy into will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(dst), slice={[1:2], [1:2], [0:2], [0:2]}

  // Preparing memory for thunk arguments.
  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) {
    src_arr[i] = i;
  }
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceAddress<int64_t> src_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> src_ref_offset_arr{3, 5, 2, 0};
  std::vector<int64_t> src_offset_arr{3, 5, 2, -3};
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_0, &src_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_1, &src_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_2, &src_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_3, &src_offset_arr[3], offset_length));

  se::DeviceAddress<int64_t> dst_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> dst_ref_offset_arr{1, 1, 0, 0};
  std::vector<int64_t> dst_offset_arr{3, 2, 5, -4};
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_0, &dst_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_1, &dst_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_2, &dst_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_3, &dst_offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, src_offset_0, src_offset_1, src_offset_2, src_offset_3,
       dst_offset_0, dst_offset_1, dst_offset_2, dst_offset_3},
      0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `dst` data back to host for verification.
  std::vector<int32_t> out(dst_count, 0);
  TF_ASSERT_OK(stream->Memcpy(out.data(), dst, dst_length));

  // Verifying that the right slice of `src` was copied to `dst`.
  std::vector<int32_t> ref(dst_count, 0);
  int64_t src_offset_val =
      src_ref_offset_arr[3] +
      2 * (src_ref_offset_arr[2] +
           10 * (src_ref_offset_arr[1] + 8 * src_ref_offset_arr[0]));
  int64_t dst_offset_val =
      dst_ref_offset_arr[3] +
      2 * (dst_ref_offset_arr[2] +
           2 * (dst_ref_offset_arr[1] + 2 * dst_ref_offset_arr[0]));
  std::copy(src_arr.begin() + src_offset_val,
            src_arr.begin() + src_offset_val + slice_count,
            ref.begin() + dst_offset_val);
  ASSERT_EQ(out, ref);
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithNullopt) {
  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::nullopt;
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto,
      SerializeOptionalDynamicSliceOffsetsToProto(offsets_item, std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_offsets,
      DeserializeOptionalDynamicSliceOffsetsFromProto(proto, {}, std::nullopt));
  EXPECT_FALSE(deserialized_offsets.has_value());
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithConstOffset) {
  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::vector<DynamicSliceThunk::Offset>{123l};
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto,
      SerializeOptionalDynamicSliceOffsetsToProto(offsets_item, std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_offsets,
      DeserializeOptionalDynamicSliceOffsetsFromProto(proto, {}, std::nullopt));
  ASSERT_TRUE(deserialized_offsets.has_value());
  ASSERT_EQ(deserialized_offsets->size(), 1);
  EXPECT_EQ(std::get<int64_t>((*deserialized_offsets)[0]), 123l);
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithSliceOffset) {
  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(0, 1024, 0);
  BufferAllocation::Slice slice(&allocations.back(), 128, 256);
  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::vector<DynamicSliceThunk::Offset>{slice};
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto,
      SerializeOptionalDynamicSliceOffsetsToProto(offsets_item, std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized_offsets,
                          DeserializeOptionalDynamicSliceOffsetsFromProto(
                              proto, allocations, std::nullopt));
  ASSERT_TRUE(deserialized_offsets.has_value());
  ASSERT_EQ(deserialized_offsets->size(), 1);
  auto deserialized_slice =
      std::get<BufferAllocation::Slice>((*deserialized_offsets)[0]);
  EXPECT_EQ(deserialized_slice.allocation(), &allocations.back());
  EXPECT_EQ(deserialized_slice.offset(), 128);
  EXPECT_EQ(deserialized_slice.size(), 256);
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithHloModuleOffset) {
  const char* hlo_text = R"(
      HloModule test_module
      ENTRY main {
        ROOT c = f32[] constant(0.0)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(hlo_text));
  HloModule* hlo_module_ptr = hlo_module.get();

  std::vector<std::unique_ptr<HloModule>> modules;
  modules.push_back(std::move(hlo_module));

  const char* indvar_init_hlo = R"(
      HloModule indvar_init
      ENTRY main {
        ROOT c0 = s32[] constant(0)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_init_module,
                          ParseAndReturnUnverifiedModule(indvar_init_hlo));

  const char* indvar_update_hlo = R"(
      HloModule indvar_update
      ENTRY main {
        p0 = s32[] parameter(0)
        c1 = s32[] constant(1)
        ROOT add = s32[] add(p0, c1)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_update_module,
                          ParseAndReturnUnverifiedModule(indvar_update_hlo));

  DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata metadata(
      std::move(indvar_init_module), std::move(indvar_update_module),
      std::move(modules));

  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::vector<DynamicSliceThunk::Offset>{hlo_module_ptr};

  TF_ASSERT_OK_AND_ASSIGN(auto proto,
                          SerializeOptionalDynamicSliceOffsetsToProto(
                              offsets_item, std::move(metadata)));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module2,
                          ParseAndReturnUnverifiedModule(hlo_text));
  std::vector<std::unique_ptr<HloModule>> modules2;
  modules2.push_back(std::move(hlo_module2));
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_init_module2,
                          ParseAndReturnUnverifiedModule(indvar_init_hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_update_module2,
                          ParseAndReturnUnverifiedModule(indvar_update_hlo));
  DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata metadata2(
      std::move(indvar_init_module2), std::move(indvar_update_module2),
      std::move(modules2));
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized_offsets,
                          DeserializeOptionalDynamicSliceOffsetsFromProto(
                              proto, {}, std::move(metadata2)));
  ASSERT_TRUE(deserialized_offsets.has_value());
  ASSERT_EQ(deserialized_offsets->size(), 1);
  EXPECT_TRUE(std::holds_alternative<HloModule*>((*deserialized_offsets)[0]));
  EXPECT_NE(std::get<HloModule*>((*deserialized_offsets)[0]), nullptr);
  EXPECT_EQ(proto.offsets().offsets(0).hlo_module_offset_idx(), 0);
}

TEST_F(DynamicSliceThunkTest, TransformNested) {
  auto seq = std::make_unique<ThunkSequence>();
  seq->emplace_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, Thunk::ThunkInfo()));
  DynamicSliceThunk thunk(Thunk::ThunkInfo(),
                          /*embedded_thunk=*/std::move(seq),
                          /*arguments=*/{},
                          /*fake_allocations=*/{},
                          /*offsets=*/{},
                          /*orig_shapes=*/{},
                          /*sliced_shapes=*/{},
                          /*offset_byte_sizes=*/{});

  TF_EXPECT_OK(thunk.TransformNested([](auto) {
    return std::make_unique<DummyThunk>(Thunk::Kind::kCustomCall,
                                        Thunk::ThunkInfo());
  }));

  EXPECT_THAT(thunk.get_embedded_executor().thunks(), SizeIs(1));
  EXPECT_THAT(thunk.get_embedded_executor().thunks()[0]->kind(),
              Thunk::Kind::kCustomCall);
}

}  // namespace
}  // namespace xla::gpu
