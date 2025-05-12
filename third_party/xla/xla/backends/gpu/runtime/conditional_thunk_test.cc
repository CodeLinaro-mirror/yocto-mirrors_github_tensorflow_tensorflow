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

#include "xla/backends/gpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

using ::testing::AllOf;
using ::testing::Pointee;
using ::testing::Property;

constexpr ExecutionStreamId kExecutionStreamId{123};
constexpr absl::string_view kProfileAnnotation = "profile_annotation";

Thunk::ThunkInfo CreateThunkInfo() {
  auto thunk_info = Thunk::ThunkInfo();
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;
  return thunk_info;
}

// A dummy `Thunk` that does nothing.
struct DummyThunk : public Thunk {
  DummyThunk() : Thunk(Thunk::Kind::kGemm, CreateThunkInfo()) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

std::unique_ptr<ConditionalThunk> CreateConditionalThunk(
    BufferAllocation* alloc,
    const BufferAllocation::Slice& branch_index_buffer_index,
    std::vector<ThunkSequence> branch_thunk_sequences,
    bool kBranchIndexIsBool) {
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  for (auto& thunk_sequence : branch_thunk_sequences) {
    branch_thunks.push_back(std::make_unique<SequentialThunk>(
        CreateThunkInfo(), std::move(thunk_sequence)));
  }

  return std::make_unique<ConditionalThunk>(
      CreateThunkInfo(), branch_index_buffer_index, std::move(branch_thunks),
      kBranchIndexIsBool);
}

TEST(ConditionalThunkTest, BufferUses) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice branch_index_slice(&alloc, /*offset=*/0,
                                             /*size=*/256);

  ThunkSequence false_sequence;
  false_sequence.push_back(std::make_unique<DummyThunk>());
  false_sequence.push_back(std::make_unique<DummyThunk>());

  ThunkSequence true_sequence;
  true_sequence.push_back(std::make_unique<DummyThunk>());
  true_sequence.push_back(std::make_unique<DummyThunk>());

  std::vector<ThunkSequence> branch_thunk_sequences;
  branch_thunk_sequences.push_back(std::move(false_sequence));
  branch_thunk_sequences.push_back(std::move(true_sequence));

  bool kBranchIndexIsBool = true;
  auto conditional_thunk = CreateConditionalThunk(
      &alloc, branch_index_slice, std::move(branch_thunk_sequences),
      kBranchIndexIsBool);

  EXPECT_EQ(conditional_thunk->branch_index_is_bool(), kBranchIndexIsBool);
  EXPECT_EQ(conditional_thunk->branch_index_buffer(), branch_index_slice);

  auto thunk_matcher = Pointee(Property(&Thunk::kind, Thunk::Kind::kGemm));
  auto branch_matcher = Pointee(Property(
      &SequentialThunk::thunks, ElementsAre(thunk_matcher, thunk_matcher)));
  EXPECT_THAT(conditional_thunk->branch_thunks(),
              ElementsAre(branch_matcher, branch_matcher));
}

TEST(ConditionalThunkTest, ToProto) {
  int64_t slice_offset = 0;
  int64_t slice_size = 256;
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice branch_index_slice(&alloc, slice_offset, slice_size);

  ThunkSequence false_sequence;
  false_sequence.push_back(std::make_unique<DummyThunk>());
  false_sequence.push_back(std::make_unique<DummyThunk>());

  ThunkSequence true_sequence;
  true_sequence.push_back(std::make_unique<DummyThunk>());
  true_sequence.push_back(std::make_unique<DummyThunk>());

  std::vector<ThunkSequence> branch_thunk_sequences;
  branch_thunk_sequences.push_back(std::move(false_sequence));
  branch_thunk_sequences.push_back(std::move(true_sequence));

  bool kBranchIndexIsBool = true;
  auto thunk = CreateConditionalThunk(&alloc, branch_index_slice,
                                      std::move(branch_thunk_sequences),
                                      kBranchIndexIsBool);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk->ToProto());

  auto thunk_info_matcher = Property(
      &ThunkProto::thunk_info,
      AllOf(Property(&ThunkInfoProto::execution_stream_id, kExecutionStreamId),
            Property(&ThunkInfoProto::profile_annotation, kProfileAnnotation)));

  ASSERT_TRUE(proto.has_conditional_thunk());
  auto conditional_thunk = proto.conditional_thunk();
  EXPECT_THAT(proto, thunk_info_matcher);
  ASSERT_TRUE(conditional_thunk.has_branch_index_buffer());
  auto branch_index_buffer = conditional_thunk.branch_index_buffer();
  EXPECT_EQ(branch_index_buffer.offset(), slice_offset);
  EXPECT_EQ(branch_index_buffer.size(), slice_size);
  EXPECT_EQ(branch_index_buffer.buffer_allocation_index(), alloc.index());

  EXPECT_TRUE(conditional_thunk.branch_index_is_bool());

  auto branch_matcher =
      Property(&SequentialThunkProto::thunks,
               ElementsAre(thunk_info_matcher, thunk_info_matcher));

  EXPECT_THAT(proto.conditional_thunk().branch_thunks(),
              ElementsAre(branch_matcher, branch_matcher));
}
}  // namespace
}  // namespace xla::gpu
