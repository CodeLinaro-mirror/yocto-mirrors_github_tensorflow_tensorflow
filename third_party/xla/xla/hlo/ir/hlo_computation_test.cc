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

#include "xla/hlo/ir/hlo_computation.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using HLOComputationTest = HloHardwareIndependentTestBase;

int64_t CountControlEdges(const HloComputation &computation) {
  int64_t count = 0;
  for (const auto &instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

TEST_F(HLOComputationTest, DefUseOrder) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  i0 = f32[100] add(p0, p1)
  i1 = f32[100] multiply(p0, p1)
  i2 = f32[100] divide(p0, p1)
  c1 = f32[100] all-reduce(i0), replica_groups={}, to_apply=sum, channel_id=1
  c2 = f32[100] all-reduce(i1), replica_groups={}, to_apply=sum, channel_id=1
  c3 = f32[100] all-reduce(i2), replica_groups={}, to_apply=sum, channel_id=1
  t = f32[100] add(c1, c2)
  ROOT out = f32[100] add(t, c3)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 0);

  const HloInstruction *root = module->entry_computation()->root_instruction();
  const HloInstruction *t = root->operand(0);   // t = add(c1, c2)
  const HloInstruction *c3 = root->operand(1);  // c3 = all-reduce(i2)...
  EXPECT_EQ(t->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(c3->opcode(), HloOpcode::kAllReduce);

  const HloInstruction *c1 = t->operand(0);
  const HloInstruction *c2 = t->operand(1);
  EXPECT_EQ(c1->opcode(), HloOpcode::kAllReduce);
  EXPECT_EQ(c2->opcode(), HloOpcode::kAllReduce);

  bool found_i0 = false;
  // Verify that i0 is before c1.
  auto post_order = module->entry_computation()->MakeInstructionPostOrder();
  for (const auto &instruction : post_order) {
    if (instruction->name() == "c1") {
      EXPECT_TRUE(found_i0);
    }
    if (instruction->name() == "i0") {
      found_i0 = true;
    }
  }

  // Verify that MakeInstructionPostOrder() is idempotent.
  auto post_order_2 = module->entry_computation()->MakeInstructionPostOrder();
  EXPECT_EQ(post_order, post_order_2);
}

TEST_F(HLOComputationTest, MakeInstructionPostOrder) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  p1 = f32[100] parameter(1)
  i0 = f32[100] add(p0, p1)
  i1 = f32[100] multiply(p0, i0)
  ROOT i2 = f32[100] divide(p1, i1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto post_order = module->entry_computation()->MakeInstructionPostOrder();

  // Verify the order of instructions in the post order.
  bool found_p0 = false;
  bool found_p1 = false;
  bool found_i0 = false;
  bool found_i1 = false;
  for (HloInstruction *instruction : post_order) {
    if (instruction->name() == "i0") {
      EXPECT_TRUE(found_p0);
      EXPECT_TRUE(found_p1);
      found_i0 = true;
    } else if (instruction->name() == "i1") {
      EXPECT_TRUE(found_p0);
      EXPECT_TRUE(found_i0);
      found_i1 = true;
    } else if (instruction->name() == "i2") {
      EXPECT_TRUE(found_p1);
      EXPECT_TRUE(found_i1);
    } else if (instruction->name() == "p0") {
      found_p0 = true;
    } else if (instruction->name() == "p1") {
      found_p1 = true;
    }
  }

  // Verify that MakeInstructionPostOrder() is idempotent.
  auto post_order_2 = module->entry_computation()->MakeInstructionPostOrder();
  EXPECT_EQ(post_order, post_order_2);
}

// Test AddCallee
TEST_F(HLOComputationTest, AddCallee) {
  absl::string_view hlo_string = R"(
HloModule module
diff {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  p2 = f32[100] map(p0, p1), to_apply=diff
  ROOT c = f32[100] map(p0, p2), to_apply=sum
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloComputation *entry = module->entry_computation();
  HloComputation *sum = module->GetComputationWithName("sum");
  ASSERT_NE(entry, nullptr);
  ASSERT_NE(sum, nullptr);

  EXPECT_EQ(entry->callee_computations().size(), 2);
  EXPECT_TRUE(entry->callee_computations().contains(sum));
  EXPECT_EQ(sum->caller_computations().size(), 1);
  EXPECT_EQ(sum->caller_computations().count(entry), 1);

  // Get the operands of the add.
  HloInstruction *entry_a = entry->root_instruction()->mutable_operand(0);
  HloInstruction *entry_b = entry->root_instruction()->mutable_operand(1);

  // Create a new computation and add it as a callee.
  auto builder = HloComputation::Builder("mul");
  auto a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "a"));
  auto b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "b"));

  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kMultiply, a, b));

  HloComputation *mul_comp = module->AddEmbeddedComputation(builder.Build());

  auto map = HloInstruction::CreateMap(entry->root_instruction()->shape(),
                                       {entry_a, entry_b}, mul_comp);

  // Add the new computation as a callee of the entry computation.
  EXPECT_OK(entry->ReplaceWithNewInstruction(entry->root_instruction(),
                                             std::move(map)));

  HloComputation *mul_int = module->GetComputationWithName("mul");

  EXPECT_EQ(entry->callee_computations().size(), 2);
  EXPECT_FALSE(entry->callee_computations().contains(sum));
  EXPECT_EQ(entry->callee_computations().count(mul_int), 1);
  EXPECT_EQ(sum->caller_computations().size(), 0);
  EXPECT_EQ(mul_int->caller_computations().size(), 1);
  EXPECT_TRUE(mul_int->caller_computations().contains(entry));
}
}  // namespace
}  // namespace xla
