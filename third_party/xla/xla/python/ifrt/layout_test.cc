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

#include "xla/python/ifrt/layout.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::Return;
using ::testing::ReturnRef;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(CompactLayoutTest, Create) {
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->permutation(), ElementsAre());
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));
    EXPECT_THAT(layout->permutation(), ElementsAre(1, 0));
  }
}

TEST(CompactLayoutTest, CreateMajorToMinor) {
  EXPECT_THAT(CompactLayout::CreateMajorToMinor(0)->permutation(),
              ElementsAre());
  EXPECT_THAT(CompactLayout::CreateMajorToMinor(2)->permutation(),
              ElementsAre(0, 1));
}

TEST(CompactLayoutTest, ByteSize) {
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kToken), Shape({})),
                IsOkAndHolds(std::nullopt));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kOpaque), Shape({})),
                IsOkAndHolds(std::nullopt));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kString), Shape({})),
                IsOkAndHolds(std::nullopt));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS8), Shape({})),
                IsOkAndHolds(Optional(1)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS32), Shape({})),
                IsOkAndHolds(Optional(4)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS32), Shape({3, 2})),
                IsOkAndHolds(Optional(24)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS4), Shape({3, 2})),
                IsOkAndHolds(Optional(3)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(
        layout->ByteSize(DType(DType::kS32), Shape({3, 2})),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("CompactLayout expects Shape with the same number of "
                      "dimensions as permutation [], but got shard_shape=")));
  }
}

TEST(LayoutTest, EquivalentLayouts) {
  auto client = std::make_unique<MockClient>();
  ON_CALL(*client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
        return BasicDeviceList::Create(devices);
      });

  auto memory0 = std::make_unique<MockMemory>();
  auto memory1 = std::make_unique<MockMemory>();
  auto memory2 = std::make_unique<MockMemory>();
  MemoryKind memory_kind0("memory kind 0");
  ON_CALL(*memory0, Kind()).WillByDefault(ReturnRef(memory_kind0));
  ON_CALL(*memory1, Kind()).WillByDefault(ReturnRef(memory_kind0));
  ON_CALL(*memory2, Kind()).WillByDefault(ReturnRef(memory_kind0));

  auto device0 = std::make_unique<MockDevice>();
  auto device1 = std::make_unique<MockDevice>();
  auto device2 = std::make_unique<MockDevice>();
  ON_CALL(*device0, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device1, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device2, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device0, Kind()).WillByDefault(Return("device kind 0"));
  ON_CALL(*device0, DefaultMemory()).WillByDefault(Return(memory0.get()));
  ON_CALL(*device1, Kind()).WillByDefault(Return("device kind 0"));
  ON_CALL(*device1, DefaultMemory()).WillByDefault(Return(memory1.get()));
  ON_CALL(*device2, Kind()).WillByDefault(Return("device kind 1"));
  ON_CALL(*device2, DefaultMemory()).WillByDefault(Return(memory2.get()));

  // A concrete layout and a default layout are not equivalent.
  {
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout0, CompactLayout::Create({1, 0}));
    LayoutRef layout1 = nullptr;
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0),
        IsOkAndHolds(false));
  }

  // Two same concrete layouts are equivalent.
  {
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout0, CompactLayout::Create({1, 0}));
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout1, CompactLayout::Create({1, 0}));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(true));
  }
  // Two different concrete layouts are not equivalent.
  {
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout0, CompactLayout::Create({1, 0}));
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout1, CompactLayout::Create({0, 1}));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
  }

  // Default layouts are equivalent if they have the same dtype, shard shape,
  // device kind, and memory kind.
  {
    LayoutRef layout0 = nullptr;
    LayoutRef layout1 = nullptr;
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(true));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device1.get(), MemoryKind()), layout1),
        IsOkAndHolds(true));
  }
  // Default layouts are not equivalent if they have different dtypes, or shard
  // shapes, device kinds, or memory kinds.
  {
    LayoutRef layout0 = nullptr;
    LayoutRef layout1 = nullptr;
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kF32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 100}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), Shape({3, 2}),
            SingleDeviceSharding::Create(device2.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
    EXPECT_THAT(EquivalentLayouts(
                    DType(DType::kS32), Shape({3, 2}),
                    SingleDeviceSharding::Create(device0.get(), MemoryKind()),
                    layout0, DType(DType::kS32), Shape({3, 2}),
                    SingleDeviceSharding::Create(device0.get(),
                                                 MemoryKind("memory kind 1")),
                    layout1),
                IsOkAndHolds(false));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
