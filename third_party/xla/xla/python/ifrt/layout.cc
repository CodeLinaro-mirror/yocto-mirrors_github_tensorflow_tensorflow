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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/lib/core/bitmap.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char Layout::ID = 0;
char CompactLayout::ID = 0;

absl::StatusOr<absl_nonnull std::unique_ptr<Layout>> Layout::FromProto(
    const LayoutProto& layout_proto) {
  return Deserialize<Layout>(layout_proto.serialized_layout(),
                             /*options=*/nullptr);
}

absl::StatusOr<LayoutProto> Layout::ToProto() const {
  LayoutProto layout_proto;
  TF_ASSIGN_OR_RETURN(*layout_proto.mutable_serialized_layout(),
                      Serialize(*this,
                                /*options=*/nullptr));
  return layout_proto;
}

absl::StatusOr<absl_nonnull std::unique_ptr<CompactLayout>>
CompactLayout::Create(absl::Span<const int> permutation) {
  tsl::core::Bitmap bitmap(permutation.size());
  for (int i : permutation) {
    if (i < 0 || i >= permutation.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("CompactLayout expects permutation with elements in "
                       "range [0, ",
                       permutation.size(), "), but got permutation=[",
                       absl::StrJoin(permutation, ","), "]"));
    }
    bitmap.set(i);
  }
  if (!bitmap.IsAllSet()) {
    return absl::InvalidArgumentError(
        absl::StrCat("CompactLayout expects permutation with all elements "
                     "in range [0, ",
                     permutation.size(), ")), but got permutation=[",
                     absl::StrJoin(permutation, ","), "]"));
  }
  return absl::WrapUnique<CompactLayout>(
      new CompactLayout(Permutation(permutation.begin(), permutation.end())));
}

absl_nonnull std::unique_ptr<CompactLayout> CompactLayout::CreateMajorToMinor(
    int num_shard_shape_dims) {
  Permutation permutation(num_shard_shape_dims);
  absl::c_iota(permutation, 0);
  return absl::WrapUnique<CompactLayout>(
      new CompactLayout(std::move(permutation)));
}

absl::StatusOr<std::optional<int64_t>> CompactLayout::ByteSize(
    DType dtype, const Shape& shard_shape) const {
  if (permutation_.size() != shard_shape.dims().size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("CompactLayout expects Shape with the same number of "
                     "dimensions as permutation [",
                     absl::StrJoin(permutation_, ","),
                     "], but got shard_shape=", shard_shape));
  }
  auto bit_size = dtype.bit_size();
  if (!bit_size.has_value()) {
    return std::nullopt;
  }
  return (shard_shape.num_elements() * *bit_size + 7) / 8;
}

bool CompactLayout::operator==(const Layout& other) const {
  if (this == &other) {
    return true;
  }
  if (const auto* other_compact = llvm::dyn_cast<CompactLayout>(&other);
      other_compact != nullptr) {
    return permutation_ == other_compact->permutation_;
  }
  return false;
}

std::string CompactLayout::ToString() const {
  return absl::StrCat("CompactLayout([", absl::StrJoin(permutation_, ","),
                      "])");
}

absl::StatusOr<bool> EquivalentLayouts(
    DType dtype1, const Shape& shape1,
    const std::shared_ptr<const Sharding>& sharding1, const LayoutRef& layout1,
    DType dtype2, const Shape& shape2,
    const std::shared_ptr<const Sharding>& sharding2,
    const LayoutRef& layout2) {
  if (layout1 == nullptr && layout2 == nullptr) {
    if (dtype1 != dtype2 || shape1 != shape2) {
      return false;
    }
    if (sharding1->memory_kind() != sharding2->memory_kind()) {
      return false;
    }
    if (sharding1->devices()->devices().empty() ||
        sharding2->devices()->devices().empty()) {
      // For now, we handle empty device lists to have a default layout that is
      // not equal to any other layout.
      return false;
    }
    // For now, we assume that devices with the same device kind have the same
    // default layout rules. This is not necessarily accurate check today
    // because device kinds do not necessarily determine the default layout
    // rules.
    //
    // TODO(hyeontaek): Introduce a concept of "default layout domain" to
    // devices to make this check more accurate and efficient.
    Device* device1 = sharding1->devices()->devices().front();
    Device* device2 = sharding2->devices()->devices().front();
    return device1 == device2 || device1->Kind() == device2->Kind();
  }
  if (layout1 != nullptr && layout2 != nullptr) {
    return *layout1 == *layout2;
  }
  return false;
}

}  // namespace ifrt
}  // namespace xla
