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

#include "xla/service/spmd/shardy/stablehlo_round_trip/export_manual_reduction_collectives.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/array.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // for CopyOp
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

namespace sdy = ::mlir::sdy;
namespace stablehlo = ::mlir::stablehlo;

using ::mlir::ArrayRef;
using ::mlir::DenseIntElementsAttr;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::RankedTensorType;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::Value;

using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::ManualComputationOp;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::TensorShardingAttr;

inline constexpr absl::string_view kNonDivisibleShardingError =
    "Non-divisible sharding with unreduced axes isn't supported. Please file a "
    "bug with a reproducer.";

// Returns the channel ID after the maximum channel ID in the given `moduleOp`.
// TODO(b/419222666): remove dependency on `channel_handle` attribute name.
int64_t getNextChannelId(ModuleOp moduleOp) {
  int64_t maxChannelId = 0;
  moduleOp->walk([&](mlir::Operation* op) {
    if (auto channelHandle =
            op->getAttrOfType<stablehlo::ChannelHandleAttr>("channel_handle")) {
      maxChannelId = std::max(maxChannelId, channelHandle.getHandle());
    }
  });
  return maxChannelId + 1;
}

template <class CollectiveTy>
bool inputHasUnreducedAxes(CollectiveTy collective) {
  if (auto sharding = sdy::getSharding(collective.getTensor());
      sharding && !sharding.getUnreducedAxes().empty()) {
    return true;
  }
  return false;
}

// Builds the replica groups for `reductionAxesAttr`.
//
// For example, given:
//
// - reductionAxesAttr = ["y"]
// - mesh = ["x"=2, "y"=2]
//
// Returns `[[0, 1], [2, 3]]`.
mlir::DenseIntElementsAttr getReplicaGroups(
    sdy::AxisRefListAttr reductionAxesAttr, MeshAttr mesh,
    OpBuilder& rewriter) {
  SmallVector<AxisRefAttr> meshAxisRefs =
      getOrderedAxisRefs(reductionAxesAttr, mesh);

  ArrayRef<AxisRefAttr> reductionAxes = reductionAxesAttr.getValue();
  int64_t groupSize = 1;
  llvm::SmallDenseMap<AxisRefAttr, int64_t> axisRefToReductionIndex;
  axisRefToReductionIndex.reserve(reductionAxes.size());
  for (auto [index, axis] : llvm::enumerate(reductionAxes)) {
    groupSize *= axis.getSize(mesh);
    axisRefToReductionIndex[axis] = index;
  }
  int64_t totalSize = mesh.getTotalSize();
  int64_t numGroups = totalSize / groupSize;

  SmallVector<int64_t> transposePerm(meshAxisRefs.size());
  SmallVector<int64_t> reshapeDims;
  reshapeDims.reserve(meshAxisRefs.size());

  int64_t nonReductionIndex = 0;
  int64_t nonReductionCount = meshAxisRefs.size() - reductionAxes.size();
  for (auto [meshIndex, axis] : llvm::enumerate(meshAxisRefs)) {
    reshapeDims.push_back(axis.getSize(mesh));
    auto reductionIndexIt = axisRefToReductionIndex.find(axis);
    if (reductionIndexIt == axisRefToReductionIndex.end()) {
      // Axis is not a reduction axis.
      transposePerm[nonReductionIndex++] = meshIndex;
    } else {
      transposePerm[nonReductionCount + reductionIndexIt->second] = meshIndex;
    }
  }

  // TODO(b/410040098): output V2 if possible, and maybe canonicalize.

  Array<int64_t> array(reshapeDims);
  if (mesh.getDeviceIds().empty()) {
    array.FillIota(0);
  } else {
    array.SetValues(mesh.getDeviceIds());
  }
  array.TransposeDimensions(transposePerm);
  array.Reshape({totalSize});
  auto replicaGroupsType = RankedTensorType::get({numGroups, groupSize},
                                                 rewriter.getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(replicaGroupsType,
                                         llvm::to_vector(array));
}

// Creates a manual computation, with all axes in `mesh` as manual, and
// populates its body via the `bodyPopulator` function.
ManualComputationOp createFullyManualComputation(
    mlir::Location loc, Value input, TensorShardingAttr outSharding,
    MeshAttr mesh, OpBuilder& builder,
    std::function<Value(mlir::BlockArgument arg, OpBuilder& blockBuilder)>
        bodyPopulator) {
  SmallVector<mlir::StringAttr> manualAxes;
  manualAxes.reserve(mesh.getAxes().size());
  for (sdy::MeshAxisAttr axis : mesh.getAxes()) {
    manualAxes.push_back(builder.getStringAttr(axis.getName()));
  }
  TensorShardingAttr inSharding = sdy::getOrCreateSharding(input, mesh);
  auto op = builder.create<ManualComputationOp>(
      loc, input.getType(), input, inSharding, outSharding, manualAxes);

  mlir::Block& block = op.getBody().emplaceBlock();
  auto globalType = mlir::dyn_cast<RankedTensorType>(input.getType());
  CHECK(globalType);
  auto localType = inSharding.getLocalTensorType(globalType, mesh,
                                                 /*allowNonDivisible=*/false);
  CHECK(localType) << kNonDivisibleShardingError;

  OpBuilder blockBuilder = OpBuilder::atBlockBegin(&block);
  blockBuilder.create<sdy::ReturnOp>(
      loc, bodyPopulator(block.addArgument(localType, input.getLoc()),
                         blockBuilder));
  return op;
}

void convertAllReduce(sdy::AllReduceOp op, int64_t channelId,
                      mlir::IRRewriter& rewriter) {
  MeshAttr mesh = op.getOutSharding().getMesh(op);
  rewriter.setInsertionPoint(op);
  ManualComputationOp manualComputation = createFullyManualComputation(
      op.getLoc(), op.getTensor(), op.getOutSharding(), mesh, rewriter,
      [&](mlir::BlockArgument arg, OpBuilder& blockBuilder) {
        // Channel type is DEVICE_TO_DEVICE.
        auto channelHandle = stablehlo::ChannelHandleAttr::get(
            op->getContext(), /*handle=*/channelId, /*type=*/1);
        // Setting `use_global_device_ids=true` as we are targeting the
        // `CollectiveOpGroupMode::kFlattenedID` mode.
        auto newAllReduce = blockBuilder.create<stablehlo::AllReduceOp>(
            op.getLoc(), arg.getType(), arg,
            getReplicaGroups(op.getReductionAxesAttr(), mesh, blockBuilder),
            channelHandle,
            /*use_global_device_ids=*/true);
        // No need to add a sharding to the all-reduce, since it's inside a
        // fully manual computation.
        stablehlo::buildReduceBody<stablehlo::AddOp>(
            mlir::cast<mlir::ShapedType>(arg.getType()).getElementType(),
            newAllReduce.getComputation(), blockBuilder);
        return newAllReduce.getResult(0);
      });
  rewriter.replaceOp(op, manualComputation);
}

int64_t convertReduceScatter(sdy::ReduceScatterOp op, int64_t nextChannelId,
                             mlir::IRRewriter& rewriter) {
  MeshAttr mesh = op.getOutSharding().getMesh(op);
  rewriter.setInsertionPoint(op);
  ManualComputationOp manualComputation = createFullyManualComputation(
      op.getLoc(), op.getTensor(), op.getOutSharding(), mesh, rewriter,
      [&](mlir::BlockArgument arg, OpBuilder& blockBuilder) {
        Value curInput = arg;
        auto inputType = mlir::cast<RankedTensorType>(curInput.getType());
        SmallVector<int64_t> curShape = llvm::to_vector(inputType.getShape());
        for (auto [dim, reductionAxes] :
             llvm::enumerate(op.getReduceScatterAxes())) {
          if (reductionAxes.empty()) {
            continue;
          }
          DenseIntElementsAttr replicaGroups =
              getReplicaGroups(reductionAxes, mesh, blockBuilder);
          int64_t groupSize = replicaGroups.getShapedType().getShape().back();
          CHECK_EQ(curShape[dim] % groupSize, 0) << kNonDivisibleShardingError;
          curShape[dim] /= groupSize;
          // Channel type is DEVICE_TO_DEVICE.
          auto channelHandle = stablehlo::ChannelHandleAttr::get(
              op->getContext(), /*handle=*/nextChannelId++, /*type=*/1);
          // Setting `use_global_device_ids=true` as we are targeting the
          // `CollectiveOpGroupMode::kFlattenedID` mode.
          auto newReduceScatter =
              blockBuilder.create<stablehlo::ReduceScatterOp>(
                  op.getLoc(),
                  RankedTensorType::get(curShape, inputType.getElementType()),
                  curInput, dim, replicaGroups, channelHandle,
                  /*use_global_device_ids=*/true);
          // No need to add a sharding to the reduce-scatter, since it's inside
          // a fully manual computation.
          stablehlo::buildReduceBody<stablehlo::AddOp>(
              mlir::cast<mlir::ShapedType>(arg.getType()).getElementType(),
              newReduceScatter.getComputation(), blockBuilder);
          curInput = newReduceScatter.getResult();
        }
        return curInput;
      });
  rewriter.replaceOp(op, manualComputation);
  return nextChannelId;
}

void convertShardedToUnreduced(sdy::ShardedToUnreducedOp op,
                               mlir::IRRewriter& rewriter) {
  MeshAttr mesh = op.getOutSharding().getMesh(op);
  rewriter.setInsertionPoint(op);

  ManualComputationOp manualComputation = createFullyManualComputation(
      op.getLoc(), op.getTensor(), op.getOutSharding(), mesh, rewriter,
      [&](mlir::BlockArgument arg, OpBuilder& blockBuilder) {
        TensorShardingAttr inSharding = sdy::getSharding(op.getTensor());
        TensorShardingAttr outSharding = op.getOutSharding();
        Value input = arg;
        mlir::Location loc = op.getLoc();

        auto outputType =
            mlir::cast<RankedTensorType>(outSharding.getLocalTensorType(
                mlir::cast<RankedTensorType>(op.getResult().getType()), mesh));
        Value zero = stablehlo::ConstantOp::create(
            blockBuilder, loc,
            blockBuilder.getZeroAttr(outputType.getElementType()));
        Value broadcast = stablehlo::BroadcastOp::create(
            blockBuilder, loc, outputType, zero, outputType.getShape());

        Value partitionId = stablehlo::PartitionIdOp::create(blockBuilder, loc);
        partitionId = stablehlo::ConvertOp::create(
            blockBuilder, loc,
            RankedTensorType::get({}, blockBuilder.getIntegerType(32)),
            partitionId);

        Value logicalId = partitionId;
        // Calculate logical device ID (index in mesh).
        CHECK(mesh.getDeviceIds().empty());

        // Decompose logicalId into axis coordinates.
        llvm::StringMap<Value> axisCoordinates;
        Value currentRem = logicalId;
        // Iterate axes reversed.
        for (const auto& axis : llvm::reverse(mesh.getAxes())) {
          Value axisSize = stablehlo::ConstantOp::create(
              blockBuilder, loc,
              blockBuilder.getIntegerAttr(blockBuilder.getIntegerType(32),
                                          axis.getSize()));

          Value coord =
              stablehlo::RemOp::create(blockBuilder, loc, currentRem, axisSize);
          axisCoordinates[axis.getName()] = coord;
          currentRem =
              stablehlo::DivOp::create(blockBuilder, loc, currentRem, axisSize);
        }

        SmallVector<Value> offsets;
        offsets.reserve(outputType.getRank());

        for (int64_t dim = 0; dim < outputType.getRank(); ++dim) {
          Value offset = stablehlo::ConstantOp::create(
              blockBuilder, loc,
              blockBuilder.getIntegerAttr(blockBuilder.getIntegerType(32), 0));

          Value currentStride = stablehlo::GetDimensionSizeOp::create(
              blockBuilder, loc, input, dim);

          // Input axes.
          SmallVector<StringRef> inAxes;
          if (auto dimSharding = inSharding.getDimSharding(dim)) {
            for (const auto& axis : dimSharding.getAxes()) {
              inAxes.push_back(axis.getName());
            }
          }

          SmallVector<StringRef> outAxes;
          if (auto dimSharding = outSharding.getDimSharding(dim)) {
            for (const auto& axis : dimSharding.getAxes()) {
              outAxes.push_back(axis.getName());
            }
          }

          // Iterate reversed input axes.
          for (auto axisName : llvm::reverse(inAxes)) {
            bool kept = llvm::is_contained(outAxes, axisName);

            if (!kept) {
              Value coord = axisCoordinates[axisName];
              Value term = stablehlo::MulOp::create(blockBuilder, loc, coord,
                                                    currentStride);
              offset =
                  stablehlo::AddOp::create(blockBuilder, loc, offset, term);

              // Update stride.
              int64_t size = 1;
              for (auto axis : mesh.getAxes())
                if (axis.getName() == axisName) size = axis.getSize();
              Value axisSizeVal = stablehlo::ConstantOp::create(
                  blockBuilder, loc,
                  blockBuilder.getIntegerAttr(blockBuilder.getIntegerType(32),
                                              size));
              currentStride = stablehlo::MulOp::create(
                  blockBuilder, loc, currentStride, axisSizeVal);
            }
          }
          offsets.push_back(offset);
        }

        return stablehlo::DynamicUpdateSliceOp::create(
            blockBuilder, loc, outputType, broadcast, input, offsets);
      });
  rewriter.replaceOp(op, manualComputation);
}

void syncInOutUnreducedAxes(mlir::Operation* op) {
  Value input = op->getOperand(0);
  TensorShardingAttr outSharding = sdy::getSharding(op->getResult(0));
  if (mlir::isa<mlir::BlockArgument>(input) || !outSharding ||
      outSharding.getUnreducedAxes().empty()) {
    return;
  }

  TensorShardingAttr inSharding = sdy::getOrCreateSharding(
      input, outSharding.getMeshOrRef(), /*closedIfMissing=*/true);
  if (inSharding.getUnreducedAxes() == outSharding.getUnreducedAxes()) {
    return;
  }

  if (inSharding.anyOfDimShardingOrReplicatedAxis([&](AxisRefAttr inAxis) {
        return llvm::any_of(outSharding.getUnreducedAxes(),
                            [&](AxisRefAttr unreducedAxis) {
                              return inAxis.overlaps(unreducedAxis);
                            });
      })) {
    LOG(WARNING) << toStringView(op->getName().getStringRef())
                 << " has unreduced axes, but input is sharded along one of "
                    "those axes, this can lead to undefined behavior. Please "
                    "contact the Shardy team.";
    return;
  }

  sdy::setSharding(
      input, inSharding.replaceUnreducedAxes(outSharding.getUnreducedAxes()));
}

class StablehloExportManualReductionCollectivesPass
    : public mlir::PassWrapper<StablehloExportManualReductionCollectivesPass,
                               OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      StablehloExportManualReductionCollectivesPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    mlir::IRRewriter rewriter(moduleOp.getContext());

    // Do very restricted backward propagation of unreduced axes along specific
    // ops that don't modify the data.
    moduleOp.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
        [&](mlir::Operation* op) {
          if (mlir::isa<stablehlo::TransposeOp, stablehlo::ReshapeOp,
                        mlir::mhlo::CopyOp>(op)) {
            syncInOutUnreducedAxes(op);
          }
        });

    int64_t nextChannelId = getNextChannelId(moduleOp);
    moduleOp.walk([&](mlir::Operation* op) {
      if (auto allReduce = mlir::dyn_cast<sdy::AllReduceOp>(op)) {
        if (inputHasUnreducedAxes(allReduce)) {
          convertAllReduce(allReduce, nextChannelId++, rewriter);
        }
      } else if (auto reduceScatter =
                     mlir::dyn_cast<sdy::ReduceScatterOp>(op)) {
        if (inputHasUnreducedAxes(reduceScatter)) {
          nextChannelId =
              convertReduceScatter(reduceScatter, nextChannelId, rewriter);
        }
      } else if (auto shardedToUnreduced =
                     mlir::dyn_cast<sdy::ShardedToUnreducedOp>(op)) {
        convertShardedToUnreduced(shardedToUnreduced, rewriter);
      }
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-stablehlo-export-manual-reduction-collectives";
  }

  StringRef getDescription() const override {
    return "Exports `sdy.all_reduce`, that originate from user defined "
           "shardings with unreduced axes, to `stablehlo.all_reduce` inside a "
           "fully manual `sdy.manual_computation`";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<sdy::SdyDialect, stablehlo::StablehloDialect>();
  }
};

}  // namespace

std::unique_ptr<Pass> createStablehloExportManualReductionCollectivesPass() {
  return std::make_unique<StablehloExportManualReductionCollectivesPass>();
}

void registerStablehloExportManualReductionCollectivesPass() {
  mlir::registerPass(createStablehloExportManualReductionCollectivesPass);
}

}  // namespace sdy
}  // namespace xla
