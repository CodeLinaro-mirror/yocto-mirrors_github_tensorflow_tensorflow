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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACONTRACTDIMSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Returns list of dimensions with size equal to 1. If they all dimensions have
// size 1, do not return the last dimension to avoid tripping up Triton.
SmallVector<uint32_t> FindUnitDimensions(RankedTensorType type) {
  SmallVector<uint32_t> result;
  for (auto [dim, size] : llvm::enumerate(type.getShape())) {
    if (size == 1) {
      result.push_back(dim);
    }
  }
  if (result.size() == type.getRank()) {
    result.pop_back();  // Keep one unit dimension.
  }
  return result;
}

// If all users of an op are contract_dims with the same axis, return it.
std::optional<uint32_t> GetSingleContractUserAxis(Operation* op) {
  std::optional<uint32_t> axis;
  for (Operation* user : op->getUsers()) {
    auto op = dyn_cast<ContractDimsOp>(user);
    if (!op || op.getAxis() != axis.value_or(op.getAxis())) {
      return std::nullopt;
    }
    axis = op.getAxis();
  }
  return axis;
}

// Sets the insertion point at the given op and returns the guard.
[[nodiscard]] OpBuilder::InsertionGuard SetInsertionPoint(OpBuilder& builder,
                                                          Operation* op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  return guard;
}

// Returns a new container with the given dimensions removed.
template <typename ContainerT>
auto ContractElements(ContainerT elements, ArrayRef<uint32_t> contract_dims) {
  auto it = elements.begin();
  SmallVector<typename std::iterator_traits<decltype(it)>::value_type> result;
  for (uint32_t dim : contract_dims) {
    auto end = elements.begin() + dim;
    std::copy(it, end, std::back_inserter(result));
    it = std::next(end);
  }
  std::copy(it, elements.end(), std::back_inserter(result));
  return result;
}

// Returns a new boundary check with the given dimensions removed.
SmallVector<int32_t> ContractBoundaryCheck(ArrayRef<int32_t> boundary_check,
                                           ArrayRef<uint32_t> contract_dims) {
  CHECK(absl::c_is_sorted(boundary_check));
  CHECK(absl::c_is_sorted(contract_dims));
  SmallVector<int32_t> result;
  auto it = contract_dims.begin();
  for (int32_t dim : boundary_check) {
    it = std::lower_bound(it, contract_dims.end(), dim);
    if (it != contract_dims.end() && *it == dim) {
      continue;
    }
    result.push_back(dim - (it - contract_dims.begin()));
  }
  return result;
}

// Returns a new tensor type with the given dimensions removed.
RankedTensorType ContractTensorType(RankedTensorType type,
                                    ArrayRef<uint32_t> contract_dims) {
  auto shape = ContractElements(type.getShape(), contract_dims);
  Attribute encoding = type.getEncoding();
  if (encoding) {
    auto inferLayoutInterface =
        cast<DialectInferLayoutInterface>(&encoding.getDialect());
    [[maybe_unused]] auto result = inferLayoutInterface->inferReshapeOpEncoding(
        type.getShape(), encoding, shape, encoding, std::nullopt);
    CHECK(succeeded(result));
  }
  return RankedTensorType::get(shape, type.getElementType(), encoding);
}

// Returns a new tensor value with the given dimensions removed.
// Low dimensions are contracted first.
Value ContractTensorValue(PatternRewriter& rewriter, Value value,
                          ArrayRef<uint32_t> contract_dims) {
  CHECK(absl::c_is_sorted(contract_dims));
  for (uint32_t i = 0; i < contract_dims.size(); ++i) {
    uint32_t dim = contract_dims[i] - i;
    Type type =
        ContractTensorType(cast<RankedTensorType>(value.getType()), dim);
    value = rewriter.create<ContractDimsOp>(value.getLoc(), type, value, dim);
  }
  return value;
}

// Rewrites tt.make_tensor_ptr with unit dimensions. Returns the
// new MakeTensorPtrOp result and the dimensions that were removed.
Value ContractMakeTensorPtr(PatternRewriter& rewriter, MakeTensorPtrOp op,
                            ArrayRef<uint32_t> contract_dims) {
  auto tensor_type = cast<RankedTensorType>(op.getType().getPointeeType());
  auto contracted_type = ContractTensorType(tensor_type, contract_dims);
  auto ptr_type =
      PointerType::get(contracted_type, op.getType().getAddressSpace());

  // Strides already encode the layout, so we can use the default order.
  // Note that the order attribute is ignored in the Triton lowering.
  SmallVector<int32_t> order(contracted_type.getShape().size());
  std::iota(order.rbegin(), order.rend(), 0);

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, op);
  Value result = rewriter.create<MakeTensorPtrOp>(
      op.getLoc(), ptr_type, op.getBase(),
      ContractElements(op.getShape(), contract_dims),
      ContractElements(op.getStrides(), contract_dims),
      ContractElements(op.getOffsets(), contract_dims), order);
  return result;
}

// Folds contract_dims into tt.load(tt.make_tensor_ptr).
// TODO(csigg): Add support for tt.load(tt.make_tensor_descriptor).
LogicalResult FoldContractDimsOfLoad(LoadOp op, PatternRewriter& rewriter) {
  std::optional<uint32_t> axis = GetSingleContractUserAxis(op);
  if (!axis) {
    return rewriter.notifyMatchFailure(
        op, "Not all users are equivalent contract_dims.");
  }
  auto make_tensor_ptr = op.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_tensor_ptr) {
    return rewriter.notifyMatchFailure(
        op, "Expected ptr to be defined by make_tensor_ptr.");
  }

  Value pointer = ContractMakeTensorPtr(rewriter, make_tensor_ptr, *axis);

  Value new_load = rewriter.create<LoadOp>(
      op.getLoc(), pointer, ContractBoundaryCheck(op.getBoundaryCheck(), *axis),
      op.getPadding(), op.getCache(), op.getEvict(), op.getIsVolatile());

  for (Operation* user : make_early_inc_range(op->getUsers())) {
    rewriter.replaceOp(user, new_load);
  }
  return success();
}

// Extracts unit dimensions from tt.store and prepends them as contract_dims.
LogicalResult ContractStore(StoreOp op, PatternRewriter& rewriter) {
  auto make_tensor_ptr = op.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_tensor_ptr) {
    return rewriter.notifyMatchFailure(
        op, "Expected ptr to be defined by make_tensor_ptr.");
  }
  auto tensor_type = dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!tensor_type || tensor_type.getRank() == 0) {
    return rewriter.notifyMatchFailure(op, "Expected tensor type.");
  }

  auto contract_dims = FindUnitDimensions(tensor_type);
  if (contract_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  Value pointer =
      ContractMakeTensorPtr(rewriter, make_tensor_ptr, contract_dims);
  Value value = ContractTensorValue(rewriter, op.getValue(), contract_dims);
  rewriter.replaceOpWithNewOp<StoreOp>(
      op, pointer, value,
      ContractBoundaryCheck(op.getBoundaryCheck(), contract_dims),
      op.getCache(), op.getEvict());
  return success();
}

// Extracts unit dimensions from tt.reshape and prepends them as contract_dims.
LogicalResult ContractReshape(ReshapeOp op, PatternRewriter& rewriter) {
  if (op.getAllowReorderAttr() || op.getEfficientLayoutAttr()) {
    return rewriter.notifyMatchFailure(op, "Unsupported reshape.");
  }
  auto contract_dims = FindUnitDimensions(op.getSrc().getType());
  if (contract_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  Value value = ContractTensorValue(rewriter, op.getSrc(), contract_dims);
  rewriter.modifyOpInPlace(op, [&]() { op.setOperand(value); });
  return success();
}

// Pushes tt.contract_dims up through element-wise operations.
// Example:
//   %1 = elementwise-op %0
//   %2 = tt.contract_dims %1
// is rewritten to:
//   %1 = tt.contract_dims %0
//   %2 = elementwise-op %1
class PushContractDimsUpThroughElementwise final
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
 public:
  using OpTraitRewritePattern::OpTraitRewritePattern;

 private:
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "Expected single result.");
    }

    uint32_t axis;
    if (isPure(op)) {
      auto range = map_range(op->getUsers(), [](const Operation* user) {
        return dyn_cast<ContractDimsOp>(user);
      });
      auto it = std::find_if(range.begin(), range.end(),
                             [](Operation* user) { return user; });
      if (it == range.end()) {
        return rewriter.notifyMatchFailure(op, "Expected contract_dims user.");
      }
      axis = (*it).getAxis();
    } else {
      auto axis_or = GetSingleContractUserAxis(op);
      if (!axis_or) {
        return rewriter.notifyMatchFailure(
            op, "Not all users are equivalent contract_dims.");
      }
      axis = *axis_or;
    }

    SmallVector<Value> operands;
    operands.reserve(op->getOperands().size());
    for (Value operand : op->getOperands()) {
      if (isa<RankedTensorType>(operand.getType())) {
        operand = ContractTensorValue(rewriter, operand, axis);
      }
      operands.push_back(operand);
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    state.addTypes(ContractTensorType(type, axis));
    Operation* new_op = rewriter.create(state);
    for (Operation* user : make_early_inc_range(op->getUsers())) {
      if (auto op = dyn_cast<ContractDimsOp>(user);
          op && op.getAxis() == axis) {
        rewriter.replaceOp(user, new_op->getResults());
      }
    }

    return success();
  }
};

// Pushes tt.contract_dims up through tt.broadcast.
// Example:
//   %1 = tt.broadcast %0
//   %2 = tt.contract_dims %1
// is rewritten to:
//   %1 = tt.contract_dims %0
//   %2 = tt.broadcast %1
LogicalResult PushContractDimsUpThroughBroadcast(ContractDimsOp op,
                                                 PatternRewriter& rewriter) {
  auto broadcast = op.getSrc().getDefiningOp<BroadcastOp>();
  if (!broadcast) {
    return rewriter.notifyMatchFailure(op, "Expected broadcast producer.");
  }

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, broadcast);
  Value value = ContractTensorValue(rewriter, broadcast.getSrc(), op.getAxis());
  rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), value);
  return success();
}

// Pushes tt.contract_dims up through tt.trans.
// Example:
//   %1 = tt.trans %0, perm = [1, 0]
//   %2 = tt.contract_dims %1, axis = 0
// is rewritten to:
//   %1 = tt.contract_dims %0, axis = 1
//   %2 = tt.trans %1, perm = [0]
LogicalResult PushContractDimsUpThroughTrans(ContractDimsOp op,
                                             PatternRewriter& rewriter) {
  auto trans = op.getSrc().getDefiningOp<TransOp>();
  if (!trans) {
    return rewriter.notifyMatchFailure(op, "Expected trans producer.");
  }

  auto order = trans.getOrder();
  uint32_t dst_axis = op.getAxis();
  uint32_t src_axis = order[dst_axis];

  // Compute the new permutation for the transpose.
  auto new_order = ContractElements(order, dst_axis);
  for (auto& dim : new_order) {
    dim -= dim > src_axis;
  }

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, trans);
  Value value = ContractTensorValue(rewriter, trans.getSrc(), src_axis);
  rewriter.replaceOpWithNewOp<TransOp>(op, value, new_order);
  return success();
}

// Pushes tt.contract_dims up through tt.join.
// Example:
//   %2 = tt.join %0, %1
//   %3 = tt.contract_dims %2
// is rewritten to:
//   %2 = tt.contract_dims %0
//   %3 = tt.contract_dims %1
//   %4 = tt.join %2, %3
LogicalResult PushContractDimsUpThroughJoin(ContractDimsOp op,
                                            PatternRewriter& rewriter) {
  auto join = op.getSrc().getDefiningOp<JoinOp>();
  if (!join) {
    return rewriter.notifyMatchFailure(op, "Expected join producer.");
  }

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, join);
  SmallVector<Value> operands;
  operands.reserve(join.getOperands().size());
  for (Value operand : join.getOperands()) {
    operands.push_back(ContractTensorValue(rewriter, operand, op.getAxis()));
  }

  rewriter.replaceOpWithNewOp<JoinOp>(op, op.getType(), operands);
  return success();
}

// Pushes tt.contract_dims up through tt.reduce.
// Example:
//   %1 = tt.reduce(%0) axis=1
//   %2 = tt.contract_dims %1, axis=0
// is rewritten to:
//   %1 = tt.contract_dims %0, axis=0
//   %2 = tt.reduce(%1) axis=0
LogicalResult PushContractDimsUpThroughReduce(ContractDimsOp op,
                                              PatternRewriter& rewriter) {
  auto reduce = op.getSrc().getDefiningOp<ReduceOp>();
  if (!reduce || reduce.getNumResults() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Expected single-result reduce producer.");
  }

  uint32_t contract_axis = op.getAxis() + (op.getAxis() >= reduce.getAxis());
  uint32_t reduce_axis = reduce.getAxis() - (op.getAxis() < reduce.getAxis());

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, reduce);
  SmallVector<Value> operands;
  operands.reserve(reduce.getOperands().size());
  for (Value operand : reduce.getOperands()) {
    operands.push_back(ContractTensorValue(rewriter, operand, contract_axis));
  }

  auto new_reduce = rewriter.replaceOpWithNewOp<ReduceOp>(
      op, op.getType(), operands, reduce_axis);
  rewriter.cloneRegionBefore(reduce->getRegion(0), new_reduce->getRegion(0),
                             new_reduce->getRegion(0).begin());
  return success();
}

// Pushes tt.contract_dims up through tt.expand_dims, or folds them.
// Example:
//   %1 = tt.expand_dims %0, axis=1
//   %2 = tt.contract_dims %1, axis=0
// is rewritten to:
//   %1 = tt.contract_dims %0, axis=0
//   %2 = tt.expand_dims %1, axis=0
LogicalResult PushContractDimsUpThroughExpandDims(ContractDimsOp op,
                                                  PatternRewriter& rewriter) {
  auto expand_dims = op.getSrc().getDefiningOp<ExpandDimsOp>();
  if (!expand_dims) {
    return rewriter.notifyMatchFailure(op, "Expected expand_dims producer.");
  }

  if (expand_dims.getSrc().getType() == op.getType()) {
    rewriter.replaceOp(op, expand_dims.getSrc());
    return success();
  }

  // Swap: contract_dims(expand_dims) -> expand_dims(contract_dims)
  uint32_t expand_axis = expand_dims.getAxis();
  uint32_t contract_axis = op.getAxis();
  CHECK_NE(expand_axis, contract_axis);

  contract_axis -= contract_axis > expand_axis;
  expand_axis -= expand_axis > contract_axis;

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, expand_dims);
  Value value =
      ContractTensorValue(rewriter, expand_dims.getSrc(), contract_axis);
  rewriter.replaceOpWithNewOp<ExpandDimsOp>(op, op.getType(), value,
                                            expand_axis);
  return success();
}

// Converts tt.contract_dims to tt.reshape.
LogicalResult ContractDimsToReshape(ContractDimsOp op,
                                    PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getSrc());
  return success();
}

// This pass removes unit dimensions from tensors to generate faster code
// through Triton.
//
// - contract_dims ops are extracted from tt.store and tt.reshape with unit
//   dimensions.
// - A series of patterns push contract_dims up through the graph, past
//   pure ops like broadcast, transpose, join, reduce.
// - Patterns for element-wise and load, which are non-pure ops that should
// not
//   be duplicated. This requires that all users are equivalent contract_dims.
// - contract_dims are folded into load and expand_dims, or converted back to
//   reshape.
class TritonXLAContractDimsPass
    : public impl::TritonXLAContractDimsPassBase<TritonXLAContractDimsPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(FoldContractDimsOfLoad);
    if (contract_store_) {
      patterns.add(ContractStore);
    }
    patterns.add(ContractReshape);
    patterns.add<PushContractDimsUpThroughElementwise>(&getContext());
    patterns.add(PushContractDimsUpThroughBroadcast);
    patterns.add(PushContractDimsUpThroughTrans);
    patterns.add(PushContractDimsUpThroughJoin);
    patterns.add(PushContractDimsUpThroughReduce);
    patterns.add(PushContractDimsUpThroughExpandDims);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    if (finalize_) {
      RewritePatternSet patterns(&getContext());
      patterns.add(ContractDimsToReshape);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAContractDimsPass() {
  return std::make_unique<TritonXLAContractDimsPass>();
}

}  // namespace mlir::triton::xla
