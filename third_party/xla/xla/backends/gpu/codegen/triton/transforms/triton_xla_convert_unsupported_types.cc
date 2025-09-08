
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
#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACONVERTUNSUPPORTEDTYPESPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {
class UnsupportedTypesConverter : public TypeConverter {
 public:
  UnsupportedTypesConverter() : TypeConverter() {
    // Fallback to the no conversion for all other types.
    addConversion([](Type type) -> std::optional<Type> { return type; });

    // Convert F8E8M0FNUType to i8. This is a workaround for the fact that
    // TritonXLA doesn't support F8E8M0FNUType natively.
    addConversion([](Float8E8M0FNUType type) -> std::optional<Type> {
      return IntegerType::get(type.getContext(), 8);
    });

    // Helper conversions for the nontrivial types.
    addConversion([this](Type type) -> std::optional<Type> {
      if (auto shaped_type = dyn_cast<ShapedType>(type)) {
        Type newType = convertType(shaped_type.getElementType());
        return shaped_type.clone(newType);
      }
      return std::nullopt;
    });
    addConversion([this](Type type) -> std::optional<Type> {
      if (auto pointer_type = dyn_cast<triton::PointerType>(type)) {
        Type newType = convertType(pointer_type.getPointeeType());
        return triton::PointerType::get(newType,
                                        pointer_type.getAddressSpace());
      }
      return std::nullopt;
    });
    addConversion([this](Type type) -> std::optional<Type> {
      if (auto func_type = dyn_cast<FunctionType>(type)) {
        SmallVector<Type> newInputs =
            convertTypes(*this, func_type.getInputs());
        SmallVector<Type> newResults =
            convertTypes(*this, func_type.getResults());
        if (newInputs != func_type.getInputs() ||
            newResults != func_type.getResults()) {
          return FunctionType::get(func_type.getContext(), newInputs,
                                   newResults);
        }
      }
      return std::nullopt;
    });
  }

  // Helper method to convert a range of types.
  static SmallVector<Type> convertTypes(
      const UnsupportedTypesConverter& converter, ArrayRef<Type> types) {
    SmallVector<Type> newTypes;
    for (auto type : types) {
      auto newType = converter.convertType(type);
      if (type != newType) {
        newTypes.push_back(newType);
      } else {
        newTypes.push_back(type);
      }
    }
    return newTypes;
  }
};

struct RewriteF8ToI8ConversionPattern final : ConversionPattern {
  RewriteF8ToI8ConversionPattern(const TypeConverter& converter,
                                 MLIRContext* ctx)
      : ConversionPattern::ConversionPattern(
            converter, Pattern::MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (getTypeConverter()->isLegal(op)) {
      return failure();
    }

    // The rewrite doesn't handle cloning regions.
    if (op->getNumRegions() != 0) {
      return failure();
    }

    Location loc = op->getLoc();
    const TypeConverter* converter = getTypeConverter();
    SmallVector<Type> resultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), resultTypes))) {
      // Note to anyone looking for this error message: this is a "can't
      // happen". If you're seeing it, there's a bug.
      return op->emitOpError("type conversion failed in float emulation");
    }
    Operation* replacement = rewriter.create(
        loc, op->getName().getIdentifier(), operands, resultTypes,
        op->getAttrs(), op->getSuccessors(), /*regions=*/{});
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class TritonXLAConvertUnsupportedTypesPass
    : public impl::TritonXLAConvertUnsupportedTypesPassBase<
          TritonXLAConvertUnsupportedTypesPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    auto* ctx = &getContext();
    auto module = getOperation();
    UnsupportedTypesConverter converter;
    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      if (auto func_op = dyn_cast<func::FuncOp>(op)) {
        return converter.isLegal(func_op.getFunctionType());
      }
      return converter.isLegal(op);
    });
    RewritePatternSet patterns(ctx);

    patterns.add<RewriteF8ToI8ConversionPattern>(converter,
                                                 patterns.getContext());
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAConvertUnsupportedTypesPass() {
  return std::make_unique<TritonXLAConvertUnsupportedTypesPass>();
}

}  // namespace mlir::triton::xla
