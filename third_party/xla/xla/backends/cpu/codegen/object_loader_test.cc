/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/object_loader.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_graph.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "xla/service/cpu/thunk_emitter.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

class ObjectLoaderTest : public HloRunnerAgnosticTestBase {
 public:
  ObjectLoaderTest()
      : HloRunnerAgnosticTestBase(
            std::make_unique<HloRunner>(
                se::PlatformManager::PlatformWithName("host").value()),
            std::make_unique<HloRunner>(
                PlatformUtil::GetDefaultPlatform().value())) {}

 protected:
  absl::StatusOr<std::string> ExportCompilationResultAsString(
      absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(hlo_string));

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        CreateExecutable(std::move(module), /*run_hlo_passes=*/true));

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<AotCompilationResult> exported_aot_result,  // NOLINT
        Export(executable.get()));

    // Serialize-deserialize AOT compilation result.
    TF_ASSIGN_OR_RETURN(std::string serialized_aot_result,
                        exported_aot_result->SerializeAsString());

    return serialized_aot_result;
  }
};

llvm::CodeGenOptLevel CodeGenOptLevel(const HloModuleConfig& module_config) {
  VLOG(2) << "backend_optimization_level: "
          << module_config.debug_options().xla_backend_optimization_level();
  switch (module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      return llvm::CodeGenOptLevel::Less;
    case 2:
      return llvm::CodeGenOptLevel::Default;
    case 3:
      return llvm::CodeGenOptLevel::Aggressive;
    default:
      return llvm::CodeGenOptLevel::None;
  }
}

absl::flat_hash_map<const HloComputation*, bool>
ModuleComputationsTransitivelyContainCustomCall(const HloModule& module) {
  absl::flat_hash_map<const HloComputation*, bool> custom_call_map;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(&module);

  // Can never fail because we always return an OK status from the visitor.
  TF_CHECK_OK(call_graph->VisitNodes([&custom_call_map](
                                         const CallGraphNode& node) {
    const HloComputation* computation = node.computation();

    for (const HloInstruction* instruction : computation->instructions()) {
      // The computation contains a custom-call instruction directly.
      if (DynCast<HloCustomCallInstruction>(instruction)) {
        custom_call_map[computation] = true;
        return absl::OkStatus();
      }
      // The computation calls something that contains a custom-call
      // instruction (directly or indirectly). This lookup relies on the call
      // graph traversing callees before callers, so that the map is always
      // populated for all callees at this point.
      for (const HloComputation* callee : instruction->called_computations()) {
        bool callee_contains_custom_call = FindOrDie(custom_call_map, callee);
        if (callee_contains_custom_call) {
          custom_call_map[computation] = true;
          return absl::OkStatus();
        }
      }
    }

    custom_call_map[computation] = false;
    return absl::OkStatus();
  }));

  return custom_call_map;
}

// NOTE: Used to keep function library alive until the kernel is executed.
struct KernelWrapper {
  std::unique_ptr<Kernel> kernel;
  std::unique_ptr<FunctionLibrary> function_library;
};

absl::StatusOr<KernelWrapper> GetFunctionKernel(
    std::unique_ptr<ObjectLoader> object_loader,
    const CompilationResultProto& proto) {
  // TODO(kbasioli): Once thunks are serialized into the proto most of this
  // code can be removed.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

  const HloModuleConfig& config = module->config();
  const DebugOptions& debug_options = config.debug_options();

  // Options for compiling LLVM IR to machine code.
  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/CodeGenOptLevel(config),
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(config),
      /*disable_expensive_passes=*/
      debug_options.xla_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
  };

  // We don't need any hooks when loading AOT compilation result.
  IrCompiler::CompilationHooks ir_compiler_hooks = {};

  // Definition generator to link with XLA:CPU host runtime symbols.
  JitCompiler::DefinitionGenerator definition_generator =
      [](llvm::TargetMachine* target_machine) {
        return std::make_unique<RuntimeSymbolGenerator>(
            target_machine->createDataLayout());
      };

  // Options for orchestrating the JIT compilation process.
  JitCompiler::Options jit_compiler_options{
      std::move(ir_compiler_options),
      std::move(ir_compiler_hooks),
      /*num_dylibs=*/1,
      /*definition_generator=*/std::move(definition_generator),
      /*max_cpu_isa=*/CpuFeatureFromString(debug_options.xla_cpu_max_isa()),
  };

  llvm::TargetOptions target_options;
  // Always allow FMA fusion. This increases precision instead of decreasing
  // it.
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  TF_ASSIGN_OR_RETURN(
      JitCompiler jit_compiler,
      JitCompiler::Create(target_options, std::move(jit_compiler_options)));

  HloCostAnalysis::ShapeSizeFunction shape_size =
      xla::cpu::CpuExecutable::ShapeSizeBytes;

  std::function<int64_t(const BufferValue&)> buffer_size_bytes_function =
      [shape_size](const BufferValue& buffer) {
        return shape_size(buffer.shape());
      };

  // Recreate BufferAssignment from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto.buffer_assignment(), module.get(),
                                  buffer_size_bytes_function,
                                  /*can_share_buffer=*/nullptr));

  static constexpr absl::string_view kXlaModuleIdentifier = "__compute_module";
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module =
      std::make_unique<llvm::Module>(kXlaModuleIdentifier, *llvm_context);

  TargetMachineFeatures target_machine_features(jit_compiler.target_machine());

  IrEmitter nested_ir_emitter(
      nullptr, *module, *buffer_assignment, llvm_module.get(), {}, {},
      ModuleComputationsTransitivelyContainCustomCall(*module),
      &target_machine_features, /*emit_code_for_msan=*/false);

  IrEmitter2 ir_emitter2(*module, llvm_module.get(), &nested_ir_emitter);

  ThunkEmitter thunk_emitter(ir_emitter2, *buffer_assignment,
                             target_machine_features, module->config());
  TF_ASSIGN_OR_RETURN(ThunkSequence thunks,
                      thunk_emitter.EmitEntryComputation(*module));

  // Collect compiled symbols from IrEmitter2.
  std::vector<FunctionLibrary::Symbol> compiled_symbols;

  for (auto& [name, module] : thunk_emitter.kernels()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Kernel>(name));
  }

  for (const auto& kernel : ir_emitter2.kernels()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Kernel>(kernel.name));
  }
  for (const auto& comparator : ir_emitter2.comparators()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Comparator>(comparator.name));
  }

  EXPECT_EQ(compiled_symbols.size(), 1);
  const std::string entry_point = compiled_symbols.back().name;

  VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                      std::move(*object_loader).Load(compiled_symbols));

  EXPECT_NE(function_library, nullptr);

  TF_ASSIGN_OR_RETURN(
      auto kernel_func,
      function_library->ResolveFunction<FunctionLibrary::Kernel>(entry_point));

  EXPECT_NE(kernel_func, nullptr);

  auto kernel = std::make_unique<Kernel>(/*arity=*/3, kernel_func);

  return KernelWrapper{std::move(kernel), std::move(function_library)};
}

TEST_F(ObjectLoaderTest, Load) {
  const absl::string_view hlo_string = R"(
    HloModule Test

    ENTRY main {
      a = f32[2, 2]{1,0} parameter(0)
      ROOT b = f32[2, 2]{1,0} add(a, a)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto aot_result_serialized,
                          ExportCompilationResultAsString(hlo_string));

  CompilationResultProto proto;
  EXPECT_TRUE(proto.ParseFromString(aot_result_serialized));
  EXPECT_EQ(proto.obj_files_kind(), CompilationResultProto::KERNELS);

  auto object_loader(std::make_unique<ObjectLoader>(/*num_dylibs=*/1));
  {
    size_t obj_file_index = 0;
    for (auto& obj_file : proto.obj_files()) {
      llvm::StringRef data(obj_file.data(), obj_file.size());
      TF_ASSERT_OK(object_loader->AddObjFile(
          obj_file,
          absl::StrCat(proto.entry_function_name(), "_", obj_file_index++)));
    }
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(auto function_kernel_wrapper,
                            GetFunctionKernel(std::move(object_loader), proto));

    std::vector<ExecutionInput> execution_inputs;
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    Literal input_literal(shape);
    input_literal.PopulateR2<float>({{1.0f, 1.0f}, {1.0f, 1.0f}});

    Literal output_literal(shape);

    Kernel::DeviceMemoryBase input_mem(
        input_literal.data<float>().data(),
        input_literal.data<float>().size() * sizeof(float));
    Kernel::DeviceMemoryBase out_mem(
        output_literal.data<float>().data(),
        output_literal.data<float>().size() * sizeof(float));
    std::vector<Kernel::DeviceMemoryBase> args = {input_mem, input_mem,
                                                  out_mem};

    TF_ASSERT_OK(
        function_kernel_wrapper.kernel->Launch(Kernel::ThreadDim(), args));

    for (const auto el : output_literal.data<float>()) {
      EXPECT_EQ(el, 2.0f);
    }
  }
}

}  // namespace
}  // namespace xla::cpu
