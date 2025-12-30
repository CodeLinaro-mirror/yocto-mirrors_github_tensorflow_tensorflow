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

#ifndef XLA_CODEGEN_LOGGING_IR_PRINTER_CONFIG_H_
#define XLA_CODEGEN_LOGGING_IR_PRINTER_CONFIG_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// An IR printer config that logs the IR to a file.
//
// The file is dumped to the directory specified by --xla_dump_to. If the set to
// "sponge", the file is dumped to the test's undeclared outputs directory and
// gets uploaded to Sponge.
//
// The file is dumped in the text format.
//
// One could log anything to this file by calling an_op emit functions like
// emitWarning, emitError, emitRemark etc.
class LoggingIRPrinterConfig : public mlir::PassManager::IRPrinterConfig {
 public:
  LoggingIRPrinterConfig(mlir::MLIRContext* context,
                         const HloModule& hlo_module,
                         absl::string_view kernel_name,
                         absl::string_view pass_manager_name);

  static void EnableLoggingIfRequested(mlir::PassManager& pass_manager,
                                       mlir::MLIRContext* context,
                                       const HloModule& hlo_module,
                                       absl::string_view kernel_name,
                                       absl::string_view pass_manager_name) {
    if (ShouldDumpMlirPasses(hlo_module, pass_manager_name)) {
      auto config = std::make_unique<LoggingIRPrinterConfig>(
          context, hlo_module, kernel_name, pass_manager_name);
      pass_manager.enableIRPrinting(std::move(config));
    }
  }

  ~LoggingIRPrinterConfig() override;

  void printBeforeIfEnabled(mlir::Pass* pass, mlir::Operation* operation,
                            PrintCallbackFn printCallback) override;

  void printAfterIfEnabled(mlir::Pass* pass, mlir::Operation* operation,
                           PrintCallbackFn printCallback) override;

 private:
  static bool ShouldDumpMlirPasses(const HloModule& hlo_module,
                                   absl::string_view pass_manager_name);

  std::unique_ptr<llvm::raw_fd_ostream> log_stream_;
  mlir::DiagnosticEngine::HandlerID callback_id_ = 0;
  mlir::MLIRContext* context_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_LOGGING_IR_PRINTER_CONFIG_H_
