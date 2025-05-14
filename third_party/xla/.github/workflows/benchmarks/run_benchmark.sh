# Copyright 2025 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# .github/workflows/benchmarks/run_benchmark.sh
# TODO(juliagmt): convert this to a python script.
#!/bin/bash

# IMPORTANT: pipefail is handled specifically around the runner command and stats command.
set -eu # Exit on errors, and treat unset variables as an error.

# --- Global Variables (Derived from GitHub Actions environment) ---
readonly BENCHMARK_NAME="${BENCHMARK_NAME}"
readonly CONFIG_ID="${CONFIG_ID}"
readonly HARDWARE_CATEGORY="${HARDWARE_CATEGORY}"
readonly OUTPUT_DIR="${OUTPUT_DIR}"
readonly RUNNER_BINARY="${RUNNER_BINARY}"
readonly STATS_BINARY="${STATS_BINARY}"
readonly DEVICE_TYPE_FLAG="${DEVICE_TYPE_FLAG}"
readonly LOCAL_ARTIFACT_PATH="${LOCAL_ARTIFACT_PATH}"
readonly INPUT_FORMAT="${INPUT_FORMAT}"
readonly XLA_FLAGS_JSON="${XLA_FLAGS_JSON}"
readonly RUNTIME_FLAGS_JSON="${RUNTIME_FLAGS_JSON}"
readonly COMMIT_SHA="${COMMIT_SHA}"
readonly WORKFLOW_RUN_ID="${WORKFLOW_RUN_ID}"

# Derived paths
readonly RUNNER_STDOUT_FILE="${OUTPUT_DIR}/runner_stdout.txt"
readonly XSPACE_FILE_PATH="${OUTPUT_DIR}/xspace.pb"
readonly RESULTS_JSON_FILE="${OUTPUT_DIR}/results.json"

# --- Functions ---

# Prints job info and validates essential environment variables and file paths.
validate_inputs() {
    echo "--- Running Benchmark Script ---"
    echo "Benchmark Name: $BENCHMARK_NAME"
    echo "Config ID: $CONFIG_ID"
    echo "Hardware Category: $HARDWARE_CATEGORY"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Runner Binary: $RUNNER_BINARY"
    echo "Stats Binary: $STATS_BINARY"
    echo "Device Type Flag: $DEVICE_TYPE_FLAG"
    echo "Local Artifact Path: $LOCAL_ARTIFACT_PATH"
    echo "Input Format: $INPUT_FORMAT"
    echo "XLA Flags JSON: $XLA_FLAGS_JSON"
    echo "Runtime Flags JSON: $RUNTIME_FLAGS_JSON"
    echo "Commit SHA: $COMMIT_SHA"
    echo "Workflow Run ID: $WORKFLOW_RUN_ID"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR" || { echo "::error::Failed to create output directory: $OUTPUT_DIR"; exit 1; }

    if [ -z "$LOCAL_ARTIFACT_PATH" ] || [ ! -f "$LOCAL_ARTIFACT_PATH" ]; then echo "::error::LOCAL_ARTIFACT_PATH path is invalid or file not found: '$LOCAL_ARTIFACT_PATH'"; exit 1; fi
    if [ -z "$RUNNER_BINARY" ] || [ ! -x "$RUNNER_BINARY" ]; then echo "::error::RUNNER_BINARY path is invalid or file not executable: '$RUNNER_BINARY'"; exit 1; fi
    if [ -z "$DEVICE_TYPE_FLAG" ]; then echo "::error::DEVICE_TYPE_FLAG is empty"; exit 1; fi
    if [ -z "$STATS_BINARY" ] || [ ! -x "$STATS_BINARY" ]; then echo "::error::STATS_BINARY path is invalid or file not executable: '$STATS_BINARY'"; exit 1; fi
    if ! command -v jq &> /dev/null; then echo "::error::jq command not found. Please ensure 'jq' is installed in the container."; exit 1; fi
}

# Parses JSON flag strings into bash arrays and adds required profile flags.
# Populates global arrays: xla_flags_array, runtime_flags_array
prepare_flags() {
    # Local arrays for this function, will be copied to global scope
    local -a local_xla_flags_array=()
    local -a local_runtime_flags_array=()

    # Use JQ to safely parse JSON and populate bash arrays
    if echo "$XLA_FLAGS_JSON" | jq -e '. | arrays and length > 0' > /dev/null; then
        mapfile -t local_xla_flags_array < <(echo "$XLA_FLAGS_JSON" | jq -r '.[]')
    fi
    if echo "$RUNTIME_FLAGS_JSON" | jq -e '. | arrays and length > 0' > /dev/null; then
       mapfile -t local_runtime_flags_array < <(echo "$RUNTIME_FLAGS_JSON" | jq -r '.[]')
    fi

    # Conditionally add profile flag if needed for stats. Assume we always want stats if possible.
    local needs_profile_flag=true
    for flag in "${local_runtime_flags_array[@]}"; do
        if [[ "$flag" == "--profile_execution"* ]]; then
            needs_profile_flag=false; break
        fi
    done

    if "$needs_profile_flag"; then # No need for needs_xspace_dump_flag, it's always true here
        local_runtime_flags_array+=("--profile_execution=True")
        echo "INFO: Added --profile_execution=True for stats generation."
    fi

    # Export arrays to make them available globally
    declare -g xla_flags_array=("${local_xla_flags_array[@]}")
    declare -g runtime_flags_array=("${local_runtime_flags_array[@]}")
}

# Executes the HLO runner binary with constructed flags and captures output.
# Returns: The exit code of the runner.
execute_runner() {
    local -a runner_command_array=(
        "$RUNNER_BINARY"
        "--device_type=$DEVICE_TYPE_FLAG"
    )

    if [ ${#runtime_flags_array[@]} -gt 0 ]; then runner_command_array+=("${runtime_flags_array[@]}"); fi
    if [ ${#xla_flags_array[@]} -gt 0 ]; then runner_command_array+=("${xla_flags_array[@]}"); fi
    runner_command_array+=("--xla_gpu_dump_xspace_to=$XSPACE_FILE_PATH") # Always try to dump xspace
    runner_command_array+=("$LOCAL_ARTIFACT_PATH")

    echo "Executing HLO Runner command:"
    printf "%q " "${runner_command_array[@]}"; echo

    set +e # Disable exit-on-error temporarily to capture exit code
    set -o pipefail # Ensure tee doesn't mask the runner's exit code
    "${runner_command_array[@]}" 2>&1 | tee "$RUNNER_STDOUT_FILE"
    local runner_exit_code=${PIPESTATUS[0]}
    set +o pipefail
    set -e # Re-enable exit-on-error

    echo "Runner stdout/stderr saved to $RUNNER_STDOUT_FILE"
    echo "Runner exited with code: $runner_exit_code"
    return "$runner_exit_code"
}

# Processes the XSpace file using the stats binary and extracts metrics.
# Populates global variable: METRICS_JSON_CONTENT (JSON string)
# Sets global boolean: STATS_RUN_SUCCESSFUL
process_stats() {
    local stats_exit_code=1 # Default to failure
    declare -g STATS_RUN_SUCCESSFUL=false
    declare -g METRICS_JSON_CONTENT="{}" # Default to empty JSON

    if [ ! -f "$XSPACE_FILE_PATH" ]; then
        echo "::warning::XSpace file missing at $XSPACE_FILE_PATH, skipping stats processing."
        return
    fi

    echo "XSpace file found. Running compute_xspace_stats_main..."
    local stats_platform_type=$([[ "$HARDWARE_CATEGORY" == GPU* ]] && echo "GPU" || echo "CPU")

    echo "Executing Stats command and capturing its output:"
    set +e # Temporarily disable exit-on-error for stats command
    local stats_output
    stats_output=$("$STATS_BINARY" --input="$XSPACE_FILE_PATH" --device_type="$stats_platform_type" 2>&1)
    stats_exit_code=$?
    set -e

    echo "compute_xspace_stats_main output:"
    echo "$stats_output"
    echo "compute_xspace_stats_main exited with code: $stats_exit_code"

    # Append stats tool's raw output to the main runner log for complete record
    echo -e "\n--- compute_xspace_stats_main Raw Output ---" >> "$RUNNER_STDOUT_FILE"
    echo "$stats_output" >> "$RUNNER_STDOUT_FILE"
    echo "--- End compute_xspace_stats_main Raw Output ---" >> "$RUNNER_STDOUT_FILE"

    if [ "$stats_exit_code" -eq 0 ]; then
        local metrics_obj_str="{"
        local first_metric=true

        while IFS=':' read -r key value; do
            key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

            if [[ "$value" == *us ]]; then
                local num_value=$(echo "$value" | sed 's/ us$//')
                local ms_value=$(LC_ALL=C awk -v num="$num_value" 'BEGIN { printf "%.3f", num / 1000 }')
                local base_metric_key=$(echo "$key" | tr ' ' '_' | tr '[:lower:]' '[:upper:]')
                local final_metric_key="$base_metric_key" # Default

                # Map specific metric names for consistency
                if [[ "$HARDWARE_CATEGORY" == GPU* ]]; then
                    case "$base_metric_key" in
                        "DEVICE_TIME") final_metric_key="GPU_DEVICE_TIME" ;;
                        "DEVICE_MEMCPY_TIME") final_metric_key="GPU_DEVICE_MEMCPY_TIME" ;;
                        # Add other GPU specific mappings here if needed
                    esac
                elif [[ "$HARDWARE_CATEGORY" == CPU* ]]; then
                    case "$base_metric_key" in
                        "CPU_TIME" | "TIME") final_metric_key="CPU_TIME" ;; # Handle "CPU Time" or just "Time"
                        "WALL_TIME") final_metric_key="WALL_TIME" ;;
                        # Add other CPU specific mappings here if needed
                    esac
                fi

                echo "INFO: Parsed metric: OriginalKey='$key', BaseKey='$base_metric_key', FinalKey='$final_metric_key', ValueMs='$ms_value'"

                if ! "$first_metric"; then metrics_obj_str+=","; fi
                metrics_obj_str+="\"$final_metric_key\": {\"value_ms\": $ms_value, \"unit\": \"ms\"}"
                first_metric=false
            fi
        done <<< "$stats_output"
        metrics_obj_str+="}"

        if echo "$metrics_obj_str" | jq -e . > /dev/null 2>&1; then
            METRICS_JSON_CONTENT=$(echo "$metrics_obj_str" | jq '.')
            echo "Successfully parsed metrics from stats output."
            STATS_RUN_SUCCESSFUL=true
        else
            echo "::warning::Could not construct valid JSON from stats output. Metrics object will be empty."
            echo "Problematic metrics string constructed: $metrics_obj_str"
            METRICS_JSON_CONTENT="{}"
        fi
    else
        echo "::warning::compute_xspace_stats_main failed with code $stats_exit_code. No metrics will be parsed from its output."
    fi
}

# Creates the final results.json file with run status, error messages, and metrics.
# Accepts: runner_exit_code as argument 1.
create_results_json() {
    local runner_exit_code="$1"
    local run_status_msg=""
    local error_msg_content=""

    if [ "$runner_exit_code" -ne 0 ]; then
        run_status_msg="FAILURE"
        error_msg_content="Runner failed with code $runner_exit_code"
    elif [ ! -f "$XSPACE_FILE_PATH" ]; then
        run_status_msg="SUCCESS_NO_PROFILE"
        error_msg_content="XSpace file not generated by successful run."
    elif [ "$STATS_RUN_SUCCESSFUL" = false ] ; then
        run_status_msg="STATS_FAILURE"
        error_msg_content="compute_xspace_stats_main failed or metrics parsing failed. Runner was successful."
    else
        run_status_msg="SUCCESS"
        error_msg_content=""
    fi

    jq -n \
      --arg bn "$BENCHMARK_NAME" \
      --arg cid "$CONFIG_ID" \
      --arg hc "$HARDWARE_CATEGORY" \
      --arg rs "$run_status_msg" \
      --arg em "$error_msg_content" \
      --arg cs "$COMMIT_SHA" \
      --arg wrid "$WORKFLOW_RUN_ID" \
      --argjson metrics "$METRICS_JSON_CONTENT" \
      '{
         benchmark_name: $bn,
         config_id: $cid,
         hardware_category: $hc,
         run_status: $rs,
         error_message: $em,
         commit_sha: $cs,
         workflow_run_id: $wrid,
         metrics: $metrics
       }' > "$RESULTS_JSON_FILE"

    if [ $? -eq 0 ]; then
        echo "Final results.json created at $RESULTS_JSON_FILE."
    else
        echo "::error::FATAL: Failed to create final results.json using jq."
        echo "FATAL JQ ERROR. Benchmark Name: $BENCHMARK_NAME, Run Status: $run_status_msg, Error: $error_msg_content" > "$RESULTS_JSON_FILE.txt"
        exit 1
    fi
}

# Debugging function to list generated files.
debug_file_check() {
    echo "DEBUG: Listing contents of OUTPUT_DIR ($OUTPUT_DIR):"
    ls -la "$OUTPUT_DIR"
    echo "DEBUG: Checking for RESULTS_JSON_FILE ($RESULTS_JSON_FILE):"
    if [ -f "$RESULTS_JSON_FILE" ]; then
      echo "DEBUG: RESULTS_JSON_FILE exists. Content (first 20 lines):"
      head -n 20 "$RESULTS_JSON_FILE"
    else
      echo "DEBUG: RESULTS_JSON_FILE does NOT exist."
      if [ -f "${RESULTS_JSON_FILE}.txt" ]; then
        echo "DEBUG: RESULTS_JSON_FILE.txt exists. Content:"
        cat "${RESULTS_JSON_FILE}.txt"
      else
        echo "DEBUG: RESULTS_JSON_FILE.txt also does NOT exist."
      fi
    fi
    echo "DEBUG: End of file check."
}

# --- Main Script Execution ---

main() {
    validate_inputs
    prepare_flags # Populates global xla_flags_array and runtime_flags_array

    local runner_exit_code # Declare local variable to hold the exit code
    execute_runner
    runner_exit_code=$? # Capture the exit code from the last executed command (execute_runner)

    process_stats # Populates global METRICS_JSON_CONTENT and STATS_RUN_SUCCESSFUL
    create_results_json "$runner_exit_code" # Pass the runner's exit code to the function

    debug_file_check

    if [ "$runner_exit_code" -ne 0 ]; then
      echo "::error::Benchmark run failed (HLO Runner Exit Code: $runner_exit_code)."
      exit "$runner_exit_code"
    fi

    echo "--- Run Benchmark Script Finished ---"
}

# Call the main function to start execution
main "$@"