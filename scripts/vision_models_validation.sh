#!/bin/bash

WORKLOADS_DIR="workloads"
CONFIG_PATH="configs/nodes/TPUv5e/roofline/config.json"
OUTPUT_DIR="workloads/estimates"
V5E_PB_PATH="tpuv5e.txtpb"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

for model_dir in "$WORKLOADS_DIR"/*/; do
    model_name=$(basename "$model_dir")
    
    if [ "$model_name" == "estimates" ]; then
        continue
    fi

    echo "=================================================="
    echo "Processing Model: $model_name"
    echo "=================================================="

    for mlir_file in "$model_dir"/*.mlir; do
        [ -e "$mlir_file" ] || continue
        if [[ "$mlir_file" == *".opt.mlir" ]]; then
            continue
        fi

        base_name=$(basename "$mlir_file" .mlir)
        
        echo "--> Found configuration: $base_name"

        unopt_hlo="$model_dir/${base_name}.unopt.hlo"
        opt_hlo="$model_dir/${base_name}.opt.hlo"
        opt_mlir="$model_dir/${base_name}.opt.mlir"
        estimate_out="$OUTPUT_DIR/${base_name}"

        echo "    [1/4] Translating StableHLO to HLO Text..."
        xla-translate --stablehlo-to-hlo-text \
            -o "$unopt_hlo" \
            "$mlir_file"

        echo "    [2/4] Optimizing HLO via hlo-opt..."
        hlo-opt --platform=gpu \
            --xla_gpu_target_config_filename="$V5E_PB_PATH" \
            --xla_gpu_autotune_level=0 \
            --xla_disable_hlo_passes=async-collective-conversion,gpu-reduce-scatter-combiner,spmd-partitioning,spmd-partitioner,spmd_partitioner \
            --xla_gpu_enable_cublaslt=false  \
            --o="$opt_hlo" \
            "$unopt_hlo"

        echo "    [3/4] Translating Optimized HLO back to StableHLO..."
        xla-translate --emit-stablehlo \
            --hlo-text-to-stablehlo \
            --hlo-flatten-computation-args-result \
            --hlo-import-all-computations \
            --enable-cse-in-irtranslator \
            -o "$opt_mlir" \
            "$opt_hlo"
            
        echo "    [4/4] Running Roofline Estimator..."
        python -m src.hespas.estimator.roofline_estimator --config-file "$CONFIG_PATH" \
            --mlir-file "$opt_mlir" --memory-compute-parallelism 1 \
            --log-path "$estimate_out-full-overlap.log"

        echo "    ✓ Finished processing: $base_name"
        echo "--------------------------------------------------"
    done
done

echo "Pipeline complete! All estimates saved to: $OUTPUT_DIR"