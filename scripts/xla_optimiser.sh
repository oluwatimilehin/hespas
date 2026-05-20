#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(realpath "$(dirname "$0")")"

INPUT_FILE=
OUTPUT_FILE=
XLA_TRANSLATE=
HLO_OPT=
USE_SPEC="yes"
GPU_SPEC=
RUN_DOCKER="no"
DISABLE_PASSES="yes"
DISABLED_PASSES=
NON_SPMD_DISABLED_PASSES="async-collective-conversion"
SPMD_DISABLED_PASSES="${NON_SPMD_DISABLED_PASSES},gpu-reduce-scatter-combiner,spmd-partitioning,spmd-partitioner,spmd_partitioner"
IS_SPMD=""
DEBUG="no"
DUMP_DIR=
DEFAULT_DUMP_SUBDIR="passes_dump"
DUMP_PASS_RE=".*"
KEEP_HLO="no"

usage()
{
    echo "$0: XLA optimiser script"
    echo "Usage: $0 [-i|--input-file] INPUT_FILE [-o|--output-file] OUTPUT_FILE ([-g|--spec-file] GPU_SPEC_FILE|--no-spec) ([-d|--disabled-passes DISABLED_PASSES_LIST]|--no-disabled-passes) [--spmd|--no-spmd] [--docker] [--xla-translate XLA_TRANSLATE_PATH] [--hlo-opt HLO_OPT_PATH] [--debug] [--dump-dir DUMP_DIR] [--dump-pass-re DUMP_PASS_RE] [--keep-hlo]"
    echo "-i|--input-file: Input StableHLO MLIR file (required)"
    echo "-o|--output-file: Output optimised StableHLO MLIR file (required)"
    echo "-g|--spec-file: GPU spec file from the XLA repository to use (required if no --no-spec)"
    echo "--no-spec: Don't use a spec file, compile on the current GPU (mutually exclusive with -g|--spec-file)"
    echo "-d|--disabled-passes: Command seperated list of disabled passes. (optional)"
    echo "                      By default: '${NON_SPMD_DISABLED_PASSES}' if not spmd or '${SPMD_DISABLED_PASSES}' if spmd"
    echo "--no-disabled-passes: Don't disable any passes (optional)"
    echo "--docker: Run the opt in the docker (optional)"
    echo "--spmd: Use SPMD disabled passes (optional)"
    echo "--no-spmd: Don't use SPMD disabled passes (the default) (optional)"
    echo "--xla-translate: Path to xla-translate binary (optional, currently '$(which xla-translate 2> /dev/null || echo "Not found")')"
    echo "--hlo-opt: Path to hlo-opt binary (optional, currently '$(which hlo-opt 2> /dev/null || echo "Not found")')"
    echo "--debug: Enable debug HLO pass dumping (optional)"
    echo "--dump-dir: Directory to dump debug HLO passes too (optional, defaults to '${DEFAULT_DUMP_SUBDIR}' in the directory of the output file)"
    echo "--dump-pass-re: Pass regex for dumping (optional, defaults to '${DUMP_PASS_RE}'"
    echo "--keep-hlo: Keep intermediate HLO (optional)"
}

check_param()
{
    PARAM="$1"
    shift
    METAVAR="$1"
    shift
    INPUT_VAR="$1"
    shift
    if [ -z "${INPUT_VAR}" ]
    then
        echo "${PARAM} ${METAVAR} is required"
        usage
        exit 1
    fi
}

check_filepath()
{
    PARAM="$1"
    shift
    METAVAR="$1"
    shift
    INPUT_VAR="$1"
    shift
    if [ "$#" -ne 0 ]
    then
        IS_OUTPUT="$1"
        shift
    fi
    check_param "${PARAM}" "${METAVAR}" "${INPUT_VAR}"
    if [ "${IS_OUTPUT}" != "output" ] && [ ! -f "${INPUT_VAR}" ]
    then
        echo "Cannot find ${METAVAR} '${INPUT_VAR}'"
        exit 1
    elif [ "${IS_OUTPUT}" = "output" ] && [ ! -d "$(dirname "${INPUT_VAR}")" ]
    then
        echo "Cannot find output path for ${METAVAR} '${INPUT_VAR}'"
        exit 1
    fi
}

process_args()
{
    while [ "$#" -gt 0 ]
    do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -i)
                shift
                INPUT_FILE="$1"
                ;;
            -o)
                shift
                OUTPUT_FILE="$1"
                ;;
            -g)
                shift
                GPU_SPEC="$1"
                ;;
            -d|--disabled-passes)
                shift
                DISABLED_PASSES="$1"
                ;;
            --no-disabled-passes)
                DISABLE_PASSES="no"
                ;;
            --no-spec)
                USE_SPEC="no"
                ;;
            --spmd)
                IS_SPMD="yes"
                ;;
            --no-spmd)
                IS_SPMD="no"
                ;;
            --docker)
                RUN_DOCKER="yes"
                ;;
            --xla-translate)
                shift
                XLA_TRANSLATE="$1"
                ;;
            --hlo-opt)
                shift
                HLO_OPT="$1"
                ;;
            --debug)
                DEBUG="yes"
                ;;
            --dump-dir)
                shift
                DUMP_DIR="$1"
                ;;
            --dump-pass-re)
                shift
                DUMP_PASS_RE="$1"
                ;;
            --keep-hlo)
                KEEP_HLO="yes"
                ;;
            *)
                echo "Unknown argument '$1'"
                echo ""
                usage
                exit 1
                ;;
        esac
        shift
    done

    if [ -z "${XLA_TRANSLATE}" ]
    then
        XLA_TRANSLATE="$(which xla-translate 2> /dev/null)"
    fi

    if [ -z "${HLO_OPT}" ]
    then
        HLO_OPT="$(which hlo-opt 2> /dev/null)"
    fi

    if [ "${DISABLE_PASSES}" = "yes" ] && [ -z "${DISABLED_PASSES}" ]
    then
        if [ "${IS_SPMD}" = "yes" ]
        then
            DISABLED_PASSES="${SPMD_DISABLED_PASSES}"
        else
            DISABLED_PASSES="${NON_SPMD_DISABLED_PASSES}"
        fi
    fi

    check_filepath "-i" "INPUT_FILE" "${INPUT_FILE}"
    check_filepath "-o" "OUTPUT_FILE" "${OUTPUT_FILE}" "output"
    if [ "${USE_SPEC}" = "yes" ]
    then
        check_filepath "-g" "GPU_SPEC_FILE" "${GPU_SPEC}"
    fi

    if [ -n "${DEBUG}" ] && [ -z "${DUMP_DIR}" ]
    then
        DUMP_DIR="$(dirname "${OUTPUT_FILE}")/${DEFAULT_DUMP_SUBDIR}"
    fi
}

run_opt()
{
    XLA_TRANSLATE="$1"
    shift
    XLA_HLO_OPT="$1"
    shift
    INPUT_FILE="$(realpath "$1")"
    shift
    OUTPUT_FILE="$(realpath -m "$1")"
    shift
    USE_SPEC="$1"
    shift
    GPU_SPEC=
    if [ "${USE_SPEC}" = "yes" ]
    then
        GPU_SPEC="$(realpath "$1")"
    fi
    shift
    DISABLE_PASSES="$1"
    shift
    DISABLED_PASSES=
    if [ "${DISABLE_PASSES}" = "yes" ]
    then
        DISABLED_PASSES="$1"
    fi
    shift
    DEBUG=
    if [ "$1" = "yes" ]
    then
        DEBUG="yes"
    fi
    shift
    DUMP_DIR=
    if [ -n "${DEBUG}" ]
    then
        DUMP_DIR="$(realpath -m "$1")"
    fi
    shift
    DUMP_PASS_RE=
    if [ -n "${DEBUG}" ]
    then
        DUMP_PASS_RE="$1"
    fi
    shift
    KEEP_HLO="$1"

    USE_LATENCY_HIDING_SCHED=
    if [ -n "${GPU_SPEC}" ] || [ -n "${DISABLED_PASSES}" ]
    then
        USE_LATENCY_HIDING_SCHED="yes"
    fi

    if [ ! -x "${XLA_TRANSLATE}" ]
    then
        echo "Cannot find xla-translate binary"
        exit 1
    fi

    if [ ! -x "${HLO_OPT}" ]
    then
        echo "Cannot find hlo-opt binary"
        exit 1
    fi

    UNOP_HLO_TEMP="${INPUT_FILE}.unop.hlo"
    OP_HLO_TEMP="${INPUT_FILE}.op.hlo"

    rm -f "${UNOP_HLO_TEMP}" "${OP_HLO_TEMP}"
    if [ -n "${DEBUG}" ]
    then
        mkdir -p "${DUMP_DIR}"
    fi

    OUTPUT_DIR="$(dirname "${OUTPUT_FILE}")"
    if [ ! -d "${OUTPUT_DIR}" ]
    then
        mkdir -p "${OUTPUT_DIR}"
    fi

    "${XLA_TRANSLATE}" \
        --stablehlo-to-hlo-text \
        -o="${UNOP_HLO_TEMP}" \
        "${INPUT_FILE}"

    "${XLA_HLO_OPT}" \
        --platform=gpu \
        ${GPU_SPEC:+--xla_gpu_target_config_filename="${GPU_SPEC}"} \
        ${DISABLED_PASSES:+--xla_disable_hlo_passes="${DISABLED_PASSES}"} \
        ${DISABLED_PASSES:+--xla_gpu_disable_async_collectives="ALLCOLLECTIVES"} \
        ${USE_LATENCY_HIDING_SCHED:+--xla_gpu_enable_latency_hiding_scheduler=false} \
        ${GPU_SPEC:+--xla_gpu_autotune_level=0} \
        ${DEBUG:+--xla_dump_to="${DUMP_DIR}"} \
        ${DEBUG:+--xla_dump_hlo_pass_re="${DUMP_PASS_RE}"} \
        --o="${OP_HLO_TEMP}" \
        "${UNOP_HLO_TEMP}"

    "${XLA_TRANSLATE}" \
        --emit-stablehlo \
        --hlo-text-to-stablehlo \
        --hlo-flatten-computation-args-result \
        --hlo-import-all-computations \
        --enable-cse-in-irtranslator \
        -o="${OUTPUT_FILE}" \
        "${OP_HLO_TEMP}"

    if [ "${KEEP_HLO}" = "yes" ]
    then
        rm "${UNOP_HLO_TEMP}" "${OP_HLO_TEMP}"
    fi
}

process_args "$@"
set -e

run_opt "${XLA_TRANSLATE}" "${HLO_OPT}" "${INPUT_FILE}" "${OUTPUT_FILE}" "${USE_SPEC}" "${GPU_SPEC}" "${DISABLE_PASSES}" "${DISABLED_PASSES}" "${DEBUG}" "${DUMP_DIR}" "${DUMP_PASS_RE}" "${KEEP_HLO}"
