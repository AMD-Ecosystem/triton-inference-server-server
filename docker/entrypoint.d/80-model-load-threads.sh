#!/bin/bash
#
# Load models one at a time when the image holds more than one GPU backend.
#
# This part is sourced by rocm_entrypoint.sh, which then runs "$@". It does
# nothing unless the image sets TRITON_MODEL_LOAD_THREAD_COUNT, which build.py
# does for ROCm images that contain the TensorFlow backend together with the
# ONNX Runtime or PyTorch backend (the all-in-one image).
#
# Why: tritonserver loads each backend library the first time a model needs it,
# on up to four threads at once. libtensorflow_cc.so and libtensorflow_framework.so
# register their GPU code in static constructors, so dlopen of the TensorFlow
# backend calls __hipRegisterFatBinary -> amd_comgr_do_action while it holds the
# dynamic loader lock. If another thread is inside comgr at that moment (for
# example the ONNX Runtime backend in hipStreamCreateWithPriority), comgr calls
# dladdr, which needs the loader lock. Each thread then holds the lock the other
# needs and tritonserver never serves requests. Loading models on one thread
# avoids the overlap. Images with a single GPU backend are not affected and
# keep Triton's default.
#
# When the command is tritonserver and the caller did not pass
# --model-load-thread-count, this part inserts
# --model-load-thread-count=$TRITON_MODEL_LOAD_THREAD_COUNT after it.
#
# Environment:
#   TRITON_MODEL_LOAD_THREAD_COUNT  positive integer, set by the image to 1.
#                                   "default" or unset leaves Triton's default.
#
# This only covers commands the entrypoint runs directly (docker run IMAGE
# tritonserver ...). A shell that starts tritonserver itself, or a command
# that overrides --entrypoint, must pass the flag explicitly.
#
# The code runs at the top level of the sourced file, not in a function, so
# that set -- changes the entrypoint's own arguments.

if [ -n "${TRITON_MODEL_LOAD_THREAD_COUNT:-}" ] &&
   [ "${TRITON_MODEL_LOAD_THREAD_COUNT}" != "default" ]; then
    case "${1:-}" in
        tritonserver|*/tritonserver)
            _triton_has_flag=0
            for _triton_arg in "$@"; do
                case "${_triton_arg}" in
                    --model-load-thread-count|--model-load-thread-count=*)
                        _triton_has_flag=1
                        ;;
                esac
            done
            if [ "${_triton_has_flag}" = 0 ]; then
                if [[ "${TRITON_MODEL_LOAD_THREAD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
                    set -- "$1" "--model-load-thread-count=${TRITON_MODEL_LOAD_THREAD_COUNT}" "${@:2}"
                    echo "Model load threads: ${TRITON_MODEL_LOAD_THREAD_COUNT} (set TRITON_MODEL_LOAD_THREAD_COUNT=default or pass --model-load-thread-count to change)"
                else
                    echo "WARNING: ignoring TRITON_MODEL_LOAD_THREAD_COUNT='${TRITON_MODEL_LOAD_THREAD_COUNT}' (want a positive integer or 'default')"
                fi
            fi
            unset _triton_has_flag _triton_arg
            ;;
    esac
fi
