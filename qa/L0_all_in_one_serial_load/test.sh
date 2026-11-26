#!/bin/bash
# Tests docker/entrypoint.d/80-model-load-threads.sh. No GPU is needed: the part
# only rewrites the entrypoint's arguments, so each case sources it with a
# command line and prints what the entrypoint would exec.
#
# Usage: ./test.sh            (from this directory or anywhere)
#        ENTRYPOINT_PART=/opt/rocm/entrypoint.d/80-model-load-threads.sh ./test.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="${ENTRYPOINT_PART:-${HERE}/../../docker/entrypoint.d/80-model-load-threads.sh}"

RET=0
CASE=0

# run_case <name> <expected-args> <expected-text-in-output|-> <ENV=value...|--> <command...>
#   expected-args: the arguments the entrypoint would exec, joined with "|"
#   expected-text: text the part must print, or "-" for no output at all
run_case() {
    local name="$1" want="$2" text="$3"; shift 3
    local -a envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local out got
    CASE=$((CASE + 1))
    out="$(env -i PATH="${PATH}" "${envs[@]}" bash -c '
        part="$1"; shift
        source "$part"
        printf "ARGS:"
        printf "%s|" "$@"
        echo
    ' bash "${PART}" "$@" 2>&1)"
    got="$(sed -n 's/^ARGS://p' <<< "${out}")"
    if [ "${got}" != "${want}" ]; then
        echo "FAIL [${CASE}] ${name}: args '${got}', expected '${want}'"
        RET=1
        return
    fi
    local printed
    printed="$(grep -v '^ARGS:' <<< "${out}")"
    if [ "${text}" = "-" ]; then
        if [ -n "${printed}" ]; then
            echo "FAIL [${CASE}] ${name}: unexpected output: ${printed}"
            RET=1
            return
        fi
    elif ! grep -q -- "${text}" <<< "${printed}"; then
        echo "FAIL [${CASE}] ${name}: output lacks '${text}': ${printed}"
        RET=1
        return
    fi
    echo "PASS [${CASE}] ${name}"
}

ON=TRITON_MODEL_LOAD_THREAD_COUNT=1

run_case "variable unset leaves tritonserver alone" \
    "tritonserver|--model-repository=/models|" - -- \
    tritonserver --model-repository=/models
run_case "empty variable leaves tritonserver alone" \
    "tritonserver|--model-repository=/models|" - TRITON_MODEL_LOAD_THREAD_COUNT= -- \
    tritonserver --model-repository=/models
run_case "variable=default leaves tritonserver alone" \
    "tritonserver|--model-repository=/models|" - TRITON_MODEL_LOAD_THREAD_COUNT=default -- \
    tritonserver --model-repository=/models
run_case "adds the flag after tritonserver" \
    "tritonserver|--model-load-thread-count=1|--model-repository=/models|" "Model load threads: 1" "${ON}" -- \
    tritonserver --model-repository=/models
run_case "adds the flag for a full path" \
    "/opt/tritonserver/bin/tritonserver|--model-load-thread-count=1|--exit-on-error=false|" "Model load threads: 1" "${ON}" -- \
    /opt/tritonserver/bin/tritonserver --exit-on-error=false
run_case "uses the configured count" \
    "tritonserver|--model-load-thread-count=2|" "Model load threads: 2" TRITON_MODEL_LOAD_THREAD_COUNT=2 -- \
    tritonserver
run_case "keeps an explicit --model-load-thread-count=N" \
    "tritonserver|--model-load-thread-count=4|--model-repository=/models|" - "${ON}" -- \
    tritonserver --model-load-thread-count=4 --model-repository=/models
run_case "keeps an explicit --model-load-thread-count N" \
    "tritonserver|--model-load-thread-count|4|" - "${ON}" -- \
    tritonserver --model-load-thread-count 4
run_case "ignores a zero count" \
    "tritonserver|--model-repository=/models|" "WARNING: ignoring" TRITON_MODEL_LOAD_THREAD_COUNT=0 -- \
    tritonserver --model-repository=/models
run_case "ignores a non-number" \
    "tritonserver|--model-repository=/models|" "WARNING: ignoring" TRITON_MODEL_LOAD_THREAD_COUNT=abc -- \
    tritonserver --model-repository=/models
run_case "leaves other commands alone" \
    "python3|/run_smoke.py|" - "${ON}" -- \
    python3 /run_smoke.py
run_case "leaves a tritonserver-like name alone" \
    "mytritonserver|--x|" - "${ON}" -- \
    mytritonserver --x
run_case "leaves an empty command line alone" \
    "|" - "${ON}" --
run_case "keeps arguments with spaces and quotes" \
    "tritonserver|--model-load-thread-count=1|--log-file=/tmp/a b.log|--x='y'|" "Model load threads: 1" "${ON}" -- \
    tritonserver "--log-file=/tmp/a b.log" "--x='y'"

if [ "${RET}" -eq 0 ]; then
    echo "All ${CASE} cases passed"
else
    echo "Some cases failed"
fi
exit "${RET}"
