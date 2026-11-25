#!/bin/bash
# Tests docker/entrypoint.d/60-mi300x-check.sh against fake KFD topologies.
# No GPU is needed: the part reads TRITON_KFD_TOPOLOGY_DIR, so each case builds
# a small sysfs-like tree.
#
# Usage: ./test.sh            (from this directory or anywhere)
#        ENTRYPOINT_PART=/opt/rocm/entrypoint.d/60-mi300x-check.sh ./test.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="${ENTRYPOINT_PART:-${HERE}/../../docker/entrypoint.d/60-mi300x-check.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

RET=0
CASE=0

# make_topology <dir> <node-spec>...
#   node-spec: cpu | <device_id decimal>
# MI300X 0x74a1 = 29857, MI300X (0x74a9) = 29865, MI355X 0x75a0 = 30112,
# MI325X 0x74a5 = 29861.
make_topology() {
    local dir="$1"; shift
    local i=0 spec
    mkdir -p "${dir}"
    for spec in "$@"; do
        mkdir -p "${dir}/${i}"
        if [ "${spec}" = "cpu" ]; then
            printf 'simd_count 0\ngfx_target_version 0\nvendor_id 0\ndevice_id 0\n' \
                > "${dir}/${i}/properties"
        else
            printf 'simd_count 1024\ngfx_target_version 90402\nvendor_id 4098\ndevice_id %s\n' \
                "${spec}" > "${dir}/${i}/properties"
        fi
        i=$((i + 1))
    done
}

# run_case <name> <expected: pass|fail> <expected-text> <topology-dir> [VAR=value...]
run_case() {
    local name="$1" expect="$2" text="$3" topo="$4"; shift 4
    local out rc
    CASE=$((CASE + 1))
    out="$(env -i PATH="${PATH}" TRITON_KFD_TOPOLOGY_DIR="${topo}" "$@" \
        bash -c "source '${PART}'" 2>&1)"
    rc=$?

    local want_rc=0
    [ "${expect}" = "fail" ] && want_rc=1
    if [ "${rc}" -ne "${want_rc}" ]; then
        echo "FAIL [${CASE}] ${name}: exit ${rc}, expected ${want_rc}"
        echo "${out}" | sed 's/^/    /'
        RET=1
        return
    fi
    if ! grep -q -- "${text}" <<< "${out}"; then
        echo "FAIL [${CASE}] ${name}: output lacks '${text}'"
        echo "${out}" | sed 's/^/    /'
        RET=1
        return
    fi
    echo "PASS [${CASE}] ${name}"
}

make_topology "${WORK}/one"      cpu 29857
make_topology "${WORK}/eight"    cpu 29857 29857 29857 29857 29857 29857 29857 29857
make_topology "${WORK}/hf"       cpu 29865
make_topology "${WORK}/mi355"    cpu 30112
make_topology "${WORK}/mi325"    cpu 29861
make_topology "${WORK}/mixed"    cpu 29857 30112
make_topology "${WORK}/nogpu"    cpu
mkdir -p "${WORK}/empty"

run_case "one MI300X (0x74a1) passes" pass "1 AMD Instinct MI300X" "${WORK}/one" TRITON_REQUIRE_MI300X=1
run_case "eight MI300X pass" pass "8 AMD Instinct MI300X" "${WORK}/eight" TRITON_REQUIRE_MI300X=1
run_case "MI300X variant (0x74a9) passes" pass "1 AMD Instinct MI300X" "${WORK}/hf" TRITON_REQUIRE_MI300X=1
run_case "MI355X (0x75a0) is rejected" fail "0x75a0" "${WORK}/mi355" TRITON_REQUIRE_MI300X=1
run_case "MI325X (0x74a5) is rejected" fail "0x74a5" "${WORK}/mi325" TRITON_REQUIRE_MI300X=1
run_case "mixed MI300X and MI355X is rejected" fail "0x75a0" "${WORK}/mixed" TRITON_REQUIRE_MI300X=1
run_case "CPU-only topology is rejected" fail "no AMD GPU" "${WORK}/nogpu" TRITON_REQUIRE_MI300X=1
run_case "missing topology is rejected" fail "no AMD GPU" "${WORK}/missing" TRITON_REQUIRE_MI300X=1
run_case "override skips the check on MI355X" pass "skipped" "${WORK}/mi355" \
    TRITON_REQUIRE_MI300X=1 TRITON_SKIP_GPU_CHECK=1
run_case "override skips the check with no GPU" pass "skipped" "${WORK}/nogpu" \
    TRITON_REQUIRE_MI300X=1 TRITON_SKIP_GPU_CHECK=1
run_case "check is off when the image does not require MI300X" pass "" "${WORK}/mi355"
run_case "custom device id list is honoured" pass "1 AMD Instinct MI300X" "${WORK}/mi325" \
    TRITON_REQUIRE_MI300X=1 "TRITON_MI300X_DEVICE_IDS=0x74a1 0x74a5"

if [ "${RET}" -eq 0 ]; then
    echo "*** Test Passed (${CASE} cases)"
else
    echo "*** Test FAILED"
fi
exit "${RET}"
