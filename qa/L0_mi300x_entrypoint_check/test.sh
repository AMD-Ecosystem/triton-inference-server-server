#!/bin/bash
# Tests docker/entrypoint.d/60-mi300x-check.sh against fake KFD topologies.
# No GPU is needed: the part reads TRITON_KFD_TOPOLOGY_DIR, so each case builds
# a small sysfs-like tree.
#
# The --mi300x-only image is compiled for gfx942, so the check accepts every
# gfx942 GPU (MI300X, MI308X, MI325X, MI300A) and refuses gfx950 and non-AMD
# devices. TRITON_MI300X_DEVICE_IDS narrows that to specific PCI device ids.
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
#   node-spec: cpu | <device_id decimal>[:<gfx_target_version>[:<vendor_id>]]
#   gfx_target_version defaults to 90402 (gfx942), vendor_id to 4098 (AMD).
# Device ids: MI300A 0x74a0 = 29856, MI300X 0x74a1 = 29857, MI308X 0x74a2 = 29858,
# MI325X 0x74a5 = 29861, MI300X (0x74a9) = 29865, MI355X 0x75a0 = 30112 (gfx950, 90500).
make_topology() {
    local dir="$1"; shift
    local i=0 spec dev gfx vendor
    mkdir -p "${dir}"
    for spec in "$@"; do
        mkdir -p "${dir}/${i}"
        if [ "${spec}" = "cpu" ]; then
            printf 'simd_count 0\ngfx_target_version 0\nvendor_id 0\ndevice_id 0\n' \
                > "${dir}/${i}/properties"
        else
            IFS=: read -r dev gfx vendor <<< "${spec}"
            printf 'simd_count 1024\ngfx_target_version %s\nvendor_id %s\ndevice_id %s\n' \
                "${gfx:-90402}" "${vendor:-4098}" "${dev}" > "${dir}/${i}/properties"
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

make_topology "${WORK}/mi300x"   cpu 29857
make_topology "${WORK}/eight"    cpu 29857 29857 29857 29857 29857 29857 29857 29857
make_topology "${WORK}/mi300x9"  cpu 29865
make_topology "${WORK}/mi308x"   cpu 29858
make_topology "${WORK}/mi325x"   cpu 29861
make_topology "${WORK}/mi300a"   cpu 29856
make_topology "${WORK}/mixed942" cpu 29857 29858 29861
make_topology "${WORK}/mi355x"   cpu 30112:90500
make_topology "${WORK}/mixed"    cpu 29857 30112:90500
make_topology "${WORK}/nvidia"   cpu 9999:90402:4318
make_topology "${WORK}/nogpu"    cpu
make_topology "${WORK}/nogfx"    cpu 29857:abc
mkdir -p "${WORK}/empty"

run_case "one MI300X (0x74a1) passes" pass "1 AMD Instinct gfx942" "${WORK}/mi300x" TRITON_REQUIRE_MI300X=1
run_case "eight MI300X pass" pass "8 AMD Instinct gfx942" "${WORK}/eight" TRITON_REQUIRE_MI300X=1
run_case "MI300X variant (0x74a9) passes" pass "1 AMD Instinct gfx942" "${WORK}/mi300x9" TRITON_REQUIRE_MI300X=1
run_case "MI308X (0x74a2) passes: gfx942" pass "1 AMD Instinct gfx942" "${WORK}/mi308x" TRITON_REQUIRE_MI300X=1
run_case "MI325X (0x74a5) passes: gfx942" pass "1 AMD Instinct gfx942" "${WORK}/mi325x" TRITON_REQUIRE_MI300X=1
run_case "MI300A (0x74a0) passes: gfx942" pass "1 AMD Instinct gfx942" "${WORK}/mi300a" TRITON_REQUIRE_MI300X=1
run_case "mixed gfx942 parts pass" pass "3 AMD Instinct gfx942" "${WORK}/mixed942" TRITON_REQUIRE_MI300X=1
run_case "MI355X (gfx950) is rejected" fail "0x75a0(gfx950)" "${WORK}/mi355x" TRITON_REQUIRE_MI300X=1
run_case "mixed gfx942 and gfx950 is rejected" fail "0x75a0(gfx950)" "${WORK}/mixed" TRITON_REQUIRE_MI300X=1
run_case "non-AMD vendor is rejected" fail "vendor 4318" "${WORK}/nvidia" TRITON_REQUIRE_MI300X=1
run_case "unreadable gfx target is rejected" fail "unreadable" "${WORK}/nogfx" TRITON_REQUIRE_MI300X=1
run_case "CPU-only topology is rejected" fail "no AMD GPU" "${WORK}/nogpu" TRITON_REQUIRE_MI300X=1
run_case "missing topology is rejected" fail "no AMD GPU" "${WORK}/missing" TRITON_REQUIRE_MI300X=1
run_case "override skips the check on MI355X" pass "skipped" "${WORK}/mi355x" \
    TRITON_REQUIRE_MI300X=1 TRITON_SKIP_GPU_CHECK=1
run_case "override skips the check with no GPU" pass "skipped" "${WORK}/nogpu" \
    TRITON_REQUIRE_MI300X=1 TRITON_SKIP_GPU_CHECK=1
run_case "check is off when the image does not require MI300X" pass "" "${WORK}/mi355x"
run_case "PCI id list narrows to MI300X: 0x74a1 passes" pass "1 AMD Instinct gfx942" "${WORK}/mi300x" \
    TRITON_REQUIRE_MI300X=1 "TRITON_MI300X_DEVICE_IDS=0x74a1 0x74a9"
run_case "PCI id list narrows to MI300X: MI308X is rejected" fail "0x74a2(not in TRITON_MI300X_DEVICE_IDS)" \
    "${WORK}/mi308x" TRITON_REQUIRE_MI300X=1 "TRITON_MI300X_DEVICE_IDS=0x74a1 0x74a9"
run_case "PCI id list is case-insensitive" pass "1 AMD Instinct gfx942" "${WORK}/mi308x" \
    TRITON_REQUIRE_MI300X=1 "TRITON_MI300X_DEVICE_IDS=0X74A2"

if [ "${RET}" -eq 0 ]; then
    echo "*** Test Passed (${CASE} cases)"
else
    echo "*** Test FAILED"
fi
exit "${RET}"
