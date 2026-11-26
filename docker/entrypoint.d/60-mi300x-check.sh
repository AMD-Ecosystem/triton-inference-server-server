#!/bin/bash
#
# Refuse to start unless every visible AMD GPU is an Instinct gfx942 part.
#
# This part is sourced by rocm_entrypoint.sh. It does nothing unless the image
# sets TRITON_REQUIRE_MI300X=1, which build.py --mi300x-only does, so images
# built for other GPUs are not affected.
#
# The --mi300x-only image is compiled for gfx942 only. That code object runs
# on every gfx942 part: MI300X, MI308X, MI325X and MI300A. GPUs are therefore
# identified by the gfx target the KFD topology in sysfs reports
# (gfx_target_version 90402), not by marketing name: some MI-series parts
# report a generic "AMD Radeon Graphics" name. gfx950 (MI350X, MI355X) and
# every non-AMD device are refused.
#
# Environment:
#   TRITON_REQUIRE_MI300X=1       enable the check (set by the image)
#   TRITON_SKIP_GPU_CHECK=1       skip the check (CI, `tritonserver --help`,
#                                 hosts without a GPU)
#   TRITON_MI300X_DEVICE_IDS      optional space-separated PCI device ids. When
#                                 set, a GPU must also have one of these ids,
#                                 for example "0x74a1 0x74a9" to accept only
#                                 MI300X. Unset (the default) accepts any
#                                 gfx942 GPU.
#   TRITON_KFD_TOPOLOGY_DIR       topology directory, for tests. Default
#                                 /sys/class/kfd/kfd/topology/nodes.

_triton_gfx_name() {
  # 90402 -> gfx942, 90500 -> gfx950 (major*10000 + minor*100 + stepping)
  local v="$1"
  printf 'gfx%d%d%x' "$(( v / 10000 ))" "$(( (v / 100) % 100 ))" "$(( v % 100 ))"
}

_triton_mi300x_check() {
  local topo="${TRITON_KFD_TOPOLOGY_DIR:-/sys/class/kfd/kfd/topology/nodes}"
  local allowed="${TRITON_MI300X_DEVICE_IDS:-}"
  local want_gfx=90402
  local node props simd vendor dev gfx hex id ok
  local gpus=0 bad=""

  for node in "${topo}"/*/; do
    props="${node}properties"
    [[ -r "${props}" ]] || continue

    simd=$(awk '$1=="simd_count"{print $2}' "${props}")
    # CPU nodes report no SIMDs.
    [[ "${simd:-0}" =~ ^[0-9]+$ ]] && (( simd > 0 )) || continue

    vendor=$(awk '$1=="vendor_id"{print $2}' "${props}")
    dev=$(awk '$1=="device_id"{print $2}' "${props}")
    gfx=$(awk '$1=="gfx_target_version"{print $2}' "${props}")
    gpus=$(( gpus + 1 ))

    if [[ ! "${dev}" =~ ^[0-9]+$ ]] || [[ ! "${gfx}" =~ ^[0-9]+$ ]]; then
      bad="${bad} (unreadable device_id or gfx_target_version)"
      continue
    fi
    printf -v hex '0x%04x' "${dev}"

    if [[ "${vendor}" != "4098" ]]; then
      bad="${bad} ${hex}(vendor ${vendor})"
      continue
    fi
    if (( gfx != want_gfx )); then
      bad="${bad} ${hex}($(_triton_gfx_name "${gfx}"))"
      continue
    fi
    if [[ -n "${allowed}" ]]; then
      ok=0
      for id in ${allowed}; do
        if [[ "${hex}" == "${id,,}" ]]; then
          ok=1
          break
        fi
      done
      if (( ok == 0 )); then
        bad="${bad} ${hex}(not in TRITON_MI300X_DEVICE_IDS)"
      fi
    fi
  done

  if (( gpus == 0 )); then
    echo "ERROR: no AMD GPU is visible in this container." >&2
    echo "       This image supports only AMD Instinct gfx942 GPUs (MI300X, MI308X," >&2
    echo "       MI325X, MI300A). Pass /dev/kfd and /dev/dri to the container, or set" >&2
    echo "       TRITON_SKIP_GPU_CHECK=1 to skip this check (for example for" >&2
    echo "       'tritonserver --help')." >&2
    return 1
  fi

  if [[ -n "${bad}" ]]; then
    echo "ERROR: this image supports only AMD Instinct gfx942 GPUs (MI300X, MI308X, MI325X, MI300A)." >&2
    if [[ -n "${allowed}" ]]; then
      echo "       Accepted PCI device ids (TRITON_MI300X_DEVICE_IDS): ${allowed}" >&2
    fi
    echo "       Found unsupported GPU(s), PCI device id (gfx target):${bad}" >&2
    echo "       Set TRITON_SKIP_GPU_CHECK=1 to skip this check at your own risk." >&2
    return 1
  fi

  echo "GPU check: ${gpus} AMD Instinct gfx942 device(s) found."
  return 0
}

if [[ "${TRITON_REQUIRE_MI300X:-0}" == "1" ]]; then
  if [[ "${TRITON_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    echo "GPU check: skipped (TRITON_SKIP_GPU_CHECK=1)." >&2
  elif ! _triton_mi300x_check; then
    unset -f _triton_mi300x_check _triton_gfx_name
    exit 1
  fi
fi
unset -f _triton_mi300x_check _triton_gfx_name
