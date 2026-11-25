#!/bin/bash
#
# Refuse to start unless every visible AMD GPU is an Instinct MI300X.
#
# This part is sourced by rocm_entrypoint.sh. It does nothing unless the image
# sets TRITON_REQUIRE_MI300X=1, which build.py --mi300x-only does, so images
# built for other GPUs are not affected.
#
# GPUs are identified by PCI device id read from the KFD topology in sysfs,
# not by marketing name: some MI-series parts report a generic "AMD Radeon
# Graphics" name, and gfx942 is shared by MI300X, MI308X, MI325X and MI300A.
#
# Environment:
#   TRITON_REQUIRE_MI300X=1       enable the check (set by the image)
#   TRITON_SKIP_GPU_CHECK=1       skip the check (CI, `tritonserver --help`,
#                                 hosts without a GPU)
#   TRITON_MI300X_DEVICE_IDS      space-separated PCI device ids accepted as
#                                 MI300X. Default "0x74a1 0x74a9".
#   TRITON_KFD_TOPOLOGY_DIR       topology directory, for tests. Default
#                                 /sys/class/kfd/kfd/topology/nodes.

_triton_mi300x_check() {
  local topo="${TRITON_KFD_TOPOLOGY_DIR:-/sys/class/kfd/kfd/topology/nodes}"
  local allowed="${TRITON_MI300X_DEVICE_IDS:-0x74a1 0x74a9}"
  local node props simd vendor dev hex id ok
  local gpus=0 bad=""

  for node in "${topo}"/*/; do
    props="${node}properties"
    [[ -r "${props}" ]] || continue

    simd=$(awk '$1=="simd_count"{print $2}' "${props}")
    # CPU nodes report no SIMDs.
    [[ "${simd:-0}" =~ ^[0-9]+$ ]] && (( simd > 0 )) || continue

    vendor=$(awk '$1=="vendor_id"{print $2}' "${props}")
    dev=$(awk '$1=="device_id"{print $2}' "${props}")
    gpus=$(( gpus + 1 ))

    if [[ ! "${dev}" =~ ^[0-9]+$ ]]; then
      bad="${bad} (unreadable device_id)"
      continue
    fi
    printf -v hex '0x%04x' "${dev}"

    ok=0
    for id in ${allowed}; do
      if [[ "${hex}" == "${id,,}" ]]; then
        ok=1
        break
      fi
    done
    if (( ok == 0 )) || [[ "${vendor}" != "4098" ]]; then
      bad="${bad} ${hex}"
    fi
  done

  if (( gpus == 0 )); then
    echo "ERROR: no AMD GPU is visible in this container." >&2
    echo "       This image supports only AMD Instinct MI300X. Pass /dev/kfd and" >&2
    echo "       /dev/dri to the container, or set TRITON_SKIP_GPU_CHECK=1 to skip" >&2
    echo "       this check (for example for 'tritonserver --help')." >&2
    return 1
  fi

  if [[ -n "${bad}" ]]; then
    echo "ERROR: this image supports only AMD Instinct MI300X (PCI device ids: ${allowed})." >&2
    echo "       Found unsupported GPU(s) with device id:${bad}" >&2
    echo "       Set TRITON_SKIP_GPU_CHECK=1 to skip this check at your own risk." >&2
    return 1
  fi

  echo "GPU check: ${gpus} AMD Instinct MI300X device(s) found."
  return 0
}

if [[ "${TRITON_REQUIRE_MI300X:-0}" == "1" ]]; then
  if [[ "${TRITON_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    echo "GPU check: skipped (TRITON_SKIP_GPU_CHECK=1)." >&2
  elif ! _triton_mi300x_check; then
    unset -f _triton_mi300x_check
    exit 1
  fi
fi
unset -f _triton_mi300x_check
