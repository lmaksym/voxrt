#!/usr/bin/env bash
# x86 benchmark box on Google Cloud (C4D = AMD Turin / Zen 5, full-width AVX-512), same interface as hetzner.sh.
# Only ever touches instances labelled purpose=parakeet-bench.
#
#   VOXRT_GCP_PROJECT=<project> scripts/x86/gcp.sh up [c4d-standard-16] [zone]
#   scripts/x86/gcp.sh ip | ssh [cmd...] | sync | status | down
#
# Root login: GCP images refuse it, so a startup script authorises the bench key (~/.ssh/parakeet-bench) for root,
# which keeps setup.sh and the /root/voxrt paths identical to the Hetzner box.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
: "${VOXRT_GCP_PROJECT:?set VOXRT_GCP_PROJECT}"
NAME=voxrt-x86
ZONE_FILE="$HOME/.cache/voxrt-gcp-zone"
KEY="$HOME/.ssh/parakeet-bench"
G=(gcloud --quiet --project "$VOXRT_GCP_PROJECT")
SSHO=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5)

zone() { cat "$ZONE_FILE" 2>/dev/null || echo europe-west4-a; }
server_ip() {
  "${G[@]}" compute instances list --filter="labels.purpose=parakeet-bench" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)" | head -1
}
rsh() { ssh "${SSHO[@]}" "root@$(server_ip)" "$@"; }

case "${1:-status}" in
  up)
    TYPE="${2:-c4d-standard-16}"; ZONE="${3:-europe-west4-a}"
    [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N '' -C parakeet-bench -f "$KEY"
    [ -n "$(server_ip)" ] && { echo "already up: $(server_ip)"; exit 0; }
    mkdir -p "$(dirname "$ZONE_FILE")"; echo "$ZONE" >"$ZONE_FILE"
    STARTUP=$(mktemp)
    cat >"$STARTUP" <<EOF
#!/bin/bash
mkdir -p /root/.ssh && echo "$(cat "$KEY.pub")" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
systemctl reload ssh || systemctl reload sshd
EOF
    "${G[@]}" compute instances create "$NAME" --zone "$ZONE" --machine-type "$TYPE" \
      --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
      --boot-disk-size 60GB --boot-disk-type hyperdisk-balanced \
      --labels purpose=parakeet-bench --metadata-from-file startup-script="$STARTUP" >/dev/null
    rm -f "$STARTUP"
    for _ in $(seq 60); do rsh true 2>/dev/null && break; sleep 5; done
    echo "up: $(server_ip) | $(rsh "lscpu | sed -n 's/^Model name: *//p'; nproc") | \
$(rsh "grep -o -w 'avx2\|avx512f\|avx512_vnni\|avx_vnni\|avx512_bf16' /proc/cpuinfo | sort -u | tr '\n' ' '")" ;;
  ip) server_ip ;;
  ssh) shift; rsh "$@" ;;
  sync)
    IP=$(server_ip)
    rsync -az --delete -e "ssh ${SSHO[*]}" \
      --exclude target/ --exclude 'target-*/' --exclude data/ --exclude 'models/parakeet-redux-fp/' --exclude 'bench/' \
      --exclude 'results/*.jsonl' --exclude dist/ --exclude '.venv/' --exclude 'bindings/swift/.build/' \
      --exclude 'bindings/swift/CVoxrt.xcframework/' \
      "$R/" "root@$IP:/root/voxrt/"
    rsync -az -e "ssh ${SSHO[*]}" "$R/data/manifests" "root@$IP:/root/voxrt/data/" ;;
  status)
    "${G[@]}" compute instances list --filter="labels.purpose=parakeet-bench" \
      --format="table(name,zone.basename(),machineType.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)" ;;
  down)
    for n in $("${G[@]}" compute instances list --filter="labels.purpose=parakeet-bench" --format="value(name)"); do
      "${G[@]}" compute instances delete "$n" --zone "$(zone)" >/dev/null && echo "deleted $n"
    done ;;
  *) echo "usage: $0 up [type] [zone] | ip | ssh [cmd] | sync | status | down" >&2; exit 2 ;;
esac
