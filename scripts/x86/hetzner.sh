#!/usr/bin/env bash
# x86 test box on Hetzner Cloud (dedicated vCPU CCX line), for the x86 kernel work (PLAN 3c.9).
# Only ever touches servers carrying the label purpose=parakeet-bench.
#
#   HCLOUD_TOKEN_FILE=path/to/token scripts/x86/hetzner.sh up [ccx33] [vnni]
#   scripts/x86/hetzner.sh ip | ssh [cmd...] | sync | status | down
#
# `up ... vnni` recreates the server until the CPU reports avx512_vnni (Genoa), max 6 tries.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
NAME=voxrt-x86
LABEL=purpose=parakeet-bench
KEY="$HOME/.ssh/parakeet-bench"
API=https://api.hetzner.cloud/v1
: "${HCLOUD_TOKEN_FILE:?set HCLOUD_TOKEN_FILE}"
TOKEN=$(tr -d '[:space:]' <"$HCLOUD_TOKEN_FILE")

api() { # method path [json]
  curl -sf -X "$1" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    ${3:+-d "$3"} "$API$2"
}
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
server_id() { api GET "/servers?label_selector=$LABEL" | json "' '.join(str(s['id']) for s in d['servers'])"; }
server_ip() { api GET "/servers?label_selector=$LABEL" | json "d['servers'][0]['public_net']['ipv4']['ip'] if d['servers'] else ''"; }
SSHO=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5)
rsh() { ssh "${SSHO[@]}" "root@$(server_ip)" "$@"; }

ensure_key() {
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N '' -C parakeet-bench -f "$KEY"
  if [ -z "$(api GET "/ssh_keys?name=parakeet-bench" | json "' '.join(str(k['id']) for k in d['ssh_keys'])")" ]; then
    api POST /ssh_keys "{\"name\":\"parakeet-bench\",\"public_key\":\"$(cat "$KEY.pub")\",\"labels\":{\"purpose\":\"parakeet-bench\"}}" >/dev/null
  fi
}

create() { # type location
  api POST /servers "{\"name\":\"$NAME\",\"server_type\":\"$1\",\"location\":\"$2\",\"image\":\"ubuntu-24.04\",
    \"ssh_keys\":[\"parakeet-bench\"],\"labels\":{\"purpose\":\"parakeet-bench\"}}" >/dev/null
  for _ in $(seq 60); do rsh true 2>/dev/null && return 0; sleep 5; done
  echo "ssh did not come up" >&2; return 1
}

down() {
  for id in $(server_id); do api DELETE "/servers/$id" >/dev/null; echo "deleted server $id"; done
  for _ in $(seq 30); do [ -z "$(server_id)" ] && return 0; sleep 2; done
}

case "${1:-status}" in
  up)
    TYPE="${2:-ccx33}"; WANT="${3:-}"
    ensure_key
    [ -n "$(server_id)" ] && { echo "already up: $(server_ip)"; exit 0; }
    read -r -a LOCS <<<"${VOXRT_LOCS:-fsn1 nbg1 hel1 fsn1 nbg1 hel1}"
    for i in "${!LOCS[@]}"; do
      create "$TYPE" "${LOCS[$i]}"
      CPU=$(rsh "lscpu | sed -n 's/^Model name: *//p'")
      FLAGS=$(rsh "grep -o -w 'avx2\|avx512f\|avx512_vnni\|avx512_bf16\|amx_int8' /proc/cpuinfo | sort -u | tr '\n' ' '")
      echo "try $((i + 1)) ${LOCS[$i]}: $CPU | $FLAGS"
      if [ "$WANT" != vnni ] || [[ " $FLAGS" == *" avx512_vnni "* ]]; then echo "up: $(server_ip)"; exit 0; fi
      down
    done
    echo "no VNNI host after ${#LOCS[@]} tries" >&2; exit 1 ;;
  probe) # probe TYPE [LOC]: create a throwaway server, print its CPU, delete it (separate label, leaves the box alone)
    ensure_key
    ID=$(api POST /servers "{\"name\":\"voxrt-probe-$2\",\"server_type\":\"$2\",\"location\":\"${3:-fsn1}\",
      \"image\":\"ubuntu-24.04\",\"ssh_keys\":[\"parakeet-bench\"],\"labels\":{\"purpose\":\"parakeet-probe\"}}" |
      json "d['server']['id']")
    trap 'api DELETE "/servers/$ID" >/dev/null' EXIT
    PIP=$(api GET "/servers/$ID" | json "d['server']['public_net']['ipv4']['ip']")
    for _ in $(seq 60); do ssh "${SSHO[@]}" "root@$PIP" true 2>/dev/null && break; sleep 5; done
    ssh "${SSHO[@]}" "root@$PIP" "echo \"$2 ${3:-fsn1}: \$(lscpu | sed -n 's/^Model name: *//p') | \$(grep -o -w 'avx2\\|avx512f\\|avx512_vnni\\|avx_vnni\\|avx512_bf16\\|amx_int8' /proc/cpuinfo | sort -u | tr '\\n' ' ')\"" ;;
  ip) server_ip ;;
  ssh) shift; rsh "$@" ;;
  sync) # code + model + golden; data is fetched on the box (scripts/x86/setup.sh)
    IP=$(server_ip)
    rsync -az --delete -e "ssh ${SSHO[*]}" \
      --exclude target/ --exclude data/ --exclude 'models/parakeet-redux-fp/' --exclude 'bench/' \
      --exclude 'results/*.jsonl' --exclude dist/ --exclude '.venv/' --exclude 'bindings/swift/.build/' \
      --exclude 'bindings/swift/CVoxrt.xcframework/' \
      "$R/" "root@$IP:/root/voxrt/"
    rsync -az -e "ssh ${SSHO[*]}" "$R/data/manifests" "root@$IP:/root/voxrt/data/" ;;
  status) api GET "/servers?label_selector=$LABEL" | json "'\n'.join(f\"{s['id']} {s['name']} {s['server_type']['name']} {s['status']} {s['public_net']['ipv4']['ip']}\" for s in d['servers']) or 'no server'" ;;
  down) down ;;
  *) echo "usage: $0 up [type] [vnni] | ip | ssh [cmd] | sync | status | down" >&2; exit 2 ;;
esac
