#!/usr/bin/env bash
# Record a balanced set of scripted episodes in small parts, so that a crash or a shutdown never costs more
# than the part in progress. Run it again to continue: finished parts are skipped.
#
#   04_recording/record_scripted_parts.sh [OUT_DIR]          (default datasets/parts_v2)
#
#   EPISODES=200 PART=5     how many episodes, and how many per part (a part = one simulator run = one box size)
#   HUB_REPO=org/name       optional: upload finished parts to this Hub dataset under parts/, every UPLOAD_EVERY
#   UPLOAD_EVERY=2          parts (default 2 = every 10 episodes). Needs `hf auth login` on this PC.
#   DOMAIN=6 PORT_OFFSET=2000   DDS domain and image-port offset of the private simulator this script starts
#   G1_ENV=g1sim            conda env of the simulator (the recorder always uses g1lab)
#
# Files in OUT_DIR:
#   schedule.json    which strata every episode must have (python -m g1lab.scripted.schedule), written once
#   part_NN/         a LeRobot dataset with PART episodes; part_NN/DONE marks it complete, UPLOADED = on the Hub
#   attempts.jsonl   one line per attempt: scene, verdict, measurements (coverage: g1lab.scripted.schedule --coverage)
# Afterwards: 05_training/merge_datasets.py --out datasets/<name> OUT_DIR/part_*  joins the parts.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/datasets/parts_v2}"
EPISODES="${EPISODES:-200}" PART="${PART:-5}" UPLOAD_EVERY="${UPLOAD_EVERY:-2}"
DOMAIN="${DOMAIN:-6}" PORT_OFFSET="${PORT_OFFSET:-2000}" HUB_REPO="${HUB_REPO:-}"
mkdir -p "$OUT"
SCHED="$OUT/schedule.json"
lab() { ( source "$REPO/scripts/env.sh" g1lab >/dev/null 2>&1; cd "$REPO"; "$@" ); }

[ -f "$SCHED" ] || lab python -m g1lab.scripted.schedule --episodes "$EPISODES" --part "$PART" --out "$SCHED"
N=$(python3 -c "import json; print(len(json.load(open('$SCHED'))['parts']))")

upload() {  # every finished part that is not on the Hub yet, once enough have piled up (or at the very end)
  [ -n "$HUB_REPO" ] || return 0
  local waiting=() p
  for p in "$OUT"/part_*/; do
    [ -f "$p/DONE" ] && [ ! -f "$p/UPLOADED" ] && waiting+=("${p%/}")
  done
  [ "${#waiting[@]}" -ge "$1" ] || return 0
  for p in "${waiting[@]}"; do
    if lab hf upload "$HUB_REPO" "$p" "parts/$(basename "$p")" --repo-type dataset --exclude DONE --exclude UPLOADED \
         --commit-message "$(basename "$p"): $PART scripted episodes" >/dev/null 2>&1; then
      touch "$p/UPLOADED"
      echo "[rec] uploaded $(basename "$p") to $HUB_REPO"
    else
      echo "[rec] upload of $(basename "$p") failed (no internet?) - will try again after the next part"
      return 0
    fi
  done
}

for i in $(seq 0 $((N - 1))); do
  NAME=$(printf "part_%02d" "$i")
  P="$OUT/$NAME"
  [ -f "$P/DONE" ] && continue
  rm -rf "$P"
  python3 - "$SCHED" "$i" "$OUT/$NAME.scenes.json" <<'PY'
import json, sys
part = json.load(open(sys.argv[1]))["parts"][int(sys.argv[2])]
json.dump(part["scenes"], open(sys.argv[3], "w"))
print(*part["scale"], part["sim_seed"], len(part["scenes"]), file=open(sys.argv[3] + ".args", "w"))
PY
  read -r SX SY SZ SIMSEED COUNT < "$OUT/$NAME.scenes.json.args"
  ( source "$REPO/scripts/env.sh" ""; exec python -u "$REPO/01_teleop/g1sim/g1_xr_sim.py" --scene lab --randomize \
      --seed "$SIMSEED" --no-viewer --camera-shadows none --domain "$DOMAIN" --port-offset "$PORT_OFFSET" \
      --wrap-scale "$SX" "$SY" "$SZ" ) > "$OUT/sim.log" 2>&1 &
  SIM=$!
  sleep 12
  lab nice -n 10 python -u -m g1lab.scripted.box_into_box --scenes "$OUT/$NAME.scenes.json" --root "$P" \
      --repo-id local/sim_scripted_v2 --seed "$((100 + i))" --domain "$DOMAIN" --image-port "$((60000 + PORT_OFFSET))" \
      --log "$OUT/attempts.jsonl" --tag "$NAME" \
    && python3 -c "import json, sys; sys.exit(0 if json.load(open('$P/meta/info.json'))['total_episodes'] == $COUNT else 1)" \
    && touch "$P/DONE"
  kill "$SIM" 2>/dev/null; wait "$SIM" 2>/dev/null
  DONE_PARTS=$(ls "$OUT"/part_*/DONE 2>/dev/null | wc -l)
  echo "[rec] $NAME $([ -f "$P/DONE" ] && echo finished || echo NOT finished): $DONE_PARTS/$N parts done at $(date +%H:%M)"
  upload "$UPLOAD_EVERY"
done
upload 1
[ "$(ls "$OUT"/part_*/DONE 2>/dev/null | wc -l)" -eq "$N" ] && echo "[rec] all $N parts are done"
