#!/usr/bin/env bash
# Start Unitree xr_teleoperate against the MuJoCo sim (DDS domain 1 on loopback, images from localhost).
#   scripts/run_teleop.sh               hand tracking, arms only (like debug mode on the real robot)
#   scripts/run_teleop.sh --controller  Quest controllers: arms follow the controllers, left stick walks,
#                                       right stick twists the hip (xr_teleoperate --input-mode controller --motion)
# Any other args go to teleop_hand_and_arm.py, e.g.:  scripts/run_teleop.sh --controller --record
# Env: G1_ENV conda env (default g1sim; the lab PC uses tv:  G1_ENV=tv scripts/run_teleop.sh ...)
set -e
# "" so env.sh doesn't take our first argument for the env name.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" ""
IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src") print $(i+1)}')"
IP="${IP:-127.0.0.1}"
python "$REPO/scripts/quest_link.py" "$IP" 8080 &
LINK_PID=$!
trap 'kill $LINK_PID 2>/dev/null' EXIT
echo "On the Quest 3 browser open:  http://${IP}:8080"
echo "  (redirects to https://${IP}:8012/?ws=wss://${IP}:8012)"
MODE_ARGS=()
ARGS=()
for a in "$@"; do
  if [ "$a" = "--controller" ]; then MODE_ARGS=(--input-mode controller --motion); else ARGS+=("$a"); fi
done
cd "$G1_THIRD_PARTY/xr_teleoperate/teleop"
python teleop_hand_and_arm.py "${MODE_ARGS[@]}" --arm G1_29 --sim --network-interface lo --img-server-ip 127.0.0.1 "${ARGS[@]}"
