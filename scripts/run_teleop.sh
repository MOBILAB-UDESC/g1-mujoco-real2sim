#!/usr/bin/env bash
# Start xr_teleoperate for the MuJoCo sim or the real G1, with the wrist camera views added to the headset
# display (02_wrist_cameras/teleop_views.py) and this repo's teleop script (01_teleop/teleop_hand_and_arm.py:
# Unitree's script + Quest buttons, pause, waist stick).
#   scripts/run_teleop.sh               sim, hand tracking, arms only (like debug mode on the real robot)
#   scripts/run_teleop.sh --controller  sim, Quest controllers: arms follow the controllers, left stick walks,
#                                       right stick twists the waist (--input-mode controller --motion --waist-stick)
#   scripts/run_teleop.sh --real [--controller] [--iface enp3s0]
#                                       the REAL robot (asks for "yes"): DDS domain 0 on --iface (or G1_IFACE),
#                                       images from G1_IMG_SERVER (default 192.168.123.164).
#                                       Without --controller: debug mode (robot on the gantry; the script enters debug mode).
#                                       With --controller: Unitree's controller keeps walking and balancing.
#   scripts/run_teleop.sh --plain       Unitree's unmodified teleop_hand_and_arm.py (no wrist views, sim only)
#   --wrist-layout pip|side|off         wrist views at start (default pip; toggle in VR: click the left stick /
#                                       the "WRIST VIEW" button, or the button at the bottom-left of the page)
# Quest buttons (controller mode): left X start/resume, right B pause, left Y record start/save, right A quit.
# Any other args go to teleop_hand_and_arm.py, e.g.:  scripts/run_teleop.sh --controller --record
# Env: G1_ENV conda env (default g1sim)
set -e
# "" so env.sh doesn't take our first argument for the env name.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" ""
IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src") print $(i+1)}')"
IP="${IP:-127.0.0.1}"
PLAIN=0
REAL=0
CONTROLLER=0
IFACE="${G1_IFACE:-}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --controller) CONTROLLER=1 ;;
    --plain) PLAIN=1 ;;
    --real) REAL=1 ;;
    --iface) IFACE="$2"; shift ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
MODE_ARGS=()
if [ "$CONTROLLER" = 1 ]; then
  MODE_ARGS=(--input-mode controller --motion)
  [ "$PLAIN" = 1 ] || MODE_ARGS+=(--waist-stick)   # Unitree's own script has no such flag
fi
if [ "$REAL" = 1 ]; then
  [ "$PLAIN" = 0 ] || { echo "--plain is for the sim" >&2; exit 1; }
  FIXED=(--arm G1_29 --img-server-ip "${G1_IMG_SERVER:-192.168.123.164}")
  [ -z "$IFACE" ] || FIXED+=(--network-interface "$IFACE")
  echo "REAL ROBOT checklist:"
  echo "  [ ] area around the robot clear, e-stop in reach of a second person"
  echo "  [ ] image server running on the robot's PC2 (head + wrist cameras)"
  echo "  [ ] this PC is on the robot's network${IFACE:+ through $IFACE} (ping 192.168.123.164)"
  if [ "$CONTROLLER" = 1 ]; then
    echo "  [ ] robot standing in Unitree's regular walking mode (R1+X): its controller balances the legs;"
    echo "      left stick walks, right stick twists the waist, both sticks pressed = damping (robot goes limp)"
  else
    echo "  [ ] robot hanging on the gantry / stand: debug mode holds the legs stiff, it does NOT balance"
  fi
  read -r -p "Type 'yes' to teleoperate the REAL robot: " ok
  [ "$ok" = yes ] || { echo aborted; exit 1; }
else
  FIXED=(--arm G1_29 --sim --network-interface lo --img-server-ip 127.0.0.1)
fi
python "$REPO/scripts/quest_link.py" "$IP" 8080 &
LINK_PID=$!
trap 'kill $LINK_PID 2>/dev/null' EXIT
echo "On the Quest 3 browser open:  http://${IP}:8080"
echo "  (redirects to https://${IP}:8012/?ws=wss://${IP}:8012)"
if [ "$PLAIN" = 1 ]; then
  cd "$G1_THIRD_PARTY/xr_teleoperate/teleop"
  python teleop_hand_and_arm.py "${MODE_ARGS[@]}" "${FIXED[@]}" "${ARGS[@]}"
else
  python "$REPO/02_wrist_cameras/teleop_views.py" --wrist-layout pip "${MODE_ARGS[@]}" "${FIXED[@]}" "${ARGS[@]}"
fi
