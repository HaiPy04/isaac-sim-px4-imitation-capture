#!/usr/bin/env bash
set -e

# ==============================================================================
# Imitation Learning Episode Recorder (ROS 2 Bag MCAP)
# Records synchronized PX4 uXRCE-DDS state/action telemetry and /joy inputs.
#
# Usage:
#   ./scripts/record_episode.sh              # Auto-names episode with timestamp
#   ./scripts/record_episode.sh hover_test   # Appends custom tag to episode name
# ==============================================================================

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS2_PX4_WS="/home/haipy/ros2_px4_ws"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_ROOT="$REPO_ROOT/data/raw_bags"

# Optional custom tag passed as first CLI argument
TAG="${1:-warehouse_flight}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EPISODE_DIR="$OUTPUT_ROOT/ep_${TIMESTAMP}_${TAG}"

# 1. Source ROS 2 and px4_msgs workspace so MCAP stores full message schemas
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$ROS2_PX4_WS/install/setup.bash"

mkdir -p "$OUTPUT_ROOT"

# 2. Pre-flight check: Ensure core topics are active before starting recording
REQUIRED_TOPICS=(
    "/fmu/out/vehicle_odometry"
    "/fmu/out/manual_control_setpoint"
    "/fmu/out/vehicle_status_v1"
    "/joy"
)

ACTIVE_TOPICS="$(ros2 topic list)"
for topic in "${REQUIRED_TOPICS[@]}"; do
    if ! echo "$ACTIVE_TOPICS" | grep -qx "$topic"; then
        echo "[ERROR] Missing required topic: $topic"
        echo "        Make sure scripts/run_sim_capture.sh is running first."
        exit 1
    fi
done

# 3. Record MCAP bag (includes QoS override profile for PX4 BEST_EFFORT topics)
QOS_FILE="/tmp/px4_bag_qos.yaml"
cat << 'EOF' > "$QOS_FILE"
/fmu/out/vehicle_odometry:
  reliability: best_effort
  durability: transient_local
  history: keep_last
  depth: 10
/fmu/out/manual_control_setpoint:
  reliability: best_effort
  durability: transient_local
  history: keep_last
  depth: 10
/fmu/out/vehicle_status_v1:
  reliability: best_effort
  durability: transient_local
  history: keep_last
  depth: 10
EOF

echo "=============================================================================="
echo " Recording Episode -> $EPISODE_DIR"
echo " Topics:"
echo "   - /fmu/out/vehicle_odometry         (State s_t)"
echo "   - /fmu/out/manual_control_setpoint  (PX4 Normalized Action a_t)"
echo "   - /fmu/out/vehicle_status_v1        (Arming / Navigation State)"
echo "   - /joy                              (Raw 50 Hz Controller + Arm Button)"
echo " Press Ctrl+C when the flight demonstration is complete to finalize the bag."
echo "=============================================================================="

ros2 bag record \
    --storage mcap \
    --qos-profile-overrides-path "$QOS_FILE" \
    --output "$EPISODE_DIR" \
    /fmu/out/vehicle_odometry \
    /fmu/out/manual_control_setpoint \
    /fmu/out/vehicle_status_v1 \
    /joy
