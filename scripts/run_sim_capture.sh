#!/usr/bin/env bash
set -e

# ==============================================================================
# Master Simulation & Teleoperation Capture Orchestrator
# Launches:
#   1. Isaac Sim 5.1 + Pegasus + PX4 SITL (inside Conda: env_isaaclab)
#   2. QGroundControl (host)
#   3. ROS 2 Humble + px4_msgs workspace (host)
#   4. MicroXRCEAgent UDP bridge (port 8888)
#   5. ROS 2 joy_node at 50 Hz with 5% deadzone
# ==============================================================================

# --- Exact User Paths ---
CONDA_ENV="env_isaaclab"
ISAAC_SCRIPT="/home/haipy/Desktop/isaac-sim-px4-imitation-capture/scripts/pegasusLaunch.py"
QGC_APPIMAGE="/home/haipy/Desktop/PEGASUS_ISAACSIM/QGroundControl-x86_64.AppImage"
ROS2_PX4_WS="/home/haipy/ros2_px4_ws"
ROS_DISTRO="${ROS_DISTRO:-humble}"

# --- Cleanup Trap: Terminates all background processes cleanly on exit ---
cleanup() {
    echo -e "\n[SHUTDOWN] Cleaning up simulation, QGC, DDS agent, and ROS 2 nodes..."
    kill "$JOY_PID" "$XRCE_PID" "$QGC_PID" 2>/dev/null || true
    pkill -9 -x px4 2>/dev/null || true
    pkill -9 -f MicroXRCEAgent 2>/dev/null || true
    pkill -9 -f joy_node 2>/dev/null || true
    echo "[SHUTDOWN] All processes stopped."
}
trap cleanup EXIT INT TERM

# Pre-flight: Ensure no zombie PX4 instance 0 blocks startup
pkill -9 -x px4 2>/dev/null || true

# 1. Launch Isaac Sim + Pegasus + PX4 inside Conda (isolated from host ROS 2)
echo "[1/5] Activating Conda ($CONDA_ENV) and launching Isaac Sim + Pegasus + PX4..."
(
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    python "$ISAAC_SCRIPT"
) &
ISAAC_PID=$!

# 2. Launch QGroundControl
echo "[2/5] Launching QGroundControl..."
chmod +x "$QGC_APPIMAGE"
"$QGC_APPIMAGE" >/tmp/qgc.log 2>&1 &
QGC_PID=$!

# 3. Source ROS 2 system installation + px4_msgs workspace in the host shell
echo "[3/5] Sourcing ROS 2 ($ROS_DISTRO) and $ROS2_PX4_WS..."
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$ROS2_PX4_WS/install/setup.bash"

# 4. Launch MicroXRCE-DDS Agent (bridges PX4 uORB <-> ROS 2 topics on UDP 8888)
echo "[4/5] Starting MicroXRCEAgent on UDP port 8888..."
MicroXRCEAgent udp4 -p 8888 >/tmp/uxrce_agent.log 2>&1 &
XRCE_PID=$!

# 5. Launch ROS 2 Joystick Node (50 Hz continuous publishing, 5% deadzone)
echo "[5/5] Starting ROS 2 joy_node (/joy at 50 Hz)..."
ros2 run joy joy_node --ros-args -p deadzone:=0.05 -p autorepeat_rate:=50.0 >/tmp/joy_node.log 2>&1 &
JOY_PID=$!

echo "=============================================================================="
echo " Full Stack Running! Logs available in /tmp/{qgc,uxrce_agent,joy_node}.log"
echo " Press Ctrl+C or close the Isaac Sim window to shut everything down cleanly."
echo "=============================================================================="

# Keep script alive until Isaac Sim exits
wait "$ISAAC_PID"
