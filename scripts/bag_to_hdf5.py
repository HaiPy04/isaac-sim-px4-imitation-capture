#!/usr/bin/env python3
"""
Offline MCAP ROS 2 Bag to HDF5 Dataset Converter
Imitation Learning Pipeline for Aerial Robotics (BC / ACT / Diffusion Policy)

Reads raw MCAP bag files recorded by scripts/record_episode.sh containing:
  - /fmu/out/vehicle_odometry        (State observation s_t)
  - /fmu/out/manual_control_setpoint (Action a_t)
  - /fmu/out/vehicle_status_v1       (Arming / flight mode metadata)
  - /joy                             (Raw teleoperation joystick telemetry)

Synchronizes telemetry to a fixed rate (default 50 Hz), filters idle ground
frames using arming state / button transitions, and writes structured HDF5 files
into data/processed_hdf5/.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np

# Verify dependencies
try:
    import h5py
except ImportError:
    h5py = None

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    rosbag2_py = None


def get_rosbag_options(bag_path: str, storage_id: str = "mcap"):
    """Configure rosbag2 storage options."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    return storage_options, converter_options


def read_messages_from_bag(bag_path: str):
    """
    Reads all messages from an MCAP bag directory using rosbag2_py.
    Returns a dictionary of sorted message lists grouped by topic.
    """
    if rosbag2_py is None:
        raise RuntimeError(
            "rosbag2_py is required to parse ROS 2 MCAP bags. "
            "Please ensure ROS 2 Humble is installed and sourced:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /home/haipy/ros2_px4_ws/install/setup.bash"
        )

    storage_options, converter_options = get_rosbag_options(bag_path, "mcap")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Cache topic message types
    topic_types = {}
    for topic_meta in reader.get_all_topics_and_types():
        topic_types[topic_meta.name] = topic_meta.type

    records = {
        "/fmu/out/vehicle_odometry": [],
        "/fmu/out/manual_control_setpoint": [],
        "/fmu/out/vehicle_status_v1": [],
        "/joy": [],
    }

    # Deserialization type classes
    type_classes = {}
    for topic, type_name in topic_types.items():
        if topic in records:
            try:
                type_classes[topic] = get_message(type_name)
            except Exception as e:
                print(f"[WARN] Could not load message class for {type_name}: {e}")

    print(f"[INFO] Reading messages from bag: {bag_path}")
    count = 0
    while reader.has_next():
        topic_name, serialized_msg, timestamp_ns = reader.read_next()
        if topic_name in records and topic_name in type_classes:
            msg_cls = type_classes[topic_name]
            msg = deserialize_message(serialized_msg, msg_cls)
            records[topic_name].append((timestamp_ns, msg))
            count += 1

    print(f"[INFO] Read {count} total matching messages.")
    for t, msgs in records.items():
        print(f"       - {t}: {len(msgs)} msgs")

    return records


def process_telemetry(records: dict, target_hz: float = 50.0):
    """
    Interpolates and aligns odometry (~100 Hz), actions (~50 Hz),
    and joystick telemetry to a synchronized fixed time-base.
    """
    odom_data = records.get("/fmu/out/vehicle_odometry", [])
    act_data = records.get("/fmu/out/manual_control_setpoint", [])
    joy_data = records.get("/joy", [])
    status_data = records.get("/fmu/out/vehicle_status_v1", [])

    if not odom_data or not act_data:
        print("[WARN] Insufficient odometry or action data to process episode.")
        return None

    # Compute overlapping time window
    t_start = max(odom_data[0][0], act_data[0][0])
    t_end = min(odom_data[-1][0], act_data[-1][0])

    if t_start >= t_end:
        print("[WARN] Invalid timestamps: start time >= end time.")
        return None

    dt_ns = int(1e9 / target_hz)
    sync_timestamps_ns = np.arange(t_start, t_end, dt_ns, dtype=np.int64)
    sync_timestamps_sec = (sync_timestamps_ns - t_start) * 1e-9
    num_steps = len(sync_timestamps_ns)

    print(f"[INFO] Synchronizing {num_steps} steps at {target_hz} Hz ({sync_timestamps_sec[-1]:.2f}s duration)...")

    # Extract Odometry arrays
    odom_times = np.array([t for t, _ in odom_data], dtype=np.int64)
    # PX4 VehicleOdometry coordinates (NED to body/world conversions)
    pos_raw = np.array([[m.position[0], m.position[1], m.position[2]] for _, m in odom_data], dtype=np.float32)
    q_raw = np.array([[m.q[0], m.q[1], m.q[2], m.q[3]] for _, m in odom_data], dtype=np.float32)  # [w, x, y, z]
    vel_raw = np.array([[m.velocity[0], m.velocity[1], m.velocity[2]] for _, m in odom_data], dtype=np.float32)
    angvel_raw = np.array([[m.angular_velocity[0], m.angular_velocity[1], m.angular_velocity[2]] for _, m in odom_data], dtype=np.float32)

    # Extract Actions (ManualControlSetpoint: roll, pitch, yaw, throttle)
    act_times = np.array([t for t, _ in act_data], dtype=np.int64)
    actions_raw = np.array([[m.roll, m.pitch, m.yaw, m.throttle] for _, m in act_data], dtype=np.float32)

    # 1D Linear interpolation over sync_timestamps_ns
    sync_pos = np.empty((num_steps, 3), dtype=np.float32)
    sync_vel = np.empty((num_steps, 3), dtype=np.float32)
    sync_angvel = np.empty((num_steps, 3), dtype=np.float32)
    sync_quat = np.empty((num_steps, 4), dtype=np.float32)
    sync_actions = np.empty((num_steps, 4), dtype=np.float32)

    for dim in range(3):
        sync_pos[:, dim] = np.interp(sync_timestamps_ns, odom_times, pos_raw[:, dim])
        sync_vel[:, dim] = np.interp(sync_timestamps_ns, odom_times, vel_raw[:, dim])
        sync_angvel[:, dim] = np.interp(sync_timestamps_ns, odom_times, angvel_raw[:, dim])

    for dim in range(4):
        sync_quat[:, dim] = np.interp(sync_timestamps_ns, odom_times, q_raw[:, dim])
        sync_actions[:, dim] = np.interp(sync_timestamps_ns, act_times, actions_raw[:, dim])

    # Normalize interpolated quaternions
    norms = np.linalg.norm(sync_quat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sync_quat = sync_quat / norms

    # Extract Joy axes & buttons (nearest neighbor)
    sync_joy_axes = np.zeros((num_steps, 8), dtype=np.float32)
    sync_joy_buttons = np.zeros((num_steps, 11), dtype=np.int32)
    if joy_data:
        joy_times = np.array([t for t, _ in joy_data], dtype=np.int64)
        joy_axes_raw = np.array([list(m.axes) + [0.0] * max(0, 8 - len(m.axes)) for _, m in joy_data], dtype=np.float32)[:, :8]
        joy_btn_raw = np.array([list(m.buttons) + [0] * max(0, 11 - len(m.buttons)) for _, m in joy_data], dtype=np.int32)[:, :11]

        # Nearest neighbor lookup
        indices = np.searchsorted(joy_times, sync_timestamps_ns).clip(0, len(joy_times) - 1)
        sync_joy_axes = joy_axes_raw[indices]
        sync_joy_buttons = joy_btn_raw[indices]

    # Extract flight status (arming state)
    sync_arming = np.zeros((num_steps,), dtype=np.uint8)
    if status_data:
        st_times = np.array([t for t, _ in status_data], dtype=np.int64)
        st_arming_raw = np.array([getattr(m, "arming_state", 1) for _, m in status_data], dtype=np.uint8)
        indices = np.searchsorted(st_times, sync_timestamps_ns).clip(0, len(st_times) - 1)
        sync_arming = st_arming_raw[indices]

    # Optional: Segment active flight episode (where armed == 2 or arm button was triggered)
    # Keep full trajectory or trim idle landing frames
    flight_mask = np.ones(num_steps, dtype=bool)
    if np.any(sync_arming == 2):
        # 2 is ARMING_STATE_ARMED in PX4
        first_armed = np.argmax(sync_arming == 2)
        last_armed = num_steps - 1 - np.argmax(sync_arming[::-1] == 2)
        # Add 1-second buffer before arm and after disarm
        pad = int(1.0 * target_hz)
        start_idx = max(0, first_armed - pad)
        end_idx = min(num_steps, last_armed + pad + 1)
        flight_mask = np.zeros(num_steps, dtype=bool)
        flight_mask[start_idx:end_idx] = True

    return {
        "timestamps": sync_timestamps_sec[flight_mask],
        "timestamps_ns": sync_timestamps_ns[flight_mask],
        "observations/position": sync_pos[flight_mask],
        "observations/orientation_wxyz": sync_quat[flight_mask],
        "observations/linear_velocity": sync_vel[flight_mask],
        "observations/angular_velocity": sync_angvel[flight_mask],
        "actions/manual_control": sync_actions[flight_mask],
        "actions/joy_axes": sync_joy_axes[flight_mask],
        "actions/joy_buttons": sync_joy_buttons[flight_mask],
        "flight_status/arming_state": sync_arming[flight_mask],
    }


def save_to_hdf5(dataset: dict, output_path: str):
    """Writes processed demonstration dictionary into standard HDF5 format."""
    if h5py is None:
        raise RuntimeError("h5py is required to export datasets. Install via: pip install h5py")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Create dataset groups
        obs_grp = f.create_group("observations")
        act_grp = f.create_group("actions")
        status_grp = f.create_group("flight_status")

        # Observations
        obs_grp.create_dataset("position", data=dataset["observations/position"], compression="gzip")
        obs_grp.create_dataset("orientation_wxyz", data=dataset["observations/orientation_wxyz"], compression="gzip")
        obs_grp.create_dataset("linear_velocity", data=dataset["observations/linear_velocity"], compression="gzip")
        obs_grp.create_dataset("angular_velocity", data=dataset["observations/angular_velocity"], compression="gzip")

        # Actions
        act_grp.create_dataset("manual_control", data=dataset["actions/manual_control"], compression="gzip")
        act_grp.create_dataset("joy_axes", data=dataset["actions/joy_axes"], compression="gzip")
        act_grp.create_dataset("joy_buttons", data=dataset["actions/joy_buttons"], compression="gzip")

        # Flight Status
        status_grp.create_dataset("arming_state", data=dataset["flight_status/arming_state"], compression="gzip")

        # Global timestamps
        f.create_dataset("timestamps", data=dataset["timestamps"], compression="gzip")
        f.create_dataset("timestamps_ns", data=dataset["timestamps_ns"], compression="gzip")

        # Metadata attributes
        f.attrs["num_samples"] = len(dataset["timestamps"])
        f.attrs["duration_sec"] = float(dataset["timestamps"][-1] - dataset["timestamps"][0]) if len(dataset["timestamps"]) > 0 else 0.0
        f.attrs["target_frequency_hz"] = 50.0

    print(f"[SUCCESS] Exported HDF5 dataset -> {output_path} ({len(dataset['timestamps'])} steps)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw ROS 2 MCAP bags to synchronized HDF5 imitation learning datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_bags"),
        help="Path to directory containing recorded raw bag directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_hdf5"),
        help="Path to directory where processed HDF5 files will be written.",
    )
    parser.add_argument(
        "--bag_path",
        type=str,
        default=None,
        help="Optional single bag directory or MCAP file to process directly.",
    )
    parser.add_argument(
        "--target_hz",
        type=float,
        default=50.0,
        help="Target resampling and synchronization rate in Hz (default: 50.0).",
    )
    args = parser.parse_args()

    # Determine bags to convert
    bags_to_process = []
    if args.bag_path:
        bags_to_process.append(args.bag_path)
    else:
        if os.path.exists(args.input_dir):
            for entry in sorted(os.listdir(args.input_dir)):
                full_path = os.path.join(args.input_dir, entry)
                if os.path.isdir(full_path) or full_path.endswith(".mcap"):
                    bags_to_process.append(full_path)

    if not bags_to_process:
        print(f"[INFO] No bag files found in {args.input_dir}. Nothing to convert.")
        print(f"       Run a demonstration flight first using: ./scripts/record_episode.sh")
        return

    print(f"[INFO] Found {len(bags_to_process)} bag(s) to convert.")
    for bag in bags_to_process:
        bag_name = os.path.basename(os.path.normpath(bag))
        hdf5_name = f"{bag_name}.h5"
        output_file = os.path.join(args.output_dir, hdf5_name)

        try:
            records = read_messages_from_bag(bag)
            dataset = process_telemetry(records, target_hz=args.target_hz)
            if dataset is not None and len(dataset["timestamps"]) > 0:
                save_to_hdf5(dataset, output_file)
            else:
                print(f"[WARN] Skipping {bag}: empty or unaligned dataset.")
        except Exception as e:
            print(f"[ERROR] Failed to process {bag}: {e}")


if __name__ == "__main__":
    main()
