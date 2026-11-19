#!/usr/bin/env python3
"""
Standalone Isaac Sim 5.1 + Pegasus Simulator Launch Script
Spawns a PX4-enabled quadrotor in the default Warehouse environment
for Imitation Learning teleoperation data capture.
"""


from isaacsim import SimulationApp

# 1. Start the Omniverse / Isaac Sim application BEFORE any omni or pegasus imports
simulation_app = SimulationApp({"headless": False})

# ------------------------------------------------------------------
# All Omniverse, Isaac Sim, and Pegasus imports MUST happen below
# ------------------------------------------------------------------
import carb
import omni.timeline
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class WarehouseDroneApp:
    def __init__(self):
        # Acquire the Omniverse timeline to control play/stop state
        self.timeline = omni.timeline.get_timeline_interface()

        # 2. Initialize the Pegasus Interface & Isaac Sim World
        self.pg = PegasusInterface()
        self.pg._world = None
        self.pg.initialize_world()
        self.world = self.pg.world

        # 3. Load the built-in Pegasus "Warehouse" environment
        # Other built-in options: "Warehouse with Forklifts", "Warehouse with Shelves", "Full Warehouse"
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse"])

        # 4. Configure the PX4 MAVLink Backend (SITL)
        px4_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,           # Automatically spawns PX4 SITL
                "px4_dir": "/home/haipy/PX4-Autopilot",      # Uses your configured PX4-Autopilot path
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )

        # 5. Configure and Spawn the Multirotor Drone
        drone_config = MultirotorConfig()
        drone_config.backends = [PX4MavlinkBackend(px4_config)]

        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            vehicle_id=0,
            init_pos=[0.0, 0.0, 0.07],  # Spawn slightly above floor to prevent ground clipping
            init_orientation=Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=drone_config,
        )

        # 6. Reset the physics world so all rigid bodies and sensors initialize cleanly
        self.world.reset()
        self.stop_sim = False

    def run(self):
        """Main real-time simulation loop."""
        self.timeline.play()

        while simulation_app.is_running() and not self.stop_sim:
            # Step physics and render viewport in sync
            self.world.step(render=True)

        # Graceful shutdown
        carb.log_warn("WarehouseDroneApp is shutting down.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = WarehouseDroneApp()
    app.run()


if __name__ == "__main__":
    main()
