import sys
import os
import time
import signal
import threading
import numpy as np

# --- QLabs & Hardware Imports ---
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.qcar_flooring import QLabsQCarFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight

# --- Hardware & Math Imports ---
from pal.products.qcar import QCar, QCarGPS
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap

# ===========================
# 1. CONFIGURATION
# ===========================
# This sequence loops the outer track (adjust as needed for specific map nodes)
NODE_SEQUENCE = [10, 1, 13, 19, 20, 22, 9, 7, 5, 3, 1, 8, 10]
V_REF = 1.0  # Locked cruise speed
CONTROLLER_RATE = 100  # 100Hz loop
START_DELAY = 2.0  # Time for EKF to stabilize before moving

# Initial Pose (Matches Setup_Real_Scenario default)
INITIAL_POS = [-1.205, -0.83, 0.005]
INITIAL_ROT = [0, 0, -44.7]

KILL_PROGRAM = False


def sig_handler(*args):
    global KILL_PROGRAM
    KILL_PROGRAM = True


signal.signal(signal.SIGINT, sig_handler)

# ===========================
# 2. CONTROLLERS (From main.py)
# ===========================


class SpeedController:
    """Locked speed controller with anti-surge logic."""

    def __init__(self, kp=0.04, ki=0.15):
        self.maxThrottle = 1.0
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        self.ei = np.clip(self.ei, -0.2, 0.2)
        return np.clip(self.kp * e + self.ki * self.ei, 0.0, self.maxThrottle)


class SteeringController:
    """Dampened Stanley Controller."""

    def __init__(self, waypoints, k=0.4, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    def update(self, p, th, speed):
        calc_speed = max(speed, 0.2)

        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
        v_seg = wp_2 - wp_1
        v_mag = np.linalg.norm(v_seg)

        v_uv = v_seg / v_mag if v_mag > 0 else np.array([1, 0])
        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)

        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        side_dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        ect = np.linalg.norm(ct) * np.sign(side_dir)
        psi = wrap_to_pi(tangent - th)

        steering = psi + np.arctan2(self.k * ect, calc_speed)
        return np.clip(
            wrap_to_pi(steering), -self.maxSteeringAngle, self.maxSteeringAngle
        )


# ===========================
# 3. ENVIRONMENT SETUP (From Setup_Real_Scenario.py)
# ===========================


def traffic_light_logic(trafficLight1, trafficLight2, trafficLight3, trafficLight4):
    """Cycles traffic lights in a background thread."""
    intersection1Flag = 0
    while not KILL_PROGRAM:
        if intersection1Flag == 0:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_GREEN)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_GREEN)

        if intersection1Flag == 1:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_YELLOW)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_YELLOW)

        if intersection1Flag == 2:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_GREEN)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_GREEN)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_RED)

        if intersection1Flag == 3:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_YELLOW)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_YELLOW)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_RED)

        intersection1Flag = (intersection1Flag + 1) % 4
        time.sleep(5)


def setup_environment(qlabs, initialPosition, initialOrientation):
    """Full environment setup from Setup_Real_Scenario.py"""

    # Set the Workspace Title
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string("Simple Node Follower", waitForConfirmation=True)

    ### Flooring
    x_offset = 0.13
    y_offset = 1.67
    hFloor = QLabsQCarFlooring(qlabs)
    hFloor.spawn_degrees([x_offset, y_offset, 0.001], rotation=[0, 0, -90])

    ### Walls
    hWall = QLabsWalls(qlabs)
    hWall.set_enable_dynamics(False)

    for y in range(5):
        hWall.spawn_degrees(
            location=[-2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
            rotation=[0, 0, 0],
        )
    for x in range(5):
        hWall.spawn_degrees(
            location=[-1.9 + x + x_offset, 3.05 + y_offset, 0.001], rotation=[0, 0, 90]
        )
    for y in range(6):
        hWall.spawn_degrees(
            location=[2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
            rotation=[0, 0, 0],
        )
    for x in range(4):
        hWall.spawn_degrees(
            location=[-0.9 + x + x_offset, -3.05 + y_offset, 0.001], rotation=[0, 0, 90]
        )

    hWall.spawn_degrees(
        location=[-2.03 + x_offset, -2.275 + y_offset, 0.001], rotation=[0, 0, 48]
    )
    hWall.spawn_degrees(
        location=[-1.575 + x_offset, -2.7 + y_offset, 0.001], rotation=[0, 0, 48]
    )

    # Spawn QCar
    car2 = QLabsQCar2(qlabs)
    car2.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=True,
    )

    # Spawn Cameras
    camera1Loc = [0.15, 1.7, 5]
    camera1Rot = [0, 90, 0]
    camera1 = QLabsFreeCamera(qlabs)
    camera1.spawn_degrees(location=camera1Loc, rotation=camera1Rot)

    # Stop Signs
    myStopSign = QLabsStopSign(qlabs)
    myStopSign.spawn_degrees(
        location=[-1.5, 3.6, 0.006],
        rotation=[0, 0, -35],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myStopSign.spawn_degrees(
        location=[-1.5, 2.2, 0.006],
        rotation=[0, 0, 35],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myStopSign.spawn_degrees(
        location=[2.410, 0.206, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myStopSign.spawn_degrees(
        location=[1.766, 1.697, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )

    # Roundabout Signs
    myRoundaboutSign = QLabsRoundaboutSign(qlabs)
    myRoundaboutSign.spawn_degrees(
        location=[2.392, 2.522, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myRoundaboutSign.spawn_degrees(
        location=[0.698, 2.483, 0.006],
        rotation=[0, 0, -145],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myRoundaboutSign.spawn_degrees(
        location=[0.007, 3.973, 0.006],
        rotation=[0, 0, 135],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )

    # Yield Signs
    myYieldSign = QLabsYieldSign(qlabs)
    myYieldSign.spawn_degrees(
        location=[0.0, -1.3, 0.006],
        rotation=[0, 0, -180],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myYieldSign.spawn_degrees(
        location=[2.4, 3.2, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myYieldSign.spawn_degrees(
        location=[1.1, 2.8, 0.006],
        rotation=[0, 0, -145],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myYieldSign.spawn_degrees(
        location=[0.49, 3.8, 0.006],
        rotation=[0, 0, 135],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )

    # Crosswalks
    myCrossWalk = QLabsCrosswalk(qlabs)
    myCrossWalk.spawn_degrees(
        location=[-2 + x_offset, -1.475 + y_offset, 0.01],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[-0.5, 0.95, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[0.15, 0.32, 0.006],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[0.75, 0.95, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[0.13, 1.57, 0.006],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[1.45, 0.95, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )

    # Splines (Line Guidance)
    mySpline = QLabsBasicShape(qlabs)
    mySpline.spawn_degrees(
        location=[2.21, 0.2, 0.006],
        rotation=[0, 0, 0],
        scale=[0.27, 0.02, 0.001],
        waitForConfirmation=False,
    )
    mySpline.spawn_degrees(
        location=[1.951, 1.68, 0.006],
        rotation=[0, 0, 0],
        scale=[0.27, 0.02, 0.001],
        waitForConfirmation=False,
    )
    mySpline.spawn_degrees(
        location=[-0.05, -1.02, 0.006],
        rotation=[0, 0, 90],
        scale=[0.38, 0.02, 0.001],
        waitForConfirmation=False,
    )

    # Start Real-Time Model
    rtModel = os.path.normpath(
        os.path.join(os.environ["RTMODELS_DIR"], "QCar2/QCar2_Workspace_studio")
    )
    QLabsRealTime().start_real_time_model(rtModel)

    return car2


# ===========================
# 4. MAIN
# ===========================
def main():
    os.system("cls")
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return

    print("Connected. Resetting Environment...")
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # 1. Setup Complex Environment (Walls, Signs, etc.)
    car_actor = setup_environment(qlabs, INITIAL_POS, INITIAL_ROT)

    # 2. Setup Traffic Lights
    trafficLight1 = QLabsTrafficLight(qlabs)
    trafficLight2 = QLabsTrafficLight(qlabs)
    trafficLight3 = QLabsTrafficLight(qlabs)
    trafficLight4 = QLabsTrafficLight(qlabs)

    trafficLight1.spawn_id_degrees(
        actorNumber=1,
        location=[0.6, 1.55, 0.006],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )
    trafficLight2.spawn_id_degrees(
        actorNumber=2,
        location=[-0.6, 1.28, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )
    trafficLight3.spawn_id_degrees(
        actorNumber=3,
        location=[-0.37, 0.3, 0.006],
        rotation=[0, 0, 180],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )
    trafficLight4.spawn_id_degrees(
        actorNumber=4,
        location=[0.75, 0.48, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )

    tl_thread = threading.Thread(
        target=traffic_light_logic,
        args=(trafficLight1, trafficLight2, trafficLight3, trafficLight4),
        daemon=True,
    )
    tl_thread.start()

    # 3. Setup Hardware & Pathing (Simpler main.py approach)
    qcar = QCar(readMode=1, frequency=CONTROLLER_RATE)
    gps = QCarGPS(initialPose=INITIAL_POS)
    ekf = QCarEKF(x_0=INITIAL_POS)

    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(NODE_SEQUENCE)

    speed_ctrl = SpeedController()
    steer_ctrl = SteeringController(waypoints=waypointSequence, cyclic=True)

    print(f"Environment Ready. Following Nodes: {NODE_SEQUENCE}")

    # 4. Main Control Loop
    with qcar, gps:
        t0 = time.time()
        while not KILL_PROGRAM:
            t = time.time() - t0
            dt = 1.0 / CONTROLLER_RATE

            # Read Sensors
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([qcar.motorTach, 0], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, 0], dt, None, qcar.gyroscope[2])

            x, y, th = ekf.x_hat[0, 0], ekf.x_hat[1, 0], ekf.x_hat[2, 0]
            v = qcar.motorTach

            # Calculate Steering (Front Axle)
            p_front = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2

            if t >= START_DELAY:
                thr = speed_ctrl.update(v, V_REF, dt)
                str_ang = steer_ctrl.update(p_front, th, v)
                qcar.write(thr, str_ang)
            else:
                qcar.write(0, 0)

            time.sleep(dt)

    # Cleanup
    print("Stopping...")
    QLabsRealTime().terminate_all_real_time_models()


if __name__ == "__main__":
    main()
