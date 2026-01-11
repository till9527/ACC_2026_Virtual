# region: package imports
import os
import time
import numpy as np
import threading
# environment objects

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.qcar_flooring import QLabsQCarFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight

from pal.products.qcar import QCar, QCarGPS
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
V_REF = 1.0
CONTROLLER_RATE = 100
TAXI_HUB_POS = [-1.205, -0.83,0.005]
PICKUP_POS   = [0.125, 4.395]
DROPOFF_POS  = [-0.905, 0.800]

# Colors [R, G, B]
MAGENTA = [1.0, 0.0, 1.0]
GREEN   = [0.0, 1.0, 0.0]
BLUE    = [0.0, 0.0, 1.0]
ORANGE  = [1.0, 0.5, 0.0]
class SpeedController:
    """Locked speed controller with anti-surge logic """
    def __init__(self, kp=0.04, ki=0.15): 
        self.maxThrottle = 1.0
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        self.ei = np.clip(self.ei, -0.2, 0.2) # Tight windup limit to prevent speed spikes 
        return np.clip(self.kp * e + self.ki * self.ei, 0.0, self.maxThrottle)

class SteeringController:
    """Dampened Stanley Controller to eliminate 'waddling' """
    def __init__(self, waypoints, k=0.4, cyclic=True): 
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    def update(self, p, th, speed):
        # Prevent division by zero; use a floor for calculation stability 
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
        
        # High stability steering law 
        steering = psi + np.arctan2(self.k * ect, calc_speed)
        return np.clip(wrap_to_pi(steering), -self.maxSteeringAngle, self.maxSteeringAngle)
    
def wait_keep_alive(qcar, gps, ekf, duration):
    """Waits for a set duration while maintaining the QLabs connection."""
    start_time = time.time()
    dt = 1.0 / CONTROLLER_RATE
    
    print(f"Waiting for {duration} seconds...")
    while time.time() - start_time < duration:
        # Keep reading sensors to prevent timeout
        qcar.read()
        
        # Update EKF so the car doesn't 'jump' when it starts moving again
        if gps.readGPS():
            y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
            ekf.update([qcar.motorTach, 0], dt, y_gps, qcar.gyroscope[2])
        else:
            ekf.update([qcar.motorTach, 0], dt, None, qcar.gyroscope[2])
            
        # Explicitly command 0 throttle and 0 steering to stay still
        qcar.write(0, 0)
        time.sleep(dt)

from hal.products.mats import SDCSRoadMap #

def drive_to_target_hybrid(qcar, gps, ekf, node_sequence, final_coord, speed_ctrl, car_actor):
    """
    Drives through a sequence of roadmap nodes for lane centering, 
    then finishes at a specific coordinate.
    """
    car_actor.set_led_strip_uniform(GREEN)
    dt = 1.0 / CONTROLLER_RATE
    roadmap = SDCSRoadMap(leftHandTraffic=False) #
    
    # 1. Generate the 'Lane Centered' part of the path from nodes
    if node_sequence:
        path_waypoints = roadmap.generate_path(node_sequence) #
    else:
        # If no nodes provided, start from current position
        qcar.read()
        path_waypoints = np.array([[ekf.x_hat[0,0]], [ekf.x_hat[1,0]]])

    # 2. Append the 'Specific Coordinate' to the end of the roadmap waypoints
    # This creates a smooth transition from the lane center to your specific spot
    final_segment = np.linspace(path_waypoints[:, -1], final_coord, num=10).T 
    full_waypoints = np.hstack((path_waypoints, final_segment)) #
    
    # 3. Initialize one controller for the entire combined path
    steer_ctrl = SteeringController(waypoints=full_waypoints, k=0.15) #
    
    print(f"Navigating via nodes {node_sequence} to coordinate {final_coord}")
    
    while True:
        qcar.read() #
        
        if gps.readGPS():
            y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
            ekf.update([qcar.motorTach, 0], dt, y_gps, qcar.gyroscope[2])
        else:
            ekf.update([qcar.motorTach, 0], dt, None, qcar.gyroscope[2])
        x, y, th = ekf.x_hat[0, 0], ekf.x_hat[1, 0], ekf.x_hat[2, 0]
        v = qcar.motorTach
        
        # Look-ahead point for Stanley
        p_front = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.3 #

        # Calculate Control
        thr = speed_ctrl.update(v, V_REF, dt)
        str_ang = steer_ctrl.update(p_front, th, v)
        qcar.write(thr, str_ang)

        # Check if we reached the final specific coordinate
        dist_to_final = np.linalg.norm(np.array(final_coord) - np.array([x, y]))
        if dist_to_final < 0.2:
            print("Arrived at specific coordinate.")
            qcar.write(0, 0) #
            break
            
        time.sleep(dt)
def traffic_light_logic(trafficLight1,trafficLight2,trafficLight3,trafficLight4):
    intersection1Flag = 0
    while True:
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

        intersection1Flag = (intersection1Flag + 1)%4

        time.sleep(5)


def main():

        # Try to connect to Qlabs

    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except:
        print("Unable to connect to QLabs")
        quit()

    # Delete any previous QCar instances and stop any running spawn models
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()


    car_actor = setup(qlabs=qlabs, initialPosition = [-1.205, -0.83, 0.005], initialOrientation = [0, 0, -44.7])

    #spawing active components (traffic lights)
    
    # initialize 7 traffic light instances in qlabs
    trafficLight1 = QLabsTrafficLight(qlabs)
    trafficLight2 = QLabsTrafficLight(qlabs)
    trafficLight3 = QLabsTrafficLight(qlabs)
    trafficLight4 = QLabsTrafficLight(qlabs)

    #intersection 1
    trafficLight1.spawn_id_degrees(actorNumber=1, location=[0.6, 1.55, 0.006], rotation=[0,0,0], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)
    trafficLight2.spawn_id_degrees(actorNumber=2, location=[-0.6, 1.28, 0.006], rotation=[0,0,90], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)
    trafficLight3.spawn_id_degrees(actorNumber=3, location=[-0.37, 0.3, 0.006], rotation=[0,0,180], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)
    trafficLight4.spawn_id_degrees(actorNumber=4, location=[0.75, 0.48, 0.006], rotation=[0,0,-90], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)

    

    print('Starting Traffic Light Sequence')
    tl_thread = threading.Thread(target=traffic_light_logic, args=(trafficLight1, trafficLight2, trafficLight3, trafficLight4), daemon=True)
    tl_thread.start()
    speed_ctrl = SpeedController()
    qcar = QCar(readMode=1, frequency=CONTROLLER_RATE)
    gps = QCarGPS(initialPose=TAXI_HUB_POS)
    ekf = QCarEKF(x_0=TAXI_HUB_POS)

    with qcar, gps:
        # 2. Magenta (Awaiting mission)
        print("Step 2: Awaiting Mission (Magenta)")
        car_actor.set_led_strip_uniform(MAGENTA)
        wait_keep_alive(qcar, gps, ekf, 5)

        # 3. Green & Navigate to Pickup
        print("Step 3: Moving to Pickup (Green)")
        #car_actor.set_led_strip_uniform(GREEN)
        # Example: Use nodes 10 and 4 to stay in the lane, then go to the PICKUP_POS
        drive_to_target_hybrid(
            qcar, gps, ekf, 
            node_sequence=[2, 4,14,20], 
            final_coord=PICKUP_POS, 
            speed_ctrl=speed_ctrl, 
            car_actor=car_actor,
            
        )

        # 4. Stop & Blue (Passenger Pick up)
        print("Step 4: Passenger Pickup (Blue)")
        
        car_actor.set_led_strip_uniform(BLUE)
        wait_keep_alive(qcar, gps, ekf, 5)

        # 5. Navigate to Drop-off
        print("Step 5: Moving to Drop-off (Green)")
        # #car_actor.set_led_strip_uniform(GREEN)
        # drive_to_coordinate(qcar, gps, ekf, DROPOFF_POS, speed_ctrl,car_actor)
        drive_to_target_hybrid(
            qcar, gps, ekf, 
            node_sequence=[20,22,9], 
            final_coord=DROPOFF_POS, 
            speed_ctrl=speed_ctrl, 
            car_actor=car_actor,
            
        )

        # # 6. Stop & Orange (Passenger Drop off)
        print("Step 6: Passenger Drop-off (Orange)")
        car_actor.set_led_strip_uniform(ORANGE)
        wait_keep_alive(qcar, gps, ekf, 5)

        # # 7. Back to Hub & Magenta
        print("Step 7: Returning to Hub")
        # drive_to_coordinate(qcar, gps, ekf, TAXI_HUB_POS[:2], speed_ctrl, car_actor)
        drive_to_target_hybrid(
            qcar, gps, ekf, 
            node_sequence=[9,7,14,20,22,10], 
            final_coord=TAXI_HUB_POS[:2], 
            speed_ctrl=speed_ctrl, 
            car_actor=car_actor,
        )
        car_actor.set_led_strip_uniform(MAGENTA)
        # wait_keep_alive(qcar, gps, ekf, 5)
        print("Mission Complete.")
        
   



#Function to setup QLabs, Spawn in QCar, and run real time model
def setup(qlabs, initialPosition = [-1.205, -0.83, 0.005], initialOrientation = [0, 0, -44.7]):

    # Try to connect to Qlabs

    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except:
        print("Unable to connect to QLabs")
        quit()

    # Delete any previous QCar instances and stop any running spawn models
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    #Set the Workspace Title
    hSystem = QLabsSystem(qlabs)
    x = hSystem.set_title_string('ACC Self Driving Car Competition', waitForConfirmation=True)

    ### Flooring

    x_offset = 0.13
    y_offset = 1.67
    hFloor = QLabsQCarFlooring(qlabs)
    hFloor.spawn_degrees([x_offset, y_offset, 0.001],rotation = [0, 0, -90])


    ### region: Walls
    hWall = QLabsWalls(qlabs)
    hWall.set_enable_dynamics(False)

    for y in range (5):
        hWall.spawn_degrees(location=[-2.4 + x_offset, (-y*1.0)+2.55 + y_offset, 0.001], rotation=[0, 0, 0])

    for x in range (5):
        hWall.spawn_degrees(location=[-1.9+x + x_offset, 3.05+ y_offset, 0.001], rotation=[0, 0, 90])

    for y in range (6):
        hWall.spawn_degrees(location=[2.4+ x_offset, (-y*1.0)+2.55 + y_offset, 0.001], rotation=[0, 0, 0])

    for x in range (4):
        hWall.spawn_degrees(location=[-0.9+x+ x_offset, -3.05+ y_offset, 0.001], rotation=[0, 0, 90])

    hWall.spawn_degrees(location=[-2.03 + x_offset, -2.275+ y_offset, 0.001], rotation=[0, 0, 48])
    hWall.spawn_degrees(location=[-1.575+ x_offset, -2.7+ y_offset, 0.001], rotation=[0, 0, 48])


    # Spawn a QCar at the given initial pose
    car2 = QLabsQCar2(qlabs)
    car2.spawn_id(actorNumber=0, 
                location=initialPosition, 
                rotation=initialOrientation,
                scale=[.1, .1, .1], 
                configuration=0, 
                waitForConfirmation=True)

    #spawn cameras 1. birds eye, 2. edge 1, possess the qcar

    camera1Loc = [0.15, 1.7, 5]
    camera1Rot = [0, 90, 0]
    camera1 = QLabsFreeCamera(qlabs)
    camera1.spawn_degrees(location=camera1Loc, rotation=camera1Rot)

    #camera1.possess()

    camera2Loc = [-0.36+ x_offset, -3.691+ y_offset, 2.652]
    camera2Rot = [0, 47, 90]
    camera2=QLabsFreeCamera(qlabs)
    camera2.spawn_degrees (location = camera2Loc, rotation=camera2Rot)

    camera2.possess()

    # stop signs
    #parking lot
    myStopSign = QLabsStopSign(qlabs)
    
    myStopSign.spawn_degrees (location=[-1.5, 3.6, 0.006], 
                            rotation=[0, 0, -35], 
                            scale=[0.1, 0.1, 0.1], 
                            waitForConfirmation=False)    

    myStopSign.spawn_degrees (location=[-1.5, 2.2, 0.006], 
                            rotation=[0, 0, 35], 
                            scale=[0.1, 0.1, 0.1], 
                            waitForConfirmation=False)  
    
    #x+ side
    myStopSign.spawn_degrees (location=[2.410, 0.206, 0.006], 
                            rotation=[0, 0, -90], 
                            scale=[0.1, 0.1, 0.1], 
                            waitForConfirmation=False)  
    
    myStopSign.spawn_degrees (location=[1.766, 1.697, 0.006], 
                            rotation=[0, 0, 90], 
                            scale=[0.1, 0.1, 0.1], 
                            waitForConfirmation=False)  

    #roundabout signs
    myRoundaboutSign = QLabsRoundaboutSign(qlabs)
    myRoundaboutSign.spawn_degrees(location= [2.392, 2.522, 0.006],
                              rotation=[0, 0, -90],
                              scale= [0.1, 0.1, 0.1],
                              waitForConfirmation=False)
    
    myRoundaboutSign.spawn_degrees(location= [0.698, 2.483, 0.006],
                              rotation=[0, 0, -145],
                              scale= [0.1, 0.1, 0.1],
                              waitForConfirmation=False)
    
    myRoundaboutSign.spawn_degrees(location= [0.007, 3.973, 0.006],
                            rotation=[0, 0, 135],
                            scale= [0.1, 0.1, 0.1],
                            waitForConfirmation=False)


    #yield sign
    #one way exit yield
    myYieldSign = QLabsYieldSign(qlabs)
    myYieldSign.spawn_degrees(location= [0.0, -1.3, 0.006],
                              rotation=[0, 0, -180],
                              scale= [0.1, 0.1, 0.1],
                              waitForConfirmation=False)
    
    #roundabout yields
    myYieldSign.spawn_degrees(location= [2.4, 3.2, 0.006],
                            rotation=[0, 0, -90],
                            scale= [0.1, 0.1, 0.1],
                            waitForConfirmation=False)
    
    myYieldSign.spawn_degrees(location= [1.1, 2.8, 0.006],
                            rotation=[0, 0, -145],
                            scale= [0.1, 0.1, 0.1],
                            waitForConfirmation=False)
    
    myYieldSign.spawn_degrees(location= [0.49, 3.8, 0.006],
                            rotation=[0, 0, 135],
                            scale= [0.1, 0.1, 0.1],
                            waitForConfirmation=False)
    
    

    # Spawning crosswalks
    myCrossWalk = QLabsCrosswalk(qlabs)
    myCrossWalk.spawn_degrees   (location =[-2 + x_offset, -1.475 + y_offset, 0.01],
                                rotation=[0,0,0], 
                                scale = [0.1,0.1,0.075],
                                configuration = 0)

    myCrossWalk.spawn_degrees   (location =[-0.5, 0.95, 0.006],
                                rotation=[0,0,90], 
                                scale = [0.1,0.1,0.075],
                                configuration = 0)
    
    myCrossWalk.spawn_degrees   (location =[0.15, 0.32, 0.006],
                                rotation=[0,0,0], 
                                scale = [0.1,0.1,0.075],
                                configuration = 0)

    myCrossWalk.spawn_degrees   (location =[0.75, 0.95, 0.006],
                                rotation=[0,0,90], 
                                scale = [0.1,0.1,0.075],
                                configuration = 0)

    myCrossWalk.spawn_degrees   (location =[0.13, 1.57, 0.006],
                                rotation=[0,0,0], 
                                scale = [0.1,0.1,0.075],
                                configuration = 0)

    myCrossWalk.spawn_degrees   (location =[1.45, 0.95, 0.006],
                                rotation=[0,0,90], 
                                scale = [0.1,0.1,0.075],
                                configuration = 0)

    #Signage line guidance (white lines)
    mySpline = QLabsBasicShape(qlabs)
    mySpline.spawn_degrees (location=[2.21, 0.2, 0.006], 
                            rotation=[0, 0, 0], 
                            scale=[0.27, 0.02, 0.001], 
                            waitForConfirmation=False)

    mySpline.spawn_degrees (location=[1.951, 1.68, 0.006], 
                            rotation=[0, 0, 0], 
                            scale=[0.27, 0.02, 0.001], 
                            waitForConfirmation=False)

    mySpline.spawn_degrees (location=[-0.05, -1.02, 0.006], 
                            rotation=[0, 0, 90], 
                            scale=[0.38, 0.02, 0.001], 
                            waitForConfirmation=False)

    # define rt model path
    rtModel = os.path.normpath(os.path.join(os.environ['RTMODELS_DIR'], 'QCar2/QCar2_Workspace_studio'))
    # Start spawn model
    QLabsRealTime().start_real_time_model(rtModel)

    return car2

#function to terminate the real time model running
def terminate():
    rtModel = os.path.normpath(os.path.join(os.environ['RTMODELS_DIR'], 'QCar2/QCar2_Workspace_studio'))
    QLabsRealTime().terminate_real_time_model(rtModel)
if __name__ == '__main__':
    main()