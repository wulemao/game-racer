#!/usr/bin/env python3

import os
import sys
import traceback
import yaml
import numpy as np
import torch
import jax
import pickle
import argparse
import rospy
import time
from std_msgs.msg import Float64, Int8
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point32, PolygonStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from tf.transformations import quaternion_from_euler, euler_matrix
from tf2_ros import TransformBroadcaster

# Path Setup
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _GAME_RACER_ROOT not in sys.path:
    sys.path.insert(0, _GAME_RACER_ROOT)

from car_dynamics.controllers_jax import WaypointGenerator

# --- Global Directory Definitions ---
CAR_ROS_DIR = os.path.join(_GAME_RACER_ROOT, "car_ros")
RECORDED_RACES_DIR = os.path.join(CAR_ROS_DIR, "recorded_races")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
VIS = True  # Enable visualization

class CarFleet:
    def __init__(self, config):
        self.config = config
        # Determine number of cars from config
        self.num_cars = config.get('sim', {}).get('num_cars', 3)
        
        self.cars = {}
        self.publishers = {}
        
        # Load params
        dyn = config['car_dynamics']
        self.DT = dyn['DT']
        self.LF = dyn['LF']
        self.LR = dyn['LR']
        self.L = self.LF + self.LR
        
        # Simulation/Playback settings
        self.sim_type = config['sim']['type']
        sim_config = config['sim'][self.sim_type]
        self.trajectory_type = sim_config['trajectory_type']
        
        # Load Race Data
        self.exp_name = config.get('argparse', {}).get('exp_name', 'default_experiment')
        # Construct filename based on conventions or config
        # Assuming the file is named roughly after the experiment
        self.filename = os.path.join(RECORDED_RACES_DIR, f"racedata_{self.exp_name}.pkl")
        
        self.load_race_data()
        self.init_map_assets()
        self.init_publishers()
        
        self.tf_broadcaster = TransformBroadcaster()

    def load_race_data(self):
        try:
            print(f"Loading playback data from: {self.filename}")
            with open(self.filename, 'rb') as f:
                self.race_data = np.array(pickle.load(f))
            print(f"Data loaded. Shape: {self.race_data.shape}")
        except FileNotFoundError:
            print(f"Error: Playback file not found at {self.filename}")
            sys.exit(1)

    def init_map_assets(self):
        # We only need one generator for map visualization
        self.waypoint_generator = WaypointGenerator(self.trajectory_type, self.DT, 10, 1.0)

    def init_publishers(self):
        if not VIS: return
        for i in range(self.num_cars):
            # Dynamic naming: path_nn, path_nn_1, path_nn_2...
            name_idx = f"_{i}" if i > 0 else ""
            self.publishers[i] = {
                'pose': rospy.Publisher(f'pose{name_idx}', PoseWithCovarianceStamped, queue_size=1),
                'odom': rospy.Publisher(f'odom{name_idx}', Odometry, queue_size=1),
                'body': rospy.Publisher(f'body{name_idx}', PolygonStamped, queue_size=1),
            }
            # Only creating specific control publishers for the ego car (index 0) based on original logic
            # or we can generalize it. Assuming visualization for all.
            
        # Map publishers (static)
        self.map_pubs = {
            'waypoint_list': rospy.Publisher('waypoint_list', Path, queue_size=1),
            'left_boundary': rospy.Publisher('left_boundary', Path, queue_size=1),
            'right_boundary': rospy.Publisher('right_boundary', Path, queue_size=1),
            'raceline': rospy.Publisher('raceline', Path, queue_size=1)
        }
        
        self.mu_factor_pub = rospy.Publisher('mu_factor', Float64, queue_size=1)
        # Ego specific debug
        self.throttle_pub = rospy.Publisher('throttle', Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('steer', Float64, queue_size=1)

    def process_timestep(self, ep_no, step_idx):
        """
        Parses the flat observation array from the pickle file into N cars.
        Expects format: [Pose*N, Params*N, Control_Ego] 
        (Adjust logic below if your pickle format differs)
        """
        if ep_no >= len(self.race_data) or step_idx >= len(self.race_data[ep_no]):
            return False

        obs_row = self.race_data[ep_no][step_idx]
        
        # --- Parsing Logic ---
        # Assuming the pickle structure from the recording script:
        # [px, py, psi (for car 0), px, py, psi (for car 1)...] -> 3 * N floats
        # [params (5 floats) for car 0, params for car 1...] -> 5 * N floats
        # [throttle, steer] -> 2 floats
        
        pose_end_idx = 3 * self.num_cars
        poses = obs_row[:pose_end_idx]
        
        param_end_idx = pose_end_idx + (5 * self.num_cars)
        params = obs_row[pose_end_idx:param_end_idx]
        
        controls = obs_row[param_end_idx:] # throttle, steer (usually only for ego)

        # Update Ego Control Viz
        if len(controls) >= 2:
            self.throttle_pub.publish(Float64(controls[0]))
            self.steer_pub.publish(Float64(controls[1]))

        # Process each car
        timestamp = rospy.Time.now()
        
        for k in range(self.num_cars):
            # Extract Pose
            px = poses[k*3]
            py = poses[k*3 + 1]
            psi = poses[k*3 + 2]
            
            # Extract Params (Optional: print or visualize if needed)
            # p_start = k * 5
            # curr_params = params[p_start : p_start+5] 
            
            # Visualize
            self.visualize_car(k, px, py, psi, timestamp)
            
        return True

    def visualize_car(self, idx, px, py, psi, timestamp):
        pubs = self.publishers[idx]
        q = quaternion_from_euler(0, 0, psi)
        
        # 1. Pose
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = timestamp
        pose.pose.pose.position.x = px
        pose.pose.pose.position.y = py
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]
        pubs['pose'].publish(pose)

        # 2. TF Broadcast
        # Convention: base_link, base_link1, base_link2...
        child_frame = "base_link" if idx == 0 else f"base_link{idx}"
        
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = "map"
        t.child_frame_id = child_frame
        t.transform.translation.x = px
        t.transform.translation.y = py
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # 3. Odom (Simplified, mostly for position in Viz)
        odom = Odometry()
        odom.header.frame_id = 'map'
        odom.header.stamp = timestamp
        odom.pose.pose = pose.pose.pose
        pubs['odom'].publish(odom)

        # 4. Body Polygon
        pts = np.array([[self.LF, self.L/3], [self.LF, -self.L/3], [-self.LR, -self.L/3], [-self.LR, self.L/3]])
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts_w = np.dot(R, pts.T).T + np.array([px, py])
        
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = timestamp
        for pt in pts_w:
            p = Point32()
            p.x, p.y = float(pt[0]), float(pt[1])
            body.polygon.points.append(p)
        pubs['body'].publish(body)

    def visualize_map(self):
        timestamp = rospy.Time.now()
        wg = self.waypoint_generator
        
        def pub_path(pub, points):
            path = Path()
            path.header.frame_id = 'map'
            path.header.stamp = timestamp
            for i in range(len(points)):
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(points[i][0])
                pose.pose.position.y = float(points[i][1])
                path.poses.append(pose)
            pub.publish(path)

        pub_path(self.map_pubs['raceline'], wg.raceline)
        pub_path(self.map_pubs['left_boundary'], wg.left_boundary)
        pub_path(self.map_pubs['right_boundary'], wg.right_boundary)
        pub_path(self.map_pubs['waypoint_list'], wg.waypoint_list_np)


class CarNode:
    def __init__(self, config_path):
        rospy.init_node('car_playback_node')
        self.config = self.load_config(config_path)
        
        self.fleet = CarFleet(self.config)
        
        # Loop Control
        self.ep_start = self.config['car_dynamics'].get('EP_START', 0)
        self.ep_end = self.config['car_dynamics'].get('EP_END', 100)
        self.ep_len = self.config['car_dynamics'].get('EP_LEN', 500)
        self.run_speed = self.config['car_dynamics'].get('RUN_SPEED', 1.0)
        self.dt = self.config['car_dynamics']['DT']
        
        self.curr_ep = self.ep_start
        self.curr_step = 0
        
        # Timers
        # Playback speed adjustment
        self.timer = rospy.Timer(rospy.Duration(self.dt / self.run_speed), self.timer_callback)
        self.slow_timer = rospy.Timer(rospy.Duration(5.0), self.slow_timer_callback)

    @staticmethod
    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def timer_callback(self, event):
        try:
            # Check Episode Boundaries
            if self.curr_ep > self.ep_end:
                rospy.loginfo("Playback finished (Max Episodes reached).")
                rospy.signal_shutdown("Done")
                sys.exit(0)

            # Process Step
            success = self.fleet.process_timestep(self.curr_ep, self.curr_step)
            
            # Increment
            self.curr_step += 1
            
            # Handle End of Episode or Data
            if not success or self.curr_step >= self.ep_len:
                rospy.loginfo(f"Episode {self.curr_ep} finished.")
                self.curr_ep += 1
                self.curr_step = 0
                
                # Small pause between episodes for visual clarity
                time.sleep(0.5)

        except Exception:
            traceback.print_exc()

    def slow_timer_callback(self, event):
        # Publish static map elements periodically
        self.fleet.visualize_map()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Playback recorded race data")
    # Default to the path used in the original script if not provided
    default_config = os.path.join(_GAME_RACER_ROOT, "config", "collect_params.yaml")
    
    parser.add_argument('--config', default=default_config, help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        CarNode(args.config)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass