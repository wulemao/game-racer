import os
import sys
import traceback
# this file: game-racer/car_ros/scripts/racing_blocking.py
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # -> game-racer

if _GAME_RACER_ROOT not in sys.path:
    sys.path.insert(0, _GAME_RACER_ROOT)

PARAMS_PATH = os.path.join(
    _GAME_RACER_ROOT,
    "sim_params",
    "params-num.yaml",
)
import rospy
from std_msgs.msg import Float64, Int8, String
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point32, PolygonStamped, TwistStamped
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import MarkerArray, Marker
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import jax
import jax.numpy as jnp
from tf.transformations import quaternion_from_euler, euler_matrix
from ackermann_msgs.msg import AckermannDrive
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import pickle
from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_jax import WaypointGenerator
from car_dynamics.envs.car3 import OffroadCar
from mpc_controller import mpc


print("DEVICE", jax.devices())

DT = 0.1
DT_torch = 0.1
DELAY = 2
N_ROLLOUTS = 10000
H = 8
SIGMA = 1.0
track_width = 1.
LON_THRES = 3.

EP_LEN = 500
EP_START = 0
EP_END = 100
RUN_SPEED = 2.
# FILENAME = 'racedata_ours_vs_mpc_vs_mpc_mpc.pkl'
# FILENAME = 'recorded_races/racedata_ours_vs_mpc_vs_mpc_grad.pkl'
FILENAME = os.path.join(
    _GAME_RACER_ROOT,
    "car_ros",
    "recorded_races",
    "racedata_ours_vs_mpc_vs_mpc_grad.pkl",
)

# trajectory_type = "counter oval"
trajectory_type = "berlin_2018"
SIM = 'numerical' # 'numerical' or 'unity' or 'vicon'


if SIM == 'numerical' :
    # trajectory_type = "../../sim_params/params-num.yaml"
    trajectory_type = PARAMS_PATH
    LF = 0.12
    LR = 0.24
    L = LF+LR

if SIM=='unity' :
    # trajectory_type = "../../sim_params/params.yaml"
    trajectory_type = os.path.join(
    _GAME_RACER_ROOT,
    "sim_params",
    "params.yaml",
    )
    LF = 1.6
    LR = 1.5
    L = LF+LR

if SIM == 'unity' :
    SPEED = 10.0
else :
    SPEED = 2.

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if SIM == 'unity' :
    yaml_contents = yaml.load(open(trajectory_type, 'r'), Loader=yaml.FullLoader)
    
    decay_start = yaml_contents['vehicle_params']['friction_decay_start']
    decay_rate = yaml_contents['vehicle_params']['friction_decay_rate']

AUGMENT = False
use_gt = True
HISTORY = 8
ART_DELAY = 0
MAX_DELAY = 7
new_delay_factor = 0.1
curr_delay = 0.
N_ensembles = 1
append_delay_type = 'OneHot' # 'OneHot' or 'Append'
LAST_LAYER_ADAPTATION = False
mass = 1.
I = 1.

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')

args = parser.parse_args()

model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)#random.randint(1,5))
model_params_single_opp = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)#random.randint(1,5))
model_params_single_opp1 = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)#random.randint(1,5))


exp_name = args.exp_name

dynamics_single = DynamicBicycleModel(model_params_single)
dynamics_single_opp = DynamicBicycleModel(model_params_single_opp)
dynamics_single_opp1 = DynamicBicycleModel(model_params_single_opp)

dynamics_single.reset()
dynamics_single_opp.reset()
dynamics_single_opp1.reset()


waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 2.)
waypoint_generator_opp = WaypointGenerator(trajectory_type, DT, H, 1.)
waypoint_generator_opp1 = WaypointGenerator(trajectory_type, DT, H, 1.)


done = False
frames = []

if SIM == 'numerical' :
    env = OffroadCar({}, dynamics_single)
    env_opp = OffroadCar({}, dynamics_single_opp)
    env_opp1 = OffroadCar({}, dynamics_single_opp1)
    obs = env.reset(pose=[3.,5.,-np.pi/2.-0.72])
    obs_opp = env_opp.reset(pose=[0.,0.,-np.pi/2.-0.5])
    obs_opp1 = env_opp1.reset(pose=[-2.,-6.,-np.pi/2.-0.3])

goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []


pos2d = []
target_list_all = []


curr_steer = 0.

class CarNode:
    def __init__(self):
        rospy.init_node('car_node')  # Initialize ROS 1 node

        # Publishers
        self.path_pub_ = rospy.Publisher('path', Path, queue_size=1)
        self.left_boundary_pub_ = rospy.Publisher('left_boundary', Path, queue_size=1)
        self.right_boundary_pub_ = rospy.Publisher('right_boundary', Path, queue_size=1)
        self.raceline_pub_ = rospy.Publisher('raceline', Path, queue_size=1)
        self.ref_trajectory_pub_ = rospy.Publisher('ref_trajectory', Path, queue_size=1)
        self.pose_pub_ = rospy.Publisher('pose', PoseWithCovarianceStamped, queue_size=1)
        self.odom_pub_ = rospy.Publisher('odom', Odometry, queue_size=1)
        self.odom_opp_pub_ = rospy.Publisher('odom_opp', Odometry, queue_size=1)
        self.odom_opp1_pub_ = rospy.Publisher('odom_opp1', Odometry, queue_size=1)

        # Timer (using rospy.Timer instead of self.create_timer)
        self.timer_ = rospy.Timer(rospy.Duration(DT / RUN_SPEED), self.timer_callback)
        self.slow_timer_ = rospy.Timer(rospy.Duration(10.0), self.slow_timer_callback)

        # Additional publishers
        self.throttle_pub_ = rospy.Publisher('throttle', Float64, queue_size=1)
        self.steer_pub_ = rospy.Publisher('steer', Float64, queue_size=1)
        self.body_pub_ = rospy.Publisher('body', PolygonStamped, queue_size=1)
        self.body_opp_pub_ = rospy.Publisher('body_opp', PolygonStamped, queue_size=1)
        self.body_opp1_pub_ = rospy.Publisher('body_opp1', PolygonStamped, queue_size=1)
        self.status_pub_ = rospy.Publisher('status', Int8, queue_size=1)

        self.raceline = waypoint_generator.raceline
        self.ep_no = EP_START
        self.raceline_dev = waypoint_generator.raceline_dev
        self.race_data = np.array(pickle.load(open(FILENAME,'rb')))

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster()

        # Initialize the URDF content
        # urdf_path = "car.urdf"  # Replace with your URDF file path
        urdf_path = os.path.join(
        _GAME_RACER_ROOT,
        "sim_params",
        "car.urdf",
        )

        try:
            with open(urdf_path, 'r') as urdf_file:
                urdf_content = urdf_file.read()
            rospy.loginfo(f"Successfully loaded URDF from {urdf_path}")
        except FileNotFoundError:
            rospy.logerr(f"URDF file not found at {urdf_path}")
            urdf_content = ""
        
        self.urdf_content = urdf_content

        # Initialize other parameters
        self.last_i = -1
        self.last_i_opp = -1
        self.last_i_opp1 = -1
        self.L = LF + LR

        self.curr_speed_factor = 1.0
        self.curr_lookahead_factor = 0.24
        self.curr_sf1 = 0.2
        self.curr_sf2 = 0.2
        self.blocking = 0.2
        
        self.curr_speed_factor_opp = 1.0
        self.curr_lookahead_factor_opp = 0.15
        self.curr_sf1_opp = 0.1
        self.curr_sf2_opp = 0.5
        self.blocking_opp = 0.2
        
        self.curr_speed_factor_opp1 = 1.0
        self.curr_lookahead_factor_opp1 = 0.15
        self.curr_sf1_opp1 = 0.1
        self.curr_sf2_opp1 = 0.5
        self.blocking_opp1 = 0.2

        # Unity-related setup (if applicable)
        if SIM == 'unity':
            self.unity_publisher_ = rospy.Publisher('/cmd', AckermannDrive, queue_size=10)
            self.ackermann_msg = AckermannDrive()

            # Unity subscribers
            self.unity_subscriber_ = rospy.Subscriber('car_pose', PoseStamped, self.unity_callback)
            self.unity_subscriber_twist_ = rospy.Subscriber('car_twist', TwistStamped, self.unity_twist_callback)

            # Unity state initialization
            self.unity_state = [yaml_contents['respawn_loc']['z'], yaml_contents['respawn_loc']['x'], 0., 0., 0., 0.]
            self.pose_received = True
            self.vel_received = True
            self.mu_factor_pub_ = rospy.Publisher('mu_factor', Float64, queue_size=1)

        # Timer setup (ROS 1 uses rospy.Timer)
        self.timer_ = rospy.Timer(rospy.Duration(DT / RUN_SPEED), self.timer_callback)
        self.slow_timer_ = rospy.Timer(rospy.Duration(10.0), self.slow_timer_callback)

        # Initialize variables
        self.i = 0
        self.curr_t_counter = 0.0
        self.unity_state_new = [0., 0., 0., 0., 0., 0.]
        self.dataset = []

    def timer_callback(self, event):
        ti = time.time()
        if SIM == 'unity' and not self.pose_received :
            return
        if SIM == 'unity' and not self.vel_received :
            return
        
        if self.i >= EP_LEN :
            self.ep_no += 1
            self.i = 1
            print("Race no ", self.ep_no)
            
        if self.ep_no > EP_END :
            exit(0)
        
        mu_factor = 1.
        if SIM == 'unity' :
            if self.i*DT > decay_start :
                mu_factor = 1. - (self.i*DT - decay_start)*decay_rate
            mu_msg = Float64()
            mu_msg.data = mu_factor
            self.mu_factor_pub_.publish(mu_msg)
        
        if SIM == 'unity':
            if self.i==1 :
                action = np.array([0.,0.])
                action[0] = -3.
                action[1] = -3.
            self.unity_state_new = self.unity_state.copy()
            obs = np.array(self.unity_state)
            self.ackermann_msg.acceleration = float(action[0])
            self.ackermann_msg.steering_angle = float(action[1])
            self.unity_publisher_.publish(self.ackermann_msg)
        
        obs = self.race_data[self.ep_no][self.i]
        self.i += 1

        px,py,psi,px_opp,py_opp,psi_opp,px_opp1,py_opp1,psi_opp1 = obs[:9]
        self.curr_sf1, self.curr_sf2, self.curr_lookahead_factor, self.curr_speed_factor, self.blocking = obs[9:14]
        self.curr_sf1_opp, self.curr_sf2_opp, self.curr_lookahead_factor_opp, self.curr_speed_factor_opp, self.blocking_opp = obs[14:19]
        self.curr_sf1_opp1, self.curr_sf2_opp1, self.curr_lookahead_factor_opp1, self.curr_speed_factor_opp1, self.blocking_opp1 = obs[19:24]
        throttle_, steer_ = obs[24:]
        q = quaternion_from_euler(0, 0, psi)
        q_opp = quaternion_from_euler(0, 0, psi_opp)
        q_opp1 = quaternion_from_euler(0, 0, psi_opp1)
        now = rospy.Time.now()
        
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = now
        pose.pose.pose.position.x = px
        pose.pose.pose.position.y = py
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]
        self.pose_pub_.publish(pose)

        # Create a TransformStamped message
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"  # Parent frame
        t.child_frame_id = "base_link"  # Child frame (unique for each car)
        t.transform.translation.x = px
        t.transform.translation.y = py
        t.transform.translation.z = 0.
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]  # No rotation (identity quaternion)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

        time.sleep(0.02)
        # Repeat for other cars (opponents)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link1"  # Unique for opponent
        t.transform.translation.x = px_opp
        t.transform.translation.y = py_opp
        t.transform.translation.z = 0.
        t.transform.rotation.x = q_opp[0]
        t.transform.rotation.y = q_opp[1]
        t.transform.rotation.z = q_opp[2]
        t.transform.rotation.w = q_opp[3]
        self.tf_broadcaster.sendTransform(t)
        
        time.sleep(0.02)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link2"  # Unique for opponent 2
        t.transform.translation.x = px_opp1
        t.transform.translation.y = py_opp1
        t.transform.translation.z = 0.
        t.transform.rotation.x = q_opp1[0]
        t.transform.rotation.y = q_opp1[1]
        t.transform.rotation.z = q_opp1[2]
        t.transform.rotation.w = q_opp1[3]
        self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.frame_id = 'map'
        odom.header.stamp = now
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self.odom_pub_.publish(odom)

        # Odom for opponent
        odom_opp = Odometry()
        odom_opp.header.frame_id = 'map'
        odom_opp.header.stamp = now
        odom_opp.pose.pose.position.x = px_opp
        odom_opp.pose.pose.position.y = py_opp
        odom_opp.pose.pose.orientation.x = q_opp[0]
        odom_opp.pose.pose.orientation.y = q_opp[1]
        odom_opp.pose.pose.orientation.z = q_opp[2]
        odom_opp.pose.pose.orientation.w = q_opp[3]
        self.odom_opp_pub_.publish(odom_opp)

        # Publish throttle and steer
        throttle = Float64()
        throttle.data = float(throttle_)
        self.throttle_pub_.publish(throttle)
        
        steer = Float64()
        steer.data = steer_
        self.steer_pub_.publish(steer)

        # Body polygon for main car
        pts = np.array([[LF, L/3], [LF, -L/3], [-LR, -L/3], [-LR, L/3]])
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts = np.dot(R, pts.T).T
        pts += np.array([px, py])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = now
        for i in range(pts.shape[0]):
            p = Point32()
            p.x = float(pts[i, 0])
            p.y = float(pts[i, 1])
            p.z = 0.0
            body.polygon.points.append(p)
        self.body_pub_.publish(body)

        # Repeat for other cars (opponents)
        # For opponent 1 and opponent 2
        # Similar process for the body polygons of opponent cars...

        tf = time.time()
        if SIM == 'unity':
            self.pose_received = False
            self.vel_received = False

    def slow_timer_callback(self, event):
        # publish waypoint_list as path
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()  # Use rospy.Time.now() instead of self.get_clock().now()

        # Publish left boundary
        for i in range(waypoint_generator.left_boundary.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.left_boundary[i][0])
            pose.pose.position.y = float(waypoint_generator.left_boundary[i][1])
            path.poses.append(pose)
        self.left_boundary_pub_.publish(path)

        # Publish right boundary
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for i in range(waypoint_generator.right_boundary.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.right_boundary[i][0])
            pose.pose.position.y = float(waypoint_generator.right_boundary[i][1])
            path.poses.append(pose)
        self.right_boundary_pub_.publish(path)

        # Publish raceline
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for i in range(waypoint_generator.raceline.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.raceline[i][0])
            pose.pose.position.y = float(waypoint_generator.raceline[i][1])
            path.poses.append(pose)
        self.raceline_pub_.publish(path)

def main():
    rospy.init_node('car_node')  # Initialize ROS 1 node
    car_node = CarNode()  # Instantiate your CarNode class
    rospy.spin()  # Keep the node running

if __name__ == '__main__':
    main()

