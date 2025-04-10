import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import pickle
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.subscription = self.create_subscription(
            PoseStamped,
            'qualysis/tb3_3',  # or whatever your marker_deck_name is
            self.pose_callback,
            10)

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        self.current_pose = None
        algo_name = "c3m"
        # Load the entire policy object
        with open(f"model/{algo_name}.p", "rb") as f:
            self.policy = pickle.load(f)
        self.ref_trajectory = np.load('ref.npz')
        self.timer = self.create_timer(0.1, self.control_loop)  

        self.internal_counter = 0

        time.sleep(5) # wait for the initialization of robots
        

    def pose_callback(self, msg):
        self.current_pose = msg

    def control_loop(self):
        if self.current_pose is None:
            return
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        yaw = self.current_pose.pose.orientation.z  # assuming yaw was stored in `.z`
        yaw = yaw % (2 * math.pi)

        x = np.array([x, y, yaw])
        state = np.concatenate((x, self.ref_trajectory[self.internal_counter]))

        # Turn toward the goal
        a = self.policy(state)
        a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]
        
        # translate to the ros2 command
        vel_msg = Twist()
        vel_msg.linear.x = a[0]
        vel_msg.angular.z = a[1]
            
        self.publisher_.publish(vel_msg)

        self.internal_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()