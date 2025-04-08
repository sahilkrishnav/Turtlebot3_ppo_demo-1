import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import math
import numpy as np
import time

class MDPDataCollector(Node):
    def __init__(self, total_steps=1000, save_path='mdp_data.npz'):
        super().__init__('mdp_data_collector')

        # Parameters
        self.total_steps = total_steps
        self.save_path = save_path

        # ROS 2 interfaces
        self.subscription = self.create_subscription(
            PoseStamped,
            'qualysis/tb3_3',  # your motion capture topic
            self.pose_callback,
            10)

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Data storage
        self.current_pose = None
        self.state_history = []
        self.action_history = []
        self.next_state_history = []

        # Control setup
        self.step_count = 0
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.previous_state = None  # For saving previous state

        # Fixed linear velocity
        self.linear_velocity_range = [0.05, 0.2]  # Constant forward velocity

        # Sinusoidal angular velocity parameters
        self.start_time = time.time()
        self.angular_amplitude = np.random.uniform(0.3, 1.0)  # Random amplitude
        self.angular_frequency = np.random.uniform(0.2, 1.0)  # Random frequency (rad/s)

        self.get_logger().info(f"Sinusoidal angular velocity: amplitude={self.angular_amplitude:.3f}, frequency={self.angular_frequency:.3f}")

    def pose_callback(self, msg):
        self.current_pose = msg

    def control_loop(self):
        if self.current_pose is None:
            return  # wait for first pose update

        # Extract current state
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        theta = self.current_pose.pose.position.z
        current_state = [x, y, theta]

        # If we have a previous state, save (state, action, next_state)
        if self.previous_state is not None and hasattr(self, 'previous_action'):
            self.state_history.append(self.previous_state)
            self.action_history.append(self.previous_action)
            self.next_state_history.append(current_state)

        # Compute time elapsed
        elapsed_time = time.time() - self.start_time

        # Sinusoidal angular velocity
        linear_velocity = np.random.uniform(low=0.05, high=0.2)
        angular_velocity = self.angular_amplitude * math.sin(self.angular_frequency * elapsed_time)

        # Publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = linear_velocity
        vel_msg.angular.z = angular_velocity
        self.publisher_.publish(vel_msg)

        # Save current action for next step
        self.previous_action = [linear_velocity, angular_velocity]
        self.previous_state = current_state

        self.step_count += 1

        if self.step_count >= self.total_steps:
            self.finish_data_collection()

    def finish_data_collection(self):
        self.get_logger().info("Finished data collection, stopping robot.")

        # Stop the robot
        stop_msg = Twist()
        self.publisher_.publish(stop_msg)

        # Prepare data
        states = np.array(self.state_history)
        actions = np.array(self.action_history)
        next_states = np.array(self.next_state_history)

        # Concatenate: [state, action, next_state]
        data = {
            'state': states,             # shape (N, 3): [x, y, theta]
            'action': actions,           # shape (N, 2): [linear_vel, angular_vel]
            'next_state': next_states    # shape (N, 3): [next_x, next_y, next_theta]
        }

        # Save to .npz
        np.savez(self.save_path, **data)

        self.get_logger().info(f"Data saved to {self.save_path}")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MDPDataCollector(total_steps=1000, save_path='mdp_data.txt')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
