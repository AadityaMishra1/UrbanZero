#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
import os, sys

CARLA_HOST = os.environ.get("CARLA_HOST", "172.25.176.1")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))

carla_path = "/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla"
if carla_path not in sys.path:
    sys.path.insert(0, carla_path)
import carla

class UrbanZeroNode(Node):
    def __init__(self):
        super().__init__("urbanzero_node")
        self.image_pub = self.create_publisher(Image, "/urbanzero/camera/semantic", 10)
        self.control_pub = self.create_publisher(Twist, "/urbanzero/vehicle/control", 10)
        self.speed_pub = self.create_publisher(Float32, "/urbanzero/vehicle/speed", 10)
        self.status_pub = self.create_publisher(String, "/urbanzero/status", 10)
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.timer = self.create_timer(0.05, self.publish_state)
        self.get_logger().info("UrbanZero ROS2 node started")
        self.get_logger().info("Topics: /urbanzero/camera/semantic, /urbanzero/vehicle/control, /urbanzero/vehicle/speed")

    def publish_state(self):
        try:
            vehicles = self.world.get_actors().filter("vehicle.tesla.model3")
            if not vehicles:
                return
            vehicle = list(vehicles)[0]
            vel = vehicle.get_velocity()
            speed = float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5)
            speed_msg = Float32()
            speed_msg.data = speed
            self.speed_pub.publish(speed_msg)
            control = vehicle.get_control()
            twist = Twist()
            twist.linear.x = float(control.throttle - control.brake)
            twist.angular.z = float(control.steer)
            self.control_pub.publish(twist)
            status = String()
            status.data = f"speed={speed:.2f}m/s steer={control.steer:.3f} throttle={control.throttle:.3f}"
            self.status_pub.publish(status)
        except Exception as e:
            self.get_logger().warn(f"Publish error: {e}")

def main():
    rclpy.init()
    node = UrbanZeroNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
