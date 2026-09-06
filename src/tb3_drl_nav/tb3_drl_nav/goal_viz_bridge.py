#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

class GoalBridge(Node):
    """
    Bridges the text-based goal topic (/tb3_drl/goal) 
    to a PointStamped topic for RViz visualization.
    """
    def __init__(self):
        super().__init__('goal_viz_bridge')
        self.pub = self.create_publisher(PointStamped, '/goal_point', 10)
        self.sub = self.create_subscription(String, '/tb3_drl/goal', self.cb, 10)
        self.get_logger().info("Goal Viz Bridge started. Monitoring /tb3_drl/goal...")

    def cb(self, msg):
        try:
            # Parse 'x,y' string
            x, y = [float(v) for v in msg.data.split(',')]
            
            p = PointStamped()
            p.header.frame_id = 'odom'
            p.header.stamp = self.get_clock().now().to_msg()
            p.point.x = x
            p.point.y = y
            p.point.z = 0.01  # Slightly above ground
            
            self.pub.publish(p)
        except Exception as e:
            self.get_logger().error(f"Failed to parse goal: {msg.data} ({e})")

def main():
    rclpy.init()
    node = GoalBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
