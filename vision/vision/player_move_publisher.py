import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_publisher_node')
        self.publisher_ = self.create_publisher(Image, 'webcam_topic', 10)
        self.timer = self.create_timer(1.0 / 30, self.publish_frame)  # Adjust frame rate as needed
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # Initialize webcam capture

    def publish_frame(self):
        ret, frame = self.cap.read()  # Read frame from webcam
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')  # Convert frame to ROS Image message
            self.publisher_.publish(msg)  # Publish frame

def main(args=None):
    rclpy.init(args=args)
    webcam_publisher_node = WebcamPublisherNode()
    rclpy.spin(webcam_publisher_node)
    webcam_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
