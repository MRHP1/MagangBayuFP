import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VisionPublisher(Node):
    def __init__(self):
        super().__init__('vision_publisher')
        self.publisher_ = self.create_publisher(Image, 'webcam', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0 / 30, self.publish_frame)  # Adjust the publishing rate as needed
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow('Webcam')  # Create a window for the webcam feed

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the frame to half
            cv2.imshow('Webcam', frame)  # Display the frame in the 'Webcam' window

            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(img_msg)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.destroy_node()
                rclpy.shutdown()
                self.cap.release()
                cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    vision_publisher = VisionPublisher()
    rclpy.spin(vision_publisher)
    

if __name__ == '__main__':
    main()
