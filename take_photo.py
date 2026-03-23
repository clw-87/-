# 單次擷取樹梅派相機畫面並存檔，用於校正 HSV 顏色範圍與相機視角。
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class PhotoTaker(Node):
    def __init__(self):
        super().__init__('photo_taker')
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.callback, 10)
        self.bridge = CvBridge()
        self.got_image = False

    def callback(self, msg):
        if self.got_image: return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 存檔到你的工作區根目錄
            filename = '/home/ubuntu/debug_view.jpg'
            cv2.imwrite(filename, cv_img)
            self.get_logger().info(f'📸 成功拍照！已存檔至: {filename}')
            self.got_image = True
            # 拍完就關閉程式
            raise SystemExit
        except Exception as e:
            self.get_logger().error(str(e))

def main():
    rclpy.init()
    node = PhotoTaker()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
