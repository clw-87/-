# (專題最終版) 整合視覺雙軌巡線、盲彎記憶導航、雷達防撞、ArUco 交通號誌辨識 (紅綠燈/轉向牌)，並具備網頁即時影像回傳功能。
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import threading
from flask import Flask, Response
from rclpy.qos import qos_profile_sensor_data

# --- Flask 網頁伺服器 ---
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

@app.route('/')
def index():
    return "<html><body style='background-color:#222; color:white; text-align:center;'><h1>🏎️ 終極自駕車控制中心 (虛擬雷射防壓線系統)</h1><img src='/video_feed' style='width:100%; max-width:800px; border:3px solid #555;'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# --- ROS2 主程式 ---
class FusionDriver(Node):
    def __init__(self):
        super().__init__('fusion_driver')
        
        self.img_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.bridge = CvBridge()
        
        self.front_dist = 10.0    
        self.line_error = 0.0     
        self.line_detected = False 
        self.system_state = "CRUISING" 
        
        self.current_speed = 0.0
        self.current_steer = 0.0
        self.prev_error = 0.0
        
        self.detected_sign = "NONE"  
        self.sign_timer = 0          
        self.fork_duration = 3.0     
        
        self.last_seen_time = time.time()
        # 預設走廊寬度
        self.corridor_width = 250 
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('🚀 系統啟動：虛擬邊界探測 + 壓線微調修正')

    def scan_callback(self, msg):
        front_ranges = msg.ranges[0:30] + msg.ranges[-30:]
        valid_ranges = [r for r in front_ranges if 0.05 < r < 3.5]
        self.front_dist = min(valid_ranges) if valid_ranges else 10.0

    def image_callback(self, msg):
        global output_frame
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        height, width, _ = cv_image.shape
        debug_img = cv_image.copy()

        # ==========================================
        # 模組 1: ArUco 號誌辨識 (保持不變)
        # ==========================================
        top_img = cv_image[0:int(height/2), 0:width]
        corners, ids, rejected = aruco.detectMarkers(top_img, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            aruco.drawDetectedMarkers(debug_img, corners, ids)
            for marker_id in ids.flatten():
                if marker_id == 0: self.detected_sign = "STOP"
                elif marker_id == 1: self.detected_sign = "TURN LEFT"
                elif marker_id == 2: self.detected_sign = "TURN RIGHT"
            self.sign_timer = time.time()

        # ==========================================
        # 模組 2: 虛擬雷射探測 (Virtual Bumper)
        # ==========================================
        # 核心修改 1：只掃描最靠近車輪的底盤前方 (20% 的高度)
        scan_top = int(height * 0.75)
        scan_bottom = int(height * 0.95)
        crop_img = cv_image[scan_top:scan_bottom, 0:width]
        
        # 畫出雷射掃描區間
        cv2.rectangle(debug_img, (0, scan_top), (width, scan_bottom), (255, 255, 0), 2)
        cv2.putText(debug_img, "VIRTUAL BUMPER ZONE", (10, scan_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 100, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([15, 60, 100]), np.array([45, 255, 255]))
        
        # 核心修改 2：把黃線白線合併，顏色不重要，都是牆壁
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # 形態學除雜訊 (大掃除)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 核心修改 3：分岔路遮眼法 (直接遮掉遮罩)
        time_since_sign = time.time() - self.sign_timer
        if time_since_sign < self.fork_duration:
            if self.detected_sign == "TURN LEFT":
                mask[:, int(width*0.5):width] = 0 # 遮掉右半邊
            elif self.detected_sign == "TURN RIGHT":
                mask[:, 0:int(width*0.5)] = 0 # 遮掉左半邊

        # 核心修改 4：尋找走廊的左右邊界 (防壓線邏輯)
        # 將二值化影像垂直壓扁，看每個 X 座標是否有碰到牆壁
        projection = np.any(mask > 0, axis=0)
        
        center_x = width // 2
        left_bound = None
        right_bound = None

        # 尋找左邊界：從畫面中間往左找，碰到第一個牆壁的座標
        left_indices = np.where(projection[0:center_x])[0]
        if len(left_indices) > 0:
            left_bound = left_indices[-1] # 最靠近中心點的左側障礙物

        # 尋找右邊界：從畫面中間往右找，碰到第一個牆壁的座標
        right_indices = np.where(projection[center_x:width])[0]
        if len(right_indices) > 0:
            right_bound = right_indices[0] + center_x # 最靠近中心點的右側障礙物

        target_cx = None
        
        # 決策：根據找到的邊界計算安全中心點
        if left_bound is not None and right_bound is not None:
            # 兩邊都有牆，走正中間
            target_cx = (left_bound + right_bound) // 2
            w = right_bound - left_bound
            if 100 < w < 500: self.corridor_width = w
            
        elif left_bound is not None:
            # 只看到左邊界，保持走廊寬度一半的距離
            target_cx = left_bound + (self.corridor_width // 2)
            
        elif right_bound is not None:
            # 只看到右邊界，保持走廊寬度一半的距離
            target_cx = right_bound - (self.corridor_width // 2)

        # 在 HUD 上畫出邊界與目標
        draw_y = int((scan_top + scan_bottom) / 2)
        if left_bound:
            cv2.line(debug_img, (left_bound, scan_top), (left_bound, scan_bottom), (0, 0, 255), 4) # 左紅牆
        if right_bound:
            cv2.line(debug_img, (right_bound, scan_top), (right_bound, scan_bottom), (0, 0, 255), 4) # 右紅牆

        if target_cx is not None:
            img_center = width / 2
            self.line_error = target_cx - img_center
            self.line_detected = True
            self.last_seen_time = time.time()
            cv2.circle(debug_img, (target_cx, draw_y), 10, (0, 255, 0), -1) # 綠色目標點
        else:
            self.line_detected = False

        # ==========================================
        # 模組 3: HUD 抬頭顯示器
        # ==========================================
        overlay = debug_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, debug_img, 0.4, 0, debug_img)
        
        state_color = (0, 255, 0) if self.system_state == "CRUISING" else (0, 0, 255)
        cv2.putText(debug_img, f"SYS: {self.system_state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(debug_img, f"SPD: {self.current_speed:.2f} m/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        bar_center_x = int(width * 0.75)
        bar_y = 30
        cv2.line(debug_img, (bar_center_x - 50, bar_y), (bar_center_x + 50, bar_y), (100, 100, 100), 4)
        cv2.circle(debug_img, (bar_center_x, bar_y), 4, (255, 255, 255), -1) 
        
        steer_px = int(self.current_steer * -30) 
        steer_px = max(-50, min(50, steer_px)) 
        cv2.circle(debug_img, (bar_center_x + steer_px, bar_y), 8, (0, 255, 255), -1) 
        cv2.putText(debug_img, "STEER", (bar_center_x - 25, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if time.time() - self.sign_timer < self.fork_duration and self.detected_sign != "NONE":
            cv2.putText(debug_img, f"SIGN: {self.detected_sign}", (int(width/2)-60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        with lock: output_frame = debug_img

    # ==========================================
    # 模組 4: 決策中心 (PD控制 + 動態調速)
    # ==========================================
    def control_loop(self):
        cmd = Twist()
        
        if self.front_dist < 0.25:
            self.system_state = "OBSTACLE STOP"
            self.current_speed = 0.0
            self.current_steer = 0.0
            
        elif self.detected_sign == "STOP" and (time.time() - self.sign_timer) < 3.0:
            self.system_state = "SIGN STOP"
            self.current_speed = 0.0
            self.current_steer = 0.0
            
        elif self.line_detected:
            self.system_state = "CRUISING"
            
            error = float(self.line_error)
            derivative = error - self.prev_error 
            
            # PD 參數：因為雷射邊界反應極其靈敏，我們稍微調小靈敏度，讓車子滑順一點
            kp = 0.0035  
            kd = 0.005  
            
            steer = -(error * kp + derivative * kd)
            self.prev_error = error
            self.current_steer = steer
            cmd.angular.z = steer
            
            target_speed = max(0.08, 0.18 - abs(steer) * 0.08)
            
            if (time.time() - self.sign_timer) < self.fork_duration and self.detected_sign in ["TURN LEFT", "TURN RIGHT"]:
                target_speed = min(target_speed, 0.10)
                
            self.current_speed = target_speed
            cmd.linear.x = target_speed
                
        else:
            if (time.time() - self.last_seen_time) < 1.0: 
                self.system_state = "BLIND CORNER"
                cmd.linear.x = 0.10
                cmd.angular.z = self.current_steer 
            else:
                self.system_state = "SEARCHING"
                self.current_speed = 0.0
                self.current_steer = 0.25
                cmd.linear.x = 0.0
                cmd.angular.z = 0.25

        self.publisher_.publish(cmd)

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main(args=None):
    rclpy.init(args=args)
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    node = FusionDriver()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.publisher_.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
