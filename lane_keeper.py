# 僅使用 Lidar 雷達，計算左右障礙物距離，使車輛保持在賽道正中間。
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy
#Lidar 自動置中程式 (Wall Follower)
class LaneKeeper(Node):
    def __init__(self):
        super().__init__('lane_keeper')
        
        # 設定 QoS 以確保能收到雷達數據 (Best Effort 是常用設定)
        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_policy)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.get_logger().info('🤖 賽道置中模式啟動！(請確保跑道兩側有牆壁)')

    def scan_callback(self, msg):
        # TurtleBot3 的雷達數據 msg.ranges 是一個 360 度的陣列
        # 0度是正前方, 90度是左邊, 270度(-90)是右邊
        
        # 1. 取得左側與右側的距離 (取一個範圍的平均值比較準)
        # 左邊：約 60~120 度
        left_ranges = [r for r in msg.ranges[60:120] if r > 0.01 and r < 3.0]
        # 右邊：約 240~300 度 (對應 Python 索引 -120 到 -60)
        right_ranges = [r for r in msg.ranges[-120:-60] if r > 0.01 and r < 3.0]
        
        # 前方防撞偵測 (0~30度 和 330~360度)
        front_ranges = [r for r in (msg.ranges[0:30] + msg.ranges[-30:]) if r > 0.01 and r < 3.0]

        if not left_ranges or not right_ranges:
            return # 數據不足，不做動作

        # 計算平均距離
        left_dist = sum(left_ranges) / len(left_ranges)
        right_dist = sum(right_ranges) / len(right_ranges)
        front_dist = sum(front_ranges) / len(front_ranges) if front_ranges else 10.0

        # 2. 計算誤差 (Error)
        # 如果 左邊 > 右邊，代表車子太靠右，Error 為正，需要左轉
        # 如果 左邊 < 右邊，代表車子太靠左，Error 為負，需要右轉
        error = left_dist - right_dist
        
        # 3. PID 控制 (這裡只用 P)
        kp = 2.0  # 轉向靈敏度 (太晃就改小，轉不過就改大)
        angular_z = kp * error

        # 4. 決定前進速度
        linear_x = 0.15 # 預設速度
        
        # 如果前方很近，減速或轉彎更猛烈
        if front_dist < 0.3:
            linear_x = 0.0
            self.get_logger().warning('前方有障礙物！停車！')
        elif abs(error) > 0.3:
            linear_x = 0.05 # 偏離太遠時減速修正

        # 5. 發送指令
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.publisher_.publish(cmd)
        
        # 除錯資訊
        # print(f"左: {left_dist:.2f} | 右: {right_dist:.2f} | 誤差: {error:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = LaneKeeper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publisher_.publish(Twist()) # 停止
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
