// IO 层：把 ROS2 世界翻译成「最新一帧图像 + 最新云台反馈」，并把决策写回仿真器。
// 设计要点：
//   1) 采集线程只做拷贝，不做任何计算 —— 保证取图不被下游拖慢（端到端延迟最小化）
//   2) 只保留"最新一帧"，下游处理不过来就丢旧帧（实时系统宁可丢掉旧数据，也不排队）
//   3) 时间戳全部换算到 steady_clock，用的就是「图像到达到处理」的真实延迟
#pragma once

#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>

#include <atomic>
#include <deque>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

namespace rmaim {

struct Frame {
  cv::Mat bgr;        // CV_8UC3
  double stamp_sec = 0.0;   // 回调时刻（steady_clock 秒）
  double stamp_ros = 0.0;   // 消息头时间戳（仿真器采集时刻，ROS 系统时间秒）
  uint64_t seq = 0;
  bool valid() const { return !bgr.empty(); }
};

struct Feedback {
  double yaw_deg = 0.0;
  double pitch_deg = 0.0;
  double roll_deg = 0.0;
  double bullet_speed = 25.0;
  int work_mode = 0;   // 0 = AUTO_AIM
  int self_color = 0;
  double stamp_sec = 0.0;
  bool valid = false;
};

struct Command {
  bool control = false;
  bool fire = false;
  double yaw_deg = 0.0;
  double pitch_deg = 0.0;
  double target_distance = 0.0;
  double vel_x = 0.0, vel_y = 0.0, vel_yaw = 0.0;
};

class Io {
 public:
  Io();
  ~Io();

  void start();                       // 建节点、起自旋线程
  void stop();

  bool take_frame(Frame& out);         // 有未处理的新帧则取走（返回 true）
  Feedback feedback() const;
  // 取 t 时刻（steady_clock 秒）的云台姿态：线性插值，解决"图像比反馈旧 20~30 ms"的对齐问题
  Feedback feedback_at(double t) const;
  void publish(const Command& cmd);

  double now_sec() const;              // steady_clock 秒
  uint64_t frames_received() const { return frames_in_; }

 private:
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  void on_feedback(const hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr& msg);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr fb_sub_;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr cmd_pub_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> exec_;
  std::thread spin_thread_;

  mutable std::mutex frame_mu_;
  Frame latest_;
  bool frame_pending_ = false;
  uint64_t frames_in_ = 0;
  uint64_t frames_dropped_ = 0;

  mutable std::mutex fb_mu_;
  Feedback fb_;
  struct FbSample { double t, yaw, pitch, roll, bullet; };
  mutable std::deque<FbSample> fb_hist_;

  std::atomic<bool> running_{false};
  double t0_ = 0.0;
};

}  // namespace rmaim
