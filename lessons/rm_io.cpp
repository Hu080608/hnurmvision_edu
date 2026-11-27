#include "rm_aim/io.hpp"

#include <rclcpp/qos.hpp>

#include <chrono>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace rmaim {
// 静默开关（默认静默，保持仓库原有控制台输出；RMAIM_VERBOSE=1 可打开本栈日志）
bool rmaim_verbose();
namespace {
double steady_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

Io::Io() : t0_(steady_sec()) {}

Io::~Io() { stop(); }

void Io::start() {
  node_ = std::make_shared<rclcpp::Node>("rm_aim");
  auto sensor_qos = rclcpp::QoS(rclcpp::SensorDataQoS());

  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", sensor_qos, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(m); });
  fb_sub_ = node_->create_subscription<hnurm_interfaces::msg::VisionRecvData>(
      "/vision_receive_data", rclcpp::QoS(rclcpp::SensorDataQoS()),
      [this](hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr m) { on_feedback(m); });
  cmd_pub_ = node_->create_publisher<hnurm_interfaces::msg::VisionSendData>(
      "/vision_send_data", rclcpp::QoS(rclcpp::SensorDataQoS()));

  exec_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  exec_->add_node(node_);
  running_ = true;
  spin_thread_ = std::thread([this] {
    while (running_) {
      exec_->spin_some(std::chrono::milliseconds(2));
    }
  });
  if (rmaim_verbose()) std::printf("[io] 已连接：订阅 /image_raw、/vision_receive_data，发布 /vision_send_data\n");
}

void Io::stop() {
  if (!running_.exchange(false)) return;
  if (spin_thread_.joinable()) spin_thread_.join();
  if (exec_) exec_->remove_node(node_);
  exec_.reset();
  image_sub_.reset();
  fb_sub_.reset();
  cmd_pub_.reset();
  node_.reset();
}

void Io::on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!msg || msg->data.empty() || msg->width == 0 || msg->height == 0) return;
  // 换仿真器后编码/尺寸可能变化：一律按"不认识就丢弃"，绝不因为格式不符而崩溃
  if (msg->step < msg->width * 3 && msg->encoding != "mono8") return;
  const int w = static_cast<int>(msg->width), h = static_cast<int>(msg->height);
  auto* data = const_cast<unsigned char*>(msg->data.data());
  cv::Mat wrapped(h, w, CV_8UC3, data, msg->step);
  cv::Mat bgr;
  if (msg->encoding == "rgb8") {
    cv::cvtColor(wrapped, bgr, cv::COLOR_RGB2BGR);
  } else if (msg->encoding == "bgr8") {
    bgr = wrapped.clone();
  } else if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
    cv::cvtColor(cv::Mat(h, w, CV_8UC1, data, msg->step), bgr, cv::COLOR_GRAY2BGR);
  } else if (msg->encoding == "bgra8") {
    cv::cvtColor(cv::Mat(h, w, CV_8UC4, data, msg->step), bgr, cv::COLOR_BGRA2BGR);
  } else if (msg->encoding == "rgba8") {
    cv::cvtColor(cv::Mat(h, w, CV_8UC4, data, msg->step), bgr, cv::COLOR_RGBA2BGR);
  } else {
    return;
  }

  std::lock_guard<std::mutex> lk(frame_mu_);
  if (frame_pending_) ++frames_dropped_;   // 上一帧还没被处理就被覆盖
  latest_.bgr = std::move(bgr);
  latest_.stamp_sec = steady_sec();
  latest_.stamp_ros = static_cast<double>(msg->header.stamp.sec) +
                      static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
  latest_.seq = ++frames_in_;
  frame_pending_ = true;
}

void Io::on_feedback(const hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr& msg) {
  std::lock_guard<std::mutex> lk(fb_mu_);
  fb_.yaw_deg = msg->yaw;
  fb_.pitch_deg = msg->pitch;
  fb_.roll_deg = msg->roll;
  fb_.bullet_speed = msg->bullet_speed.data > 0 ? static_cast<double>(msg->bullet_speed.data)
                                                : 25.0;
  fb_.work_mode = static_cast<int>(msg->work_mode.data);
  fb_.self_color = static_cast<int>(msg->self_color.data);
  fb_.stamp_sec = steady_sec();
  fb_.valid = true;
  fb_hist_.push_back({fb_.stamp_sec, fb_.yaw_deg, fb_.pitch_deg, fb_.roll_deg, fb_.bullet_speed});
  while (fb_hist_.size() > 400) fb_hist_.pop_front();
}

bool Io::take_frame(Frame& out) {
  std::lock_guard<std::mutex> lk(frame_mu_);
  if (!frame_pending_) return false;
  out = latest_;
  frame_pending_ = false;
  return true;
}

Feedback Io::feedback() const {
  std::lock_guard<std::mutex> lk(fb_mu_);
  return fb_;
}

void Io::publish(const Command& cmd) {
  if (!cmd_pub_) return;
  hnurm_interfaces::msg::VisionSendData m;
  m.header.stamp = node_->now();
  m.header.frame_id = "aim";
  // target_state: 0=LOST_TARGET 1=CONVERGING 2=FIRE
  m.target_state.data = cmd.fire ? 2 : (cmd.control ? 1 : 0);
  m.yaw = static_cast<float>(cmd.yaw_deg);
  m.pitch = static_cast<float>(cmd.pitch_deg);
  m.target_distance = static_cast<float>(cmd.target_distance);
  m.delta_yaw = 0.0f;
  m.delta_pitch = 0.0f;
  m.vel_x = static_cast<float>(cmd.vel_x);
  m.vel_y = static_cast<float>(cmd.vel_y);
  m.vel_yaw = static_cast<float>(cmd.vel_yaw);
  m.control_id = 0.0f;
  m.original_yout = static_cast<float>(cmd.yaw_deg);
  cmd_pub_->publish(m);
}

Feedback Io::feedback_at(double t) const {
  std::lock_guard<std::mutex> lk(fb_mu_);
  if (fb_hist_.empty()) return fb_;
  // 找相邻两点线性插值
  const FbSample* a = nullptr;
  const FbSample* b = nullptr;
  for (size_t i = 1; i < fb_hist_.size(); ++i) {
    if (fb_hist_[i - 1].t <= t && t <= fb_hist_[i].t) { a = &fb_hist_[i - 1]; b = &fb_hist_[i]; break; }
  }
  Feedback out = fb_;
  if (a && b && b->t > a->t) {
    const double w = (t - a->t) / (b->t - a->t);
    out.yaw_deg = a->yaw + (b->yaw - a->yaw) * w;
    out.pitch_deg = a->pitch + (b->pitch - a->pitch) * w;
    out.roll_deg = a->roll + (b->roll - a->roll) * w;
    out.bullet_speed = a->bullet;
    out.valid = true;
  }
  return out;
}

double Io::now_sec() const { return steady_sec(); }

}  // namespace rmaim
