#include "io_context.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <thread>
#include <utility>

namespace training::io {
namespace {       // 相当于给每个函数加了 static

constexpr double kPi = 3.14159265358979323846;
// constexpr：编译期常量。编译器在编译时就把值算好并内联，运行时没有变量访问开销；
// 而且它比 const 更严格（const 只承诺"运行时不改"，constexpr 承诺"编译时已知"）。

// 弧度与角度互转：算法内部统一用 rad，ROS2 消息里是 deg。
double deg_to_rad(double deg) { return deg / 180.0 * kPi; }
double rad_to_deg(double rad) { return rad * 180.0 / kPi; }

// 云台俯仰的机械限位（度），取自仿真器 config.toml 的 gimbal_pitch_limit ≈ 0.785 rad。
constexpr double kPitchMinDeg = -45.0;
constexpr double kPitchMaxDeg = 45.0;

// read() 单次最长等待时间，保证主循环能及时响应退出。
constexpr int kReadWaitMs = 20;

// 过期阈值 = kStaleIntervals × 实测帧间隔，再夹到 [kMinStaleMs, kMaxStaleMs]。
constexpr double kStaleIntervals = 4.0;
constexpr double kMinStaleMs = 150.0;
constexpr double kMaxStaleMs = 5000.0;
constexpr double kDefaultStaleMs = 1500.0;  // 还没有帧间隔样本时的保守值

// 日志节流间隔与关停前的冲刷时间（毫秒）。
constexpr int kLogEveryMs = 1000;
constexpr int kCloseFlushMs = 100;

// roll/pitch/yaw（角度制）转四元数，旋转顺序 ZYX：q = qz * qy * qx。
Eigen::Quaterniond quat_from_rpy_deg(double roll_deg, double pitch_deg, double yaw_deg) {
  const Eigen::AngleAxisd qz(deg_to_rad(yaw_deg), Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd qy(deg_to_rad(pitch_deg), Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd qx(deg_to_rad(roll_deg), Eigen::Vector3d::UnitX());
  return (qz * qy * qx).normalized();   // 模长 = 1
}

// 把 ROS 图像消息转成 BGR、三通道、独立拥有像素内存的 cv::Mat。
cv::Mat image_to_bgr(const sensor_msgs::msg::Image& msg) {
  if (msg.data.empty() || msg.width == 0 || msg.height == 0) return {};

  const int width = static_cast<int>(msg.width);
  const int height = static_cast<int>(msg.height);
  auto* data = const_cast<unsigned char*>(msg.data.data());  // step 可能大于 width*3
                                                             // 去掉 const

  if (msg.encoding == "bgr8") {
    return cv::Mat(height, width, CV_8UC3, data, msg.step).clone();   // 深拷贝：新分配一块内存，把像素复制进去
  }
  // msg.step：每行占多少字节，可能大于 width*3（行末对齐填充）。防止逐行错位，图像变成斜的

  cv::Mat wrapped;  // 这里只是"包一层"，不拥有底层内存
  int code = -1;    // 存"转换码"，-1 表示还没定
  // 为实车作准备（不确定实车用的什么编码）
  if (msg.encoding == "rgb8") {
    wrapped = cv::Mat(height, width, CV_8UC3, data, msg.step);
    code = cv::COLOR_RGB2BGR;
  } else if (msg.encoding == "bgra8") {
    wrapped = cv::Mat(height, width, CV_8UC4, data, msg.step);
    code = cv::COLOR_BGRA2BGR;
  } else if (msg.encoding == "rgba8") {
    wrapped = cv::Mat(height, width, CV_8UC4, data, msg.step);
    code = cv::COLOR_RGBA2BGR;
  } else if (msg.encoding == "mono8") {
    wrapped = cv::Mat(height, width, CV_8UC1, data, msg.step);
    code = cv::COLOR_GRAY2BGR;
  } else {
    return {};  // 不认识的编码，这一帧作废
  }

  cv::Mat bgr;  // cvtColor 会新分配内存，因此独立拥有
  cv::cvtColor(wrapped, bgr, code);
  return bgr;
}

}  // namespace

// ---------------------------------------------------------------- 回调

// IoContext 的成员函数定义
void IoContext::on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  // 记录帧间隔，供过期阈值自适应使用。
  const auto now = Clock::now();
  if (last_arrival_.time_since_epoch().count() != 0) {
    const double gap_ms =
        std::chrono::duration<double, std::milli>(now - last_arrival_).count();
        // 两个时间点相减得到"时长"，转成毫秒的 double，再 .count() 取出数值。
    if (gap_ms > 0 && gap_ms < 10000.0) {   // 超过 10 秒视作"断流"，不参与平均（否则一次长时间暂停会把平均值拉到天上，阈值变得毫无意义）。
      gap_ema_ms_ = (gap_samples_ == 0) ? gap_ms : 0.8 * gap_ema_ms_ + 0.2 * gap_ms;
      if (gap_samples_ < 1000) ++gap_samples_;
    }
  }
  last_arrival_ = now;

  {
    std::lock_guard<std::mutex> lk(data_mu_);
    latest_image_ = msg;    // 只加引用计数，不复制像素
    pending_image_ = true;  // 标记有一帧没交给算法
  }
  data_cv_.notify_all();    // 唤醒 read()
}

void IoContext::on_feedback(
    const hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr& msg) {
  {
    std::lock_guard<std::mutex> lk(data_mu_);
    latest_feedback_ = msg;
    has_work_mode_ = true;
    work_mode_ = msg->work_mode.data;
  }
  // 按"收到时刻"存档，供 feedback_at() 插值
  FeedbackSample s;
  s.t = Clock::now();
  s.roll  = static_cast<double>(msg->roll);
  s.pitch = static_cast<double>(msg->pitch);
  s.yaw   = static_cast<double>(msg->yaw);
  s.bullet_speed = static_cast<double>(msg->bullet_speed.data);
  feedback_hist_.push_back(s);
  while (!feedback_hist_.empty() &&
         std::chrono::duration<double>(s.t - feedback_hist_.front().t).count() > 1.0)
    feedback_hist_.pop_front();
}

// ---------------------------------------------------------------- 时间
TimePoint IoContext::to_steady_clock(const builtin_interfaces::msg::Time& stamp) const {
  const std::int64_t stamp_ns = static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
                                static_cast<std::int64_t>(stamp.nanosec);
  if (stamp_ns <= 0) return Clock::now();  // 对方没填时间戳，退回“现在”
  // 兜底

  const std::int64_t offset_ns = steady_base_ns_ - ros_base_ns_;
  return TimePoint(std::chrono::nanoseconds(stamp_ns + offset_ns));
}

// 尽可能避免在不同设备上手工调参
double IoContext::stale_ms() const {
  if (gap_samples_ < 3) return kDefaultStaleMs;
  const double adaptive = kStaleIntervals * gap_ema_ms_;
  if (adaptive < kMinStaleMs) return kMinStaleMs;
  if (adaptive > kMaxStaleMs) return kMaxStaleMs;
  return adaptive;
}

// ---------------------------------------------------------------- Open

void IoContext::open() {
  if (opened_) return;

  // 1) 自己的 Context，不依赖外部是否调用过 rclcpp::init。
  // 复用主线程已经 init 过的全局 context（否则会重复初始化 rclcpp 的日志，打印一条 WARN）
  context_ = rclcpp::contexts::get_global_default_context();
  if (!context_->is_valid()) {
    context_ = std::make_shared<rclcpp::Context>();
    context_->init(0, nullptr);
  }

  // 2) 节点名 training_io，会在 ros2 node list 里看到。
  rclcpp::NodeOptions node_options;
  node_options.context(context_);
  node_ = std::make_shared<rclcpp::Node>("training_io", node_options);

  // 3) 记录时间换算基准：同一时刻的 ROS 时间与 steady_clock 读数。
  ros_base_ns_ = node_->now().nanoseconds();
  steady_base_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now().time_since_epoch())
                        .count();

  // 4) 两个订阅：传感器数据用 SensorDataQoS（尽力投递，只留最近 5 条）。
  const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(5);
  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", sensor_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(msg); });
  feedback_sub_ = node_->create_subscription<hnurm_interfaces::msg::VisionRecvData>(
      "/vision_receive_data", sensor_qos,
      [this](hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr msg) {
        on_feedback(msg);
      });

  // 5) 一个发布器：控制指令用可靠传输，不随便丢。
  command_pub_ = node_->create_publisher<hnurm_interfaces::msg::VisionSendData>(
      "/vision_send_data", rclcpp::QoS(10));

  // 6) 单线程 Executor：回调只在 read() 里 spin 时才执行。
  rclcpp::ExecutorOptions exec_options;
  exec_options.context = context_;
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(exec_options);
  executor_->add_node(node_);   // 把节点注册进去。不注册，订阅再多也不会有回调被执行。

  opened_ = true;   // 放在最后。如果中途抛异常，它仍是 false，close() 会安全地什么都不做。
  run_.store(true);
  runner_ = std::thread(&IoContext::run_loop, this);      // ★ 执行器常驻线程
  std::printf("[IO] 已连接：订阅 /image_raw、/vision_receive_data，发布 /vision_send_data"
              "（节点 training_io）\n");
  std::fflush(stdout);    // 强制把缓冲区刷出去。重定向到文件时 stdout 是块缓冲，不刷新的话日志会攒一批才出现，排查时看不到实时输出。
}

// ---------------------------------------------------------------- Read

std::optional<SensorData> IoContext::read() {
  if (!opened_ || !executor_) return std::nullopt;    // 任一为空就立刻返回 nullopt（没打开就不可能有数据）

  // 阶段一：等待"新的一帧图像 + 至少一条反馈"。
  // 注意：这里不再自己 spin 执行器——执行器由 runner_ 线程常驻 spin（见 open()）。
  // 否则"第二线程发布/计算"会把 image 回调饿死（实测检测线程只跑完 1 帧）。
  sensor_msgs::msg::Image::ConstSharedPtr image;
  hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr feedback;
  {
    std::unique_lock<std::mutex> lk(data_mu_);
    const bool ok = data_cv_.wait_for(
        lk, std::chrono::milliseconds(kReadWaitMs * 3),
        [this] { return pending_image_ && latest_image_ && latest_feedback_; });
    if (!ok) {
      ++null_;
      return std::nullopt;
    }
    image = std::move(latest_image_);       // 转移引用计数，避免重复交付同一帧
    feedback = latest_feedback_;            // 反馈要一直留着复用，不 move
    pending_image_ = false;  // 同一图像只交给算法一次
  }   // lk 作用域结束
  const double stale_limit = stale_ms();

  // 阶段三：过期检查。
  const std::int64_t stamp_ns = static_cast<std::int64_t>(image->header.stamp.sec) *
                                    1000000000LL +
                                static_cast<std::int64_t>(image->header.stamp.nanosec);
  if (stamp_ns > 0) {
    const std::int64_t age_ns = node_->now().nanoseconds() - stamp_ns;    // age_ns：这一帧从"拍摄"到"现在"过了多久
    if (age_ns > static_cast<std::int64_t>(stale_limit * 1000000.0)) {
      ++stale_;
      ++null_;
      return std::nullopt;
    }
  }

  // 阶段四：填 SensorData。
  SensorData data;
  data.image = image_to_bgr(*image);
  if (data.image.empty() || data.image.type() != CV_8UC3) {   // 再检查一次类型
    ++null_;
    return std::nullopt;
  }

  data.timestamp = to_steady_clock(image->header.stamp);
  // 时间对齐：用"图像拍摄时刻"的云台姿态，而不是"现在最新的"姿态
  const FeedbackSample att = feedback_at(data.timestamp);
  data.imu_q = quat_from_rpy_deg(att.roll, att.pitch, att.yaw);
  data.bullet_speed = static_cast<double>(feedback->bullet_speed.data);  // 嵌套消息要取 .data
  ++ok_;  // 在四个字段都填好之后才计数——前面任何一条失败路径都走不到这里，所以 ok_ 精确表示"成功交付了多少帧"。

  // 阶段五：每秒打印一次实时姿态与弹速（讲义 §5.7 的提交要求）。
  const auto now = Clock::now();
  if (last_log_.time_since_epoch().count() == 0 ||    // 还没打印过（第一次）
      now - last_log_ >= std::chrono::milliseconds(kLogEveryMs)) {    // 距上次打印超过 1 秒
    last_log_ = now;
    std::printf("[IO] yaw=%.2f deg pitch=%.2f deg bullet_speed=%.1f m/s\n",
                static_cast<double>(feedback->yaw),
                static_cast<double>(feedback->pitch), data.bullet_speed);
    std::fflush(stdout);
  }

  return data;
}

// ---------------------------------------------------------------- Write

void IoContext::write(const Command& command) {
  std::lock_guard<std::mutex> lk(cmd_mu_);
  cmd_latest_ = command;          // ★ 只更新"最新指令"，真正发布交给 2 kHz 线程
}

// 执行器常驻线程：只负责 spin，让 ROS 回调持续推进（与 read()/控制线程解耦）
void IoContext::run_loop() {
  while (run_.load()) {
    try {
      executor_->spin_some(std::chrono::milliseconds(2));
    } catch (const std::exception&) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void IoContext::publish_loop() {
  while (pub_run_.load()) {
    Command c{};
    if (aim_fn) {
      c = aim_fn();               // 高频重算瞄准（主循环只提供"最新观测"）
    } else {
      std::lock_guard<std::mutex> lk(cmd_mu_);
      c = cmd_latest_;
    }
    publish_now(c);
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
}

void IoContext::publish_now(const Command& command) {
  if (!opened_ || !command_pub_) return;

  hnurm_interfaces::msg::VisionSendData msg{};    // 值初始化——把所有字段清零
  msg.header.stamp = node_->now();  // header 用 ROS 时间
  msg.header.frame_id = "gimbal_link";

  // 把工作模式拷到局部变量，后面直接用，读起来比 work_mode_ 更清楚。
  const std::uint8_t mode = work_mode_;
  const bool has_mode = has_work_mode_;

  // 安全门控：任何一项不满足就发停止。
  const double yaw_deg = rad_to_deg(command.yaw);
  const double pitch_deg = rad_to_deg(command.pitch);
  const bool finite = std::isfinite(yaw_deg) && std::isfinite(pitch_deg);
  const bool in_limit = pitch_deg >= kPitchMinDeg && pitch_deg <= kPitchMaxDeg;
  const bool auto_aim = has_mode && mode == hnurm_interfaces::msg::WorkMode::AUTO_AIM;
  const bool want_control = command.control && finite && in_limit && auto_aim;

  if (!want_control) {
    // 停止：只靠 target_state=0 表达；target_distance 按讲义 5.4 填 0。
    msg.target_state.data = hnurm_interfaces::msg::TargetState::LOST_TARGET;  // 0
    msg.target_distance = 0.0F;
    command_pub_->publish(msg);
    return;
  }

  msg.target_type.data = hnurm_interfaces::msg::TargetType::NONE;
  msg.yaw = static_cast<float>(yaw_deg);
  msg.pitch = static_cast<float>(pitch_deg);
  msg.target_distance = 0.0F;  // 未接测距填 0

  const bool want_fire = config.allow_fire && command.shoot;
  msg.target_state.data = want_fire ? hnurm_interfaces::msg::TargetState::FIRE          // 2
                                    : hnurm_interfaces::msg::TargetState::CONVERGING;  // 1
  command_pub_->publish(msg);
}

// ---------------------------------------------------------------- Close

void IoContext::close() {
  if (!opened_) return;

  { std::lock_guard<std::mutex> lk(cmd_mu_); cmd_latest_ = Command{}; }   // 先让云台停下
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  pub_run_.store(false);
  if (pub_thread_.joinable()) pub_thread_.join();
  run_.store(false);
  if (runner_.joinable()) runner_.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(kCloseFlushMs));    // 异步处理，发完等100ms，确保真正发出去

  if (executor_ && node_) {
    try {
      executor_->remove_node(node_);    // 先从调度器上摘下来，再销毁节点
    } catch (const std::exception&) {   // 节点已经被关掉时 remove_node 可能抛异常，这里忽略
    }
  }
  executor_.reset();
  command_pub_.reset();
  image_sub_.reset();
  feedback_sub_.reset();
  node_.reset();

  if (context_) {
    if (context_->is_valid()) context_->shutdown("training io close");
    context_.reset();
  }

  const std::uint64_t ok = ok_;
  const std::uint64_t nul = null_;
  const std::uint64_t st = stale_;
  latest_image_.reset();
  latest_feedback_.reset();
  pending_image_ = false;
  has_work_mode_ = false;

  std::printf("[IO] 已断开：成功=%llu 空=%llu 过期=%llu\n",
              static_cast<unsigned long long>(ok), static_cast<unsigned long long>(nul),
              static_cast<unsigned long long>(st));
  std::fflush(stdout);

  opened_ = false;
}

IoContext::FeedbackSample IoContext::feedback_at(TimePoint t) const {
  std::lock_guard<std::mutex> lk(data_mu_);
  if (feedback_hist_.empty()) {
    FeedbackSample s;
    if (latest_feedback_) {                       // 兜底：还没有历史就用最新一条
      s.roll  = static_cast<double>(latest_feedback_->roll);
      s.pitch = static_cast<double>(latest_feedback_->pitch);
      s.yaw   = static_cast<double>(latest_feedback_->yaw);
      s.bullet_speed = static_cast<double>(latest_feedback_->bullet_speed.data);
    }
    return s;
  }
  if (t <= feedback_hist_.front().t) return feedback_hist_.front();
  if (t >= feedback_hist_.back().t)  return feedback_hist_.back();
  for (size_t i = 1; i < feedback_hist_.size(); ++i) {
    const auto& b = feedback_hist_[i];
    const auto& a = feedback_hist_[i - 1];
    if (t <= b.t) {
      const double span = std::chrono::duration<double>(b.t - a.t).count();
      const double w = span > 1e-9 ? std::chrono::duration<double>(t - a.t).count() / span : 1.0;
      FeedbackSample s;
      s.t = t;
      s.roll  = a.roll  + w * (b.roll  - a.roll);
      s.pitch = a.pitch + w * (b.pitch - a.pitch);
      s.yaw   = a.yaw   + w * (b.yaw   - a.yaw);
      s.bullet_speed = a.bullet_speed + w * (b.bullet_speed - a.bullet_speed);
      return s;
    }
  }
  return feedback_hist_.back();
}

}  // namespace training::io
