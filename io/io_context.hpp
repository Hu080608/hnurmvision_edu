#pragma once

#include "config.hpp"
#include "types.hpp"

#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cstdint>
#include <optional>

namespace training::io {

// IO 层：把外部世界（仿真器 / 实车）翻译成算法认识的 SensorData 与 Command。
// 对外只有 Open、Read、Write、Close 四个动作。
struct IoContext {
  Config config;

  // main.cpp 用 IoContext devices{config} 构造，所以必须提供这个构造函数。
  IoContext() = default;
  explicit IoContext(const Config& cfg) : config(cfg) {}

  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();

 private:
  // 辅助函数

  // 收到消息时由 Executor 调用的回调。【名字以 on_ 开头是惯例（"当……发生时"）】
  // 只读版本，不多余复制
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  void on_feedback(const hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr& msg);

  // ROS 时间（系统时钟）换算到程序使用的 steady_clock。
  TimePoint to_steady_clock(const builtin_interfaces::msg::Time& stamp) const;

  // 当前生效的图像过期阈值（毫秒），按实测帧间隔自适应。
  double stale_ms() const;

  // ---- ROS2 资源：open() 建立，close() 释放 ----
  rclcpp::Context::SharedPtr context_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr feedback_sub_;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr command_pub_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  // ROS2 的 create_* 系列函数返回的都是 shared_ptr，因为内部还有别的持有者（ROS2 运行时自己也要管这些对象）。你没法改成 unique_ptr。
  // Executor 是我们自己 new 出来的，没有别人持有 → 用 unique_ptr 更明确："只能有一个主人，reset() 就释放"。
  
  // ---- 跨帧状态：回调写，read()/write() 读 ----
  sensor_msgs::msg::Image::ConstSharedPtr latest_image_;                   // 最新一帧图像
  hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr latest_feedback_;  // 最新一条云台反馈
  bool pending_image_ = false;  // 有一帧还没交给算法
  bool has_work_mode_ = false;  // 是否收到过工作模式
  std::uint8_t work_mode_ = 0;  // 最近一次的工作模式

  // 帧间隔滑动平均，用来算过期阈值。
  // 防止因为帧率变化造成误差
  double gap_ema_ms_ = 0;
  int gap_samples_ = 0;
  TimePoint last_arrival_{};

  // 时间换算基准：open() 时记录同一时刻的两种时钟读数。
  std::int64_t ros_base_ns_ = 0;
  std::int64_t steady_base_ns_ = 0;

  // 统计与日志节流。
  std::uint64_t ok_ = 0;
  std::uint64_t null_ = 0;
  std::uint64_t stale_ = 0;
  TimePoint last_log_{};

  bool opened_ = false;
};

}  // namespace training::io
