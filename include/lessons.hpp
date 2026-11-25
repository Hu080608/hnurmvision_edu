#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training {
std::vector<Armor> lesson03_detection(const cv::Mat& image);
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state);
// 第四讲的绘图接口（只负责把结果画到图像上）
void lesson04_display(cv::Mat image, const TrackingState& state);
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed);
// imu_q    = IO 层由 yaw/pitch/roll 重建的四元数（本帧姿态）
// R_world0 = 上电第一帧的姿态矩阵（世界系基准，由第四讲 ArmorSolver 记录）
// 每帧注入"本帧云台姿态"与"上电基准姿态"
void lesson05_set_attitude(const Eigen::Quaterniond& imu_q, const Eigen::Matrix3d& R_world0);
}  // namespace training
