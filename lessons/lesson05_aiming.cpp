// ----------------------------------------------------------------------------
// 输入格式：
//   target       : const Target&；position/velocity 单位 m、m/s，
//                  位于第四讲给出的 IMU 稳定系（x 前 / y 左 / z 上，原点 = 上电时刻云台旋转中心）。
//   timestamp    : TimePoint，target 对应的【图像采集时刻】。
//   bullet_speed : 弹丸初速度 m/s。
// 额外输入（本讲新增，由 main.cpp 每帧调用 lesson05_set_attitude 喂入）：
//   imu_q        : 本帧云台姿态（= IO 层由 yaw/pitch/roll 重建的四元数）。
//   R_world0     : 上电第一帧的姿态矩阵（第四讲 ArmorSolver 里记录的"世界系基准"）。
// 输出：
//   Command{control, shoot, yaw, pitch}；yaw/pitch 为【度】，世界系绝对角。
// ============================================================================
// 第五讲：弹道解算
//
// 输入格式：
//   target       : const Target&；position 单位 m，velocity 单位 m/s，二者均采用
//                  第四讲规定的 x 向前、y 向左、z 向上坐标系。
//   timestamp    : TimePoint，target 对应的图像时刻，用于计算预测时间。
//   bullet_speed : double，弹丸初速度，单位 m/s，必须为有限正数。
// 输出格式：
//   Command：
//   - control : true 表示请求云台跟随；无法产生安全解时必须为 false；
//   - shoot   : true 表示请求发射，最终仍受 IO 层 allow_fire 和有效性门控；
//   - yaw     : 世界/IMU 参考系下的绝对偏航角，单位 rad；
//   - pitch   : 世界/IMU 参考系下的绝对俯仰角，单位 rad。
// 验收：仿真器完成静止靶、平移靶任务。

#include "lessons.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace training {
namespace {

// ---------------------------------------------------------------- 物理与零位常数
constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;
constexpr double kGravity = 9.81;        // 重力加速度 m/s²
constexpr double kPitchZeroDeg = 25.0;   // 零位：指令 pitch=25° 时枪口水平
// 安装偏差 + 光学/枪口杆臂合成的标定常数（度）
constexpr double kPitchCalibDeg = 0.93;

// 发弹延迟（秒）：电控响应 + 摩擦轮起转
constexpr double kFireDelaySec = 0.0;

// ---------------------------------------------------------------- 门限
constexpr double kMinDistanceM = 0.3;    // 太近：几何退化 + 撞枪口，不解算
constexpr double kMaxDistanceM = 20.0;   // 太远：解不出 / 精度不够
constexpr double kMaxFlightTimeSec = 1.0;// 飞行时间上限（超过说明打不到）
constexpr double kIterTolSec = 1e-4;     // 不动点迭代的收敛判据（0.1 ms）
constexpr int kIterMax = 6;              // 迭代次数上限（横向靶 2 次就收敛）
constexpr double kPitchLimitDeg = 42.0;  // 云台俯仰软限位（仿真器硬限位 ±45°）
// 火控收敛门限（度）：装甲板在 3 m 处的角尺寸 ≈ 2.6°(宽) × 1.1°(高)，所以 yaw 允许比 pitch 松一倍。门限 = 角尺寸的一半 是"打板中心"的经验取值。
constexpr double kSettleYawTolDeg = 0.60;    // yaw：|实测 − 指令| < 0.60°
constexpr double kSettlePitchTolDeg = 0.30;  // pitch：|实测 − 指令| < 0.30°
constexpr double kMaxYawRateDeg = 120.0; // 目标角速度超过它就先不开火（防止甩枪）
constexpr double kMaxYawStepDeg = 6.0;   // 单帧 yaw 指令最大跳变（防坏估计把云台甩出视野）

// ---------------------------------------------------------------- 跨帧状态
Eigen::Matrix3d g_R_world0 = Eigen::Matrix3d::Identity();  // 上电基准姿态
Eigen::Matrix3d g_R_imu = Eigen::Matrix3d::Identity();     // 本帧云台姿态
bool g_ready = false;                                     // 是否已喂入过姿态
double g_last_yaw_cmd = 0.0;                              // 上一帧指令（角速度门限用）
bool g_has_last = false;
TimePoint g_last_time{};
bool g_has_last_time = false;

// ---------------------------------------------------------------- 弹道解算
// 给定弹速 v0、水平距离 d、相对高度 h，返回枪口仰角 theta_rad（抬头为正）与飞行时间 tf_sec。
// a·tan²θ − d·tanθ + (a+h) = 0，a = g d² / (2 v0²)
bool solve_ballistic(double v0, double d, double h, double* theta_rad, double* tf_sec) {
  if (!(v0 > 1.0) || !(d > 1e-3)) return false;
  const double a = kGravity * d * d / (2.0 * v0 * v0);
  const double disc = d * d - 4.0 * a * (a + h);
  if (disc < 0.0) return false;                 // 打不到（太远 / 太高 / 弹速太低）
  const double tan_low = (d - std::sqrt(disc)) / (2.0 * a);   // 低伸弹道（取负号解）
  const double theta = std::atan(tan_low);
  const double tf = d / (v0 * std::cos(theta));
  if (!(tf > 0.0) || tf > kMaxFlightTimeSec) return false;
  *theta_rad = theta;
  *tf_sec = tf;
  return true;
}
}  // namespace

// ---------------------------------------------------------------- 姿态注入
// main.cpp 每帧调用：把"本帧云台姿态"和"上电基准姿态"交给第五讲。
void lesson05_set_attitude(const Eigen::Quaterniond& imu_q, const Eigen::Matrix3d& R_world0) {
  g_R_imu = imu_q.normalized().toRotationMatrix();
  g_R_world0 = R_world0;
  g_ready = true;
}

// ---------------------------------------------------------------- 主函数
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed) {
  Command command;                      // 默认 control=false / shoot=false（最安全）
  if (!g_ready) return command;         // 还没拿到姿态：本帧什么都不做
  const double v0 = bullet_speed;
  if (!(v0 > 1.0) || !std::isfinite(v0)) return command;

  // ⓪ 距离门控：太近几何退化、太远解不出（第四讲 PnP 侧另有 0.5~12 m 的粗略门限，这里再兜一层）
  const double range = target.position.norm();
  if (!(range >= kMinDistanceM) || range > kMaxDistanceM) return command;

  // ① 观测时刻 → 处理时刻的延迟 Δt₁（用同一个 steady_clock 直接量）
  const double dt1 = std::chrono::duration<double>(Clock::now() - timestamp).count();
  if (!(dt1 >= 0.0) || dt1 > 0.5) return command;   // 时间戳异常（过期帧）直接丢

  // ② 迭代求"命中点 + 飞行时间"（两者互相依赖 → 不动点迭代，见讲义第 05 章）
  Eigen::Vector3d p_hit = target.position;
  double tf = 0.0;
  double theta = 0.0;
  bool solved = false;
  for (int i = 0; i < kIterMax; ++i) {
    const Eigen::Vector3d lead_time_v = target.velocity * (dt1 + kFireDelaySec + tf);
    p_hit = target.position + lead_time_v;

    // 命中点 → 世界系单位方向（R_world0 把"上电时刻云台系"转成"世界系"）
    const Eigen::Vector3d e = (g_R_world0 * p_hit).normalized();
    const double d_horizontal = std::sqrt(e.x() * e.x() + e.y() * e.y()) * p_hit.norm();
    const double h_vertical = e.z() * p_hit.norm();

    double theta_new = 0.0;
    double tf_new = 0.0;
    if (!solve_ballistic(v0, d_horizontal, h_vertical, &theta_new, &tf_new)) {
      solved = false;
      break;                            // 无解：本帧不控制
    }
    solved = true;
    theta = theta_new;
    const double diff = std::fabs(tf_new - tf);
    tf = tf_new;
    if (diff < kIterTolSec) break;      // 收敛
  }
  if (!solved) return command;

  // ③ 世界系方向 → 绝对角指令（讲义第 03 章 §3.5 的完整推导）
  const Eigen::Vector3d e = (g_R_world0 * p_hit).normalized();
  const double yaw_deg = std::atan2(e.y(), e.x()) * kRad2Deg;
  const double pitch_deg =
      kPitchZeroDeg - theta * kRad2Deg - kPitchCalibDeg;   // pitch = 25° − θ − 标定

  // ④ 软限位 / 跳变限幅 / 数值安全
  if (!std::isfinite(yaw_deg) || !std::isfinite(pitch_deg)) return command;
  if (std::fabs(pitch_deg) > kPitchLimitDeg) return command;
  double yaw_out = yaw_deg;
  if (g_has_last) {                       // 单帧跳变限幅：坏估计不能把云台甩出视野
    const double d = std::remainder(yaw_deg - g_last_yaw_cmd, 360.0);
    if (std::fabs(d) > kMaxYawStepDeg)
      yaw_out = g_last_yaw_cmd + (d > 0.0 ? kMaxYawStepDeg : -kMaxYawStepDeg);
  }

  command.control = true;
  command.yaw = yaw_out * kDeg2Rad;
  command.pitch = pitch_deg * kDeg2Rad;

  // ⑤ 火控：只有"云台已经到位"才请求开火
  //    imu_q 是 Rz(yaw)·Ry(pitch)·Rx(roll) 的旋转矩阵，反解：
  //      yaw = atan2(R10, R00)，pitch = −asin(R20)
  //    注意：写成 R(2,1) 会恒等于 0（pitch 只出现在第 2 行第 0 列）。
  const Eigen::Matrix3d& R = g_R_imu;
  const double fb_yaw = std::atan2(R(1, 0), R(0, 0)) * kRad2Deg;
  const double fb_pitch = -std::asin(std::clamp(R(2, 0), -1.0, 1.0)) * kRad2Deg;
  const double yaw_err = std::fabs(std::remainder(fb_yaw - yaw_out, 360.0));
  const double pitch_err = std::fabs(fb_pitch - pitch_deg);

  // 目标角速度门限：太快（刚跳靶/刚捕获）先别开火
  double yaw_rate = 0.0;
  if (g_has_last && g_has_last_time) {
    const double dt = std::chrono::duration<double>(timestamp - g_last_time).count();
    if (dt > 1e-4)
      yaw_rate = std::fabs(std::remainder(yaw_deg - g_last_yaw_cmd, 360.0)) / dt;
  }
  g_last_yaw_cmd = yaw_out;
  g_last_time = timestamp;
  g_has_last = true;
  g_has_last_time = true;

  command.shoot = (yaw_err < kSettleYawTolDeg) && (pitch_err < kSettlePitchTolDeg) &&
                  (yaw_rate < kMaxYawRateDeg) && (tf > 0.0);
  return command;
}
}  // namespace training
