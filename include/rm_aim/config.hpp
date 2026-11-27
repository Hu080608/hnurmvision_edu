// 配置：yaml 读取 + 默认值 + 校验（任何一项不合理都在启动时立刻报错，而不是运行到一半行为诡异）
#pragma once

#include "lightbar.hpp"
#include "viz.hpp"

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <string>
#include <vector>

namespace rmaim {

struct CameraConfig {
  double fx = 1303.6753236814714;
  double fy = 1303.6753236814714;
  double cx = 720.0;
  double cy = 540.0;
  std::vector<double> dist;                       // 畸变系数（仿真里全 0）
  Eigen::Matrix3d R_camera2gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal = Eigen::Vector3d::Zero();

  Eigen::Matrix3d K() const {
    Eigen::Matrix3d k;
    k << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    return k;
  }
};

struct DetectorConfig {
  std::string model = "assets/armor_pose.onnx";
  int input_size = 768;
  int pad_gray = 114;          // letterbox 填充灰度（与训练时一致）
  double conf = 0.30;          // 全局置信度门槛
  double conf_small_digit = 0.30;  // 带数字的小板（1~5）门槛
  double min_box_w_px = 8.0;   // 框宽下限（原图像素），低于此视为噪声
  // ---- 仅用于"多板联合"的宽松门槛（不进主选板路径，避免影响静止/平移）----
  double conf_loose = 0.15;    // 宽松置信度：让半遮挡板/侧板也进候选
  double min_box_w_loose = 5.0;
  bool use_cuda = true;
  std::vector<int> ignore_classes{5, 6, 12, 13};  // 前哨站/基地装甲：不是自瞄目标
  // ---- ROI 二次检测（提高远距离/斜视/模糊时的检出率）----
  bool roi_enable = true;
  int roi_w = 512;        // ROI 宽（原图像素）；越小等效放大越大
  int roi_h = 384;
  int roi_full_every = 4; // 每 N 帧做一次全图检测（防止跟丢后无法重新捕获）
  double roi_margin = 90; // ROI 中心相对上次位置的余量（像素）
};

struct AimConfig {
  double pitch_zero_deg = 25.0;   // 指令 pitch=25° 时枪口水平（仿真装配决定）
  double pitch_calib_deg = 0.93;  // 安装/角点系统偏差标定常数（需实测标定）
  double bullet_speed_default = 25.0;
  double gravity = 9.81;

  double fire_angle_gate_deg = 0.60; // 预测角度误差小于它才开火
  double fire_range_gate_m = 18.0;
  double max_yaw_step_deg = 6.0;     // 单帧 yaw 指令限幅（防甩枪）
  double max_pitch_step_deg = 4.0;
  double aim_bias_u_px = 0.0;    // 瞄准像素偏置（标定用）
  double aim_bias_v_px = 32.0;   // 实测标定：3 m 处命中时板心在 v≈599
  double yaw_gain = -1.0;        // yaw 伺服方向（相机朝 -y 时为正反馈，故取负）
  double pitch_gain = 1.0;
  double fire_px_gate = 6.0;
  double ring_az_offset_deg = 0.0;  // 环中心方位角偏置（标定用）
  double lead_scale = 1.0;          // 提前量缩放：0=只瞄当前位置（零阶保持），1=全提前量
  bool ring_mode_enable = true;     // 是否启用"自转靶"策略
  double ring_radius = 0.212;       // 自转靶环半径（实测 /tf：0.212 m）
  double ring_omega = 2.0;          // 自转靶角速度（config.toml：2 rad/s）
  bool ring_predict = false;        // true=用余弦模型逐板预测(实测更差)；false=守驻留点
  // ---- 命中时刻对齐门控：只有"子弹到达时某块板正好在瞄准点"才开火 ----
  // 环工况开火模式（2档）：
  //   continuous = 默认：不做命中时刻判定，条件满足就一直发射
  //   impact     = 命中时刻对齐门控：只有"子弹到达时某块板会经过瞄准点"才开火
  //   （相位锁定点射 burst 已删除：实测明显更差）
  std::string ring_fire_mode = "continuous";
  bool ring_impact_gate = true;   // 兼容旧配置：ring_fire_mode 缺省时用它（false→continuous）
  double ring_gate_face_min = 0.55;   // 参与判据的板至少要"正对"到这个程度（框宽/历史最大框宽）
  double ring_gate_tol_scale = 1.0;   // 容差 = scale × 板半宽（1.0 = 板边，实测下限最稳）
  // ---- 落点高低测试/标定：统一俯仰微调（正=抬高枪口，因为 pitch_cmd 会减去它）----
  double pitch_trim_deg = 0.0;
  // 舵机滞后补偿：门控用"子弹到达时刻枪口会在哪"（fb + 角速度×time-to-hit）而不是"现在在哪"，
  // 让火控门限对控制循环频率/舵机滞后不敏感（docs/07 §32）
  bool servo_lead_comp = false;   // 实测：开启后射速下降（87.5 vs 92.3）
  double ring_gate_min_tol_deg = 0.35;// 容差下限（度），防止远距离时容差被算得过小
  double ring_gate_lead_ms = 0.0;      // 门控窗口的额外提前/延后（标定用，正=更早开火）
  double ring_dwell_scale = 0.0;    // 驻留点 = 中心 + scale×摆动半宽（实测最优 0.0 = 守环方位中心）
  std::string ring_strategy = "dwell";  // dwell(默认最优) | follow | phase2 | ekf | grid | zoh | center | timed | track3d
  bool use_el_harmonic = false;    // 是否用池化谐波预测俯仰（实测更差，默认关）
  double fire_obs_age_gate_ms = 200.0; // 观测太旧就不开火（原 aim.yaml 值；90 会把射速压到 7.8 发/s）
  std::string force_model = "auto";   // auto | cv | harm（消融用）
  double ring_el_offset_deg = 0.0;     // 预测残差门槛（像素）

  double lost_timeout_ms = 260.0;    // 超过这个时间没有观测 → 判定丢靶
  double hold_ms = 220.0;            // 丢靶后继续保持火控（短时预测）的时长

  // 运动模型参数
  double circle_omega_min = 0.3;     // rad/s，环模型角速度有效范围
  double circle_omega_max = 9.0;
  double radius_min = 0.08;          // m
  double radius_max = 0.45;
  double plate_width_small = 0.135;  // m，小板灯条外缘间距（PNP 用）
  double plate_height_small = 0.086;   // 实测标定（框高 42 px @2.67 m）
  double plate_width_large = 0.230;
  double plate_height_large = 0.086;
};

struct Config {
  CameraConfig camera;
  DetectorConfig detector;
  AimConfig aim;
  LightBarConfig lightbar;
  VizConfig viz;
  bool display = false;
  double latency_fixed_ms = 0.0;  // 额外固定延迟补偿（云台响应/发弹延时）


};

}  // namespace rmaim
