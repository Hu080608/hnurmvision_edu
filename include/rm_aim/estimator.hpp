// 估计与预测：在「像素空间」里拟合与预测（本项目的核心设计）
//
// 为什么不用三维位置 / PnP 做瞄准量？因为瞄准真正需要的量是角度：
//   1 像素在 3 m 处 ≈ 0.044°（约 2 mm），在 8 m 处 ≈ 4.6 mm —— 始终足够准；
//   而 PnP 的"深度"在 6 m 处误差可达 0.4 m（实测 250 ms 3D RMS = 397.6 mm），完全不适合做瞄准量。
// 所以在像素平面 (u,v) 上拟合运动模型，预测"命中时刻装甲板中心会落在哪个像素"，
// 再由 (Δu, Δv) 换算成角度增量 —— 这就是「角度空间瞄准 / bearing-space aiming」。
//
// 三种模型并行 + 留出验证自动选择：
//   CV   : u,v 关于时间线性                 —— 静止靶、缓变
//   HARM : u,v = a + b cos ωt + c sin ωt  —— 平移靶（正弦往复）、旋转靶（板上点回旋）
#pragma once

#include "config.hpp"

#include <Eigen/Dense>

#include <deque>
#include <memory>
#include <vector>

namespace rmaim {

struct Observation {
  double t = 0.0;      // steady_clock 秒（图像时间）
  double az = 0.0;     // 世界系方位角（rad）—— 用【该帧时刻】的云台姿态换算
  double el = 0.0;     // 世界系俯仰角（rad）
  double range = 0.0;  // 距离（m，由框高反推，仅用于重力补偿）
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();  // 板心三维位置（世界系/云台原点，供环模型用）
  double w_px = 0.0;   // 选中板的框宽
  double w_max = 0.0;  // 本帧所有检出里的最大框宽（识别自转靶用）
  double conf = 0.0;
};

struct Prediction {
  bool valid = false;
  double az = 0.0, el = 0.0;
  double range = 0.0;
  double model_sigma = 0.0;
  double face_on = 1.0;            // 1=完全正对（仅环模型会给出 <1 的值）
  // ---- 命中时刻对齐门控（impact-aligned firing gate）----
  // 含义：把每条"看得见的"逐板轨迹外推到【子弹到达时刻 t_hit】，看它是否会经过瞄准点。
  // 只有 fire_window=true 才允许开火 —— 这才是"这一枪会不会打中"的直接判据。
  bool fire_window = true;         // 非环工况恒为 true（不影响静止/平移）
  double fire_margin = 1e9;        // 命中时刻的角度余量（rad；正=还有余量，负=已经偏了）
  double fire_tol = 0.0;           // 本次判据用的容差（rad）
  double fire_face = 0.0;          // 参与判据的那块板的正对程度（0~1）
  // 可视化用：命中时刻"最该打的那块板"的方位/俯仰/正对度（不参与控制）
  double plate_az_hit = 0.0, plate_el_hit = 0.0, plate_face = 0.0;
  bool plate_hit_ok = false;
  const char* model_name = "none";
};

class Estimator {
 public:
  explicit Estimator(const AimConfig& cfg) : cfg_(cfg) {}

  void reset();
  void add(const Observation& o);
  void add_batch(const std::vector<Observation>& batch);
  void prune(double now);
  Prediction predict(double t_hit);

  bool tracking() const { return hits_ >= 3 && !obs_.empty(); }
  // 是否处于"刚捕获/刚切换工况"的收敛期（此期间不开火、且云台步长收紧）
  bool acquiring() const { return !obs_.empty() && obs_.back().t < acquire_until_; }
  int hits() const { return hits_; }
  double last_obs_time() const { return obs_.empty() ? 0.0 : obs_.back().t; }
  const char* current_model() const { return model_name_; }
  double current_sigma() const { return sigma_; }
  double current_range() const { return range_; }
  const char* phase_diag() const { return ph_diag_; }
  // 诊断：活跃 track 数 / 最长 track 的时长与样本数
  void track_stats(int& n_active, double& longest_span, int& longest_n) const;
  // ---- 旋转工况识别（自转靶）：板在"正对↔斜视"之间循环，框宽会大幅起伏；
  //      静止靶/平移靶的框宽基本稳定。据此切换"打中心等板"策略。
  bool rotation_mode() const { return rotation_mode_; }
  double center_az() const { return center_az_; }
  double center_el() const { return center_el_; }
  double width_swing() const { return width_swing_; }
  double ring_half_width() const { return ring_half_width_; }
  // 可视化：4 块板的正对度 / 方位偏移(度) / 环相位（来自逐板轨迹；不足 4 块时其余为 0）
  void viz_faces(double face[4], double plate_deg[4], double& phase) const;
  // 诊断用：最近一次各模型的留出误差与频率
  const char* diag() const { return diag_; }

 private:
  struct Fit {
    bool ok = false;
    const char* name = "none";
    double t_ref = 0.0;
    Eigen::Vector3d cu = Eigen::Vector3d::Zero();
    Eigen::Vector3d cv = Eigen::Vector3d::Zero();
    double omega = 0.0;
    double r0 = 0.0;
    double holdout = 1e9;
  };

  Fit fit_cv(double t_now) const;
  Fit fit_harm(double t_now) const;
  Eigen::Vector2d eval(const Fit& f, double t) const;

  AimConfig cfg_;
  std::deque<Observation> obs_;
  std::vector<Observation> batch_;   // 最近一帧的全部装甲板观测（多板联合估计用）

  // ---- 逐板关联（per-plate association）：给每块物理装甲板维护一条独立轨迹 ----
  struct PlateTrack {
    bool active = false;
    double az = 0, el = 0, t = 0;      // 最近一次观测
    double vaz = 0, vel = 0;           // 角速度（EMA）
    double t_first = 0;                // 本 track 起始时间（用于诊断/拟合跨度）
    double w = 0;                      // 最近一次的框宽（正对程度的直接度量）
    double w_max = 0;                  // 本 track 见过的最大框宽（=完全正对时的宽度）
    int n = 0, misses = 0;
    // 方位历史（环形缓冲）：用于最小二乘斜率 → 命中时刻外推比 EMA 稳得多（docs/07 §25）
    static constexpr int kHistN = 20;
    double h_t[kHistN] = {0};
    double h_az[kHistN] = {0};
    int h_n = 0, h_i = 0;
  };
  static constexpr int kMaxTracks = 4;
  PlateTrack tracks_[kMaxTracks];
  void update_tracks(const std::vector<Observation>& batch, double t);
  static bool fit_track_line(const PlateTrack& tr, double& a, double& b);
  double last_batch_t_ = -1.0;   // 防止同一帧被重复处理（控制循环 1 kHz 会重复喂同一批）
  int hits_ = 0;
  const char* model_name_ = "none";
  double sigma_ = 1e3;
  double range_ = 0.0;
  bool rotation_mode_ = false;
  double ring_hold_until_ = 0.0;   // 旋转模式迟滞保持（防模式抖动）
  double ring_half_width_ = 0.0;   // 观测到的方位角摆动半宽（≈R/d）
  int ring_yes_ = 0, ring_no_ = 0;  // 旋转模式进入/退出计数（去抖）
  // ---- 环相位跟踪状态（ring_strategy=track）----
  double ring_range_med_ = 3.0;   // 观测距离中位数（自转靶分支用）
  // ---- 环相位锁相环（PLL，反小陀螺用）----
  bool pll_inited_ = false;
  double pll_psi_ = 0.0;   // 当前相位 ψ0（rad）
  double pll_omega_ = 2.0; // 角速度（rad/s，由配置文件给初值）
  double pll_t_ = 0.0;     // 上次更新时间
  double pll_resid_ema_ = 1e9;  // 锁相残差滑动平均（自校验用）
  int pll_lock_count_ = 0;      // 连续锁定帧数
  // ---- 正对板跟踪（center 策略）----
  bool cen_inited_ = false;
  double cen_d_ = 0.0, cen_v_ = 0.0, cen_t_ = 0.0;
  double grid_phi_ = 0.0, grid_err_ = 1e9;   // 整车级联合拟合的相位与残差
  // ---- 整车 EKF（详见 estimator.cpp 里的 VehicleEkf）----
  struct EkfHolder;
  std::shared_ptr<EkfHolder> ekf_;   // 用 shared_ptr：析构不需要完整类型
  void ekf_step(const std::vector<Observation>& batch);
  int ekf_matched_ = 0;      // 最近一帧 EKF 成功关联的板数（健康度）
  double ekf_resid_ = 1e9;   // 最近一次关联残差
  int ekf_lost_ = 0;         // 连续无关联帧数
  // ---- 俯仰池化谐波拟合（自转靶用）----
  double el_fit_[3] = {0, 0, 0};
  double el_fit_rms_ = 1e9;
  bool el_fit_ok_ = false;
  double obl_k_ = 0.0;
  double el_at_center_ = 0.0;    // δaz=0（过零）处的俯仰回归截距（docs/07 §27）
  double el_cross_ema_ = 0.0;    // "过零瞬间直接采样"的俯仰（避开 ±1.5° 斜视伪影）
  // ---- 两板联合的环参数估计：相邻两块可见板相差 90°，
  //      满足 (a1−az0)² + (a2−az0)² = hw² —— 一次观测就能同时约束 az0 与 hw，比"2.5 s 窗口取 min/max"稳得多（后者只用到极值两个样本、且被噪声支配）。
  bool pair_ok_ = false;
  double pair_az0_ = 0.0, pair_hw_ = 0.0, pair_t_ = -1e9;
  bool el_cross_ok_ = false;
  // ---- 抖动抑制 / 快速收敛（docs/07 §26）----
  bool ring_latched_ = false;     // 一旦确认自转工况就"锁住"，直到出现明确的平移证据
  double ring_hw_max_ = 0.0;      // 环摆动半宽的"记忆最大值"（用于几何合理性校验）
  double ring_hw_t_ = -1e9;
  double ring_enter_t_ = -1e9;    // 进入自转工况的时刻
  double ring_last_evid_ = -1e9;  // 最近一次"有自转证据"的时刻
  double acquire_until_ = -1e9;   // 收敛期截止时刻（此期间不开火）
  double resid_ema_ = 0.0;        // 观测 vs 外推 残差（EMA）
  double resid_bad_since_ = -1e9; // 残差持续超阈值的起始时刻
  void fast_ring_init(double t_now);   // 进入自转工况时用短窗口快速初始化环参数
  void soft_reset(double t_now);       // 预测持续失灵 → 丢弃历史重新捕获
  double aim_prev_az_ = 0.0, aim_prev_el_ = 0.0, aim_prev_t_ = -1e9;
  bool aim_prev_valid_ = false;
  double smooth_key_t_ = -1e9;    // 平滑只按"观测帧"做一次（控制循环 2 kHz 会重复调用）   // 斜视伪影系数 d(el)/d(δaz)
  // ---- 相位二乘拟合（phase2 策略）：u²(t)=A+B·cos2ωt+C·sin2ωt，无需板号关联 ----
  std::deque<std::pair<double, double>> u2_hist_;   // (t, u²)
  double ph_B_ = 0.0, ph_C_ = 0.0, ph_ok_ = false, ph_rms_ = 1e9;
  char ph_diag_[128] = {0};
  double el_predict(double t_hit, double t_now) const;
  bool ring_inited_ = false;
  double ring_theta_ = 0.0;   // 当前被跟踪板的环角（连续、可跨换板）
  double ring_t_ = 0.0;       // 上一次相位更新的时间
  double center_az_ = 0.0, center_el_ = 0.0, width_swing_ = 0.0;
  char diag_[220] = {0};
  double dbg_amp_ = 0.0, dbg_noise_ = 0.0;   // 谐波拟合诊断
};

bool ballistic_solution(double d, double h, double v0, double g, double& theta, double& t_f);

}  // namespace rmaim
