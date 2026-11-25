#pragma once
// ============================================================================
// lesson04_tracker.hpp —— 选板 / 数据关联 / 状态机 / 预测 / 误差统计
// ----------------------------------------------------------------------------
// 依赖：lesson04_ekf.hpp（滤波器）+ lesson04_pnp.hpp（观测）
//
// 分层（借鉴 sp_vision_25 的工程分层）：
//   ArmorSolver   ：像素 → 世界系位姿（第 04 章）
//   FilterBank    ：位姿 → 平滑状态（第 06 章）
//   TargetTracker ：一条目标轨迹 = 滤波器 + 状态机 + 预测 + 误差统计
//   TrackerManager：多条轨迹的建立/回收/选优
// ============================================================================

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <optional>
#include <vector>

#include "lesson04_config.hpp"
#include "lesson04_ekf.hpp"
#include "lesson04_pnp.hpp"

namespace training {
namespace l4 {


// ---------------------------------------------------------------- 状态机
// 四态：LOST → DETECTING → TRACKING ⇄ TEMP_LOST → LOST
enum class TrackState { kLost = 0, kDetecting, kTracking, kTempLost };

inline const char* state_name(TrackState s) {
  switch (s) {
    case TrackState::kLost: return "LOST";
    case TrackState::kDetecting: return "DETECT";
    case TrackState::kTracking: return "TRACK";
    case TrackState::kTempLost: return "TEMP_LOST";
  }
  return "?";
}

// ---------------------------------------------------------------- 滤波器组
// 用一个类包住三种可选实现（单模型 / IMM / 环模型），对外接口统一。
// kTrackerMode 决定 internal_ 指向哪一个 ： 上层（状态机、预测、显示）完全不关心用了哪种滤波器。
class FilterBank {
 public:
  void init(const PlateObs& first, int armor_num) {
    mode_ = kTrackerMode;
    armor_num_ = armor_num;
    if (mode_ == 0 || mode_ == 1) {
      plate_.params = (mode_ == 0) ? PlateModelParams{kFallbackAccelNoise, false}  // 纯匀速（回退）
                                   : PlateModelParams{kAccelNoise, true};  // 纯常加速
      plate_.init(first.z);
    } else if (mode_ == 2) {
      imm_.init(first.z, armor_num);
    } else {
      ring_.init(first.z, armor_num);
    }
    init_done_ = true;
  }

  // 角速度平滑：预测专用。慢速 EMA，时间常数 kVyawEmaTau。
  void smooth_yaw_rate(double dt) {
    if (!init_done_ || dt <= 0.0) return;
    const double w = vyaw();
    if (!vyaw_ema_ready_) {
      vyaw_ema_ = w;
      vyaw_ema_ready_ = true;
      return;
    }
    const double alpha = 1.0 - std::exp(-dt / kVyawEmaTau);
    vyaw_ema_ += alpha * (w - vyaw_ema_);
  }
  double vyaw_smooth() const { return vyaw_ema_ready_ ? vyaw_ema_ : vyaw(); }

  void predict(double dt) {
    if (!init_done_) return;
    if (mode_ == 2) {
      imm_.predict(dt);
    } else if (mode_ == 3) {
      ring_.predict(dt, true);
    } else {
      plate_.predict(dt, true);
    }
  }

  // 返回是否有观测被真正吸收
  bool update(const std::vector<PlateObs>& obs) {
    if (!init_done_ || obs.empty()) return false;
    if (mode_ == 2) return imm_.update(obs);
    if (mode_ == 3) {
      bool any = false;
      for (const auto& o : obs) any = ring_.update(o, kGateChi2) || any;
      return any;
    }
    bool any = false;
    for (const auto& o : obs) any = plate_.update(o, kGateChi2) || any;
    return any;
  }

  // 当前"被跟踪那块板"的位置/速度（世界系）
  Eigen::Vector3d plate_position() const {
    if (mode_ == 2) {
      const Eigen::VectorXd x = imm_.combined_x();
      return Eigen::Vector3d(x[PX], x[PY], x[PZ]);
    }
    if (mode_ == 3) return ring_plate_position();
    return plate_.position();
  }
  Eigen::Vector3d plate_velocity() const {
    if (mode_ == 2) {
      const Eigen::VectorXd x = imm_.combined_x();
      return Eigen::Vector3d(x[PVX], x[PVY], x[PVZ]);
    }
    if (mode_ == 3) {
      const Eigen::VectorXd& x = ring_.x();
      return Eigen::Vector3d(x[RVX], x[RVY], x[RVZ]);
    }
    return plate_.velocity();
  }
  double yaw() const {
    if (mode_ == 2) return imm_.combined_x()[PYAW];
    if (mode_ == 3) return ring_.x()[RYAW];
    return plate_.yaw();
  }
  double vyaw() const {
    if (mode_ == 2) return imm_.combined_x()[PVYAW];
    if (mode_ == 3) return ring_.x()[RVYAW];
    return plate_.vyaw();
  }
  double position_sigma() const {
    if (mode_ == 2) {
      const Eigen::MatrixXd P = imm_.combined_P();
      return std::sqrt(std::max(0.0, (P(PX, PX) + P(PY, PY) + P(PZ, PZ)) / 3.0));
    }
    if (mode_ == 3) {
      const Eigen::MatrixXd& P = ring_.P();
      return std::sqrt(std::max(0.0, (P(RX, RX) + P(RY, RY) + P(RZ, RZ)) / 3.0));
    }
    const Eigen::MatrixXd& P = plate_.P();
    return std::sqrt(std::max(0.0, (P(PX, PX) + P(PY, PY) + P(PZ, PZ)) / 3.0));
  }

  // 整车环模型预测：由当前这块板反推车心，推进 dt 后按 90° 间隔给出【4 块板】的位置与朝向。
  // 依据（官方《机器人制作规范手册》S150/S151）：4 块装甲模块方向对应机体 ±X/±Y（误差≤5°），几何中心连线互相垂直 → 同一水平环上 90° 均匀分布。
  // 整车环模型预测：只做一次，给出 4 块板的位置与朝向。
  // 车道心 = 被跟踪板位置 − r·n（r 由实测半径给出，见 kRingRadius），车心按"平动速度"直线/抛物线外推，板再绕车心按 vyaw·dt 转过去。
  // → 被跟踪的那块板退化成 §6.11.1 的环形外推，其余 3 块随之给出，于是"无论 250 ms 后是哪块板朝向相机"，都有对应的预测可用。
  int predict_four_plates(double dt, std::array<Eigen::Vector3d, 4>* out_p,
                          std::array<double, 4>* out_yaw) const {
    if (!init_done_) return 0;
    if (mode_ == 3) {  // 环形状态模型（回退项）：直接用它自带的观测方程
      const Eigen::VectorXd& x = ring_.x();
      Eigen::VectorXd xn = x;
      xn[RX] += x[RVX] * dt;
      xn[RY] += x[RVY] * dt;
      xn[RYAW] += x[RVYAW] * dt;
      const int n3 = (armor_num_ > 0) ? std::min(armor_num_, 4) : 4;
      const double step3 = kTwoPi / static_cast<double>(n3);
      for (int i = 0; i < n3; ++i) {
        (*out_yaw)[static_cast<size_t>(i)] = xn[RYAW] + static_cast<double>(i) * step3;
        (*out_p)[static_cast<size_t>(i)] =
            Eigen::Vector3d(xn[RX] + x[RR] * std::cos((*out_yaw)[static_cast<size_t>(i)]),
                            xn[RY] + x[RR] * std::sin((*out_yaw)[static_cast<size_t>(i)]), xn[RZ]);
      }
      return n3;
    }

    const Eigen::Vector3d p = plate_position();
    const double psi0 = yaw();
    const double w = vyaw_smooth();
    const Eigen::Vector3d n0(std::cos(psi0), std::sin(psi0), 0.0);
    const Eigen::Vector3d t0(-std::sin(psi0), std::cos(psi0), 0.0);  // 切向（焊在车上）

    // 板的速度里含"绕车心转动"那一份，扣掉才是车心平动速度
    Eigen::Vector3d v{0.0, 0.0, 0.0};
    Eigen::Vector3d a{0.0, 0.0, 0.0};
    if (mode_ == 2) {
      const Eigen::VectorXd cx = imm_.combined_x();
      v = Eigen::Vector3d(cx[PVX], cx[PVY], cx[PVZ]);
      a = Eigen::Vector3d(cx[PAX], cx[PAY], cx[PAZ]);
    } else {
      const Eigen::VectorXd& x = plate_.x();
      v = Eigen::Vector3d(x[PVX], x[PVY], x[PVZ]);
      if (plate_.params.hold_accel) a = Eigen::Vector3d(x[PAX], x[PAY], x[PAZ]);
    }
    // 自转越快，板速里"绕车心转动"那一份越大、扣出来的平动量越不可信（换板瞬间速度还要重新对齐），所以按 tang/kCenterVelRefSpeed 衰减平动项。
    const double tang = std::fabs(w) * kRingRadius;
    const double damp = 1.0 / (1.0 + (tang / kCenterVelRefSpeed) * (tang / kCenterVelRefSpeed));
    const Eigen::Vector3d v_center = (v - w * kRingRadius * t0) * damp;
    const Eigen::Vector3d center = p - kRingRadius * n0 + v_center * dt + 0.5 * damp * a * dt * dt;

    const int n = (armor_num_ > 0) ? std::min(armor_num_, 4) : 4;
    const double step = kTwoPi / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
      const double psi = psi0 + w * dt + static_cast<double>(i) * step;
      (*out_yaw)[static_cast<size_t>(i)] = psi;
      (*out_p)[static_cast<size_t>(i)] =
          center + kRingRadius * Eigen::Vector3d(std::cos(psi), std::sin(psi), 0.0);
    }
    return n;
  }

  // 只预测"被跟踪那块板"的位置（单板口径，用于对照与回退；出厂预测走 predict_four_plates）。
  // use_arc=true 且角速度显著时用环形外推（自转靶），否则直线/抛物线外推。
  Eigen::Vector3d predict_plate(double dt, bool use_arc) const {
    const double w = vyaw_smooth();
    if (use_arc && std::fabs(w) > kArcPredictMinVyaw && mode_ != 3) {
      return predict_position_arc(plate_position(), yaw(), w, dt, kRingRadius);
    }
    if (mode_ == 3) {
      const Eigen::VectorXd& x = ring_.x();
      Eigen::VectorXd xn = x;
      xn[RX] += x[RVX] * dt;
      xn[RY] += x[RVY] * dt;
      xn[RYAW] += x[RVYAW] * dt;
      return ring_.model().observe(xn, tracked_plate_index_).head<3>();
    }
    if (mode_ == 2) return predict_position_linear(imm_.combined_x(), dt, true);
    return predict_position_linear(plate_.x(), dt, plate_.params.hold_accel);
  }

  // 换板重置：位置/朝向对齐新观测、速度保留、方差放大
  void reset_position(const PlateObs& o) {
    if (!init_done_) return;
    if (mode_ == 2) {
      imm_.reset_position(o.z, kJumpCovInflate);
    } else if (mode_ == 3) {
      ring_.reset_position(o.z, o.plate_index, kJumpCovInflate);
    } else {
      plate_.reset_position(o.z, kJumpCovInflate);
    }
  }

  // 机动自适应：过程噪声缩放（1=正常）
  void set_q_scale(double s) {
    if (mode_ == 2) {
      imm_.set_q_scale(s);
    } else if (mode_ == 3) {
      ring_.q_scale = s;
    } else {
      plate_.q_scale = s;
    }
  }

  // 硬重置：用一帧观测重建滤波器（滤波器发散时兜底）
  void hard_reset(const PlateObs& o) { init(o, armor_num_); }

  int mode() const { return mode_; }
  bool ready() const { return init_done_; }
  void set_tracked_plate_index(int i) { tracked_plate_index_ = i; }

 private:
  Eigen::Vector3d ring_plate_position() const {
    const Eigen::VectorXd& x = ring_.x();
    const double psi = ring_.model().plate_yaw(x[RYAW], tracked_plate_index_);
    return Eigen::Vector3d(x[RX] + x[RR] * std::cos(psi), x[RY] + x[RR] * std::sin(psi), x[RZ]);
  }

  int mode_ = 2;
  int armor_num_ = 4;
  int tracked_plate_index_ = 0;
  bool init_done_ = false;
  double vyaw_ema_ = 0.0;
  bool vyaw_ema_ready_ = false;
  PlateEkf plate_;
  ImmFilter<PlateEkf> imm_;
  RingEkf ring_;
};

// ---------------------------------------------------------------- 一次预测的记录
// 整车环模型：由"一块可见板 + 环几何"同时给出 **4 块板**在 t+horizon 时刻的预测。
// 250 ms 后哪块板转进视野是不确定的，4 块都预测出来 → 无论看到哪块都有对应预测，误差统计也不再受"比错了板"的干扰。
struct PredictSample {
  double t_target = 0.0;                                  // 预测的目标时刻（秒）
  int n = 0;                                              // 有效预测块数（步兵/英雄 4）
  std::array<Eigen::Vector3d, 4> p{};                     // 各板中心（世界系）
  std::array<double, 4> yaw{};                             // 各板外法向朝向
  int primary = 0;                                        // 主预测 = 最朝向相机的那块（画箭头用）
  // 兼容旧字段：主预测点
  Eigen::Vector3d p_pred = Eigen::Vector3d::Zero();
  double yaw_pred = 0.0;
  // 到某个实测点的最近距离（用于误差统计）
  double min_distance_to(const Eigen::Vector3d& q) const {
    double best = 1e18;
    for (int i = 0; i < n; ++i) best = std::min(best, (p[static_cast<size_t>(i)] - q).norm());
    return (n > 0) ? best : (q - p_pred).norm();
  }
};

// ---------------------------------------------------------------- 单目标轨迹
class TargetTracker {
 public:
  // 用第一帧观测建立轨迹
  void spawn(const PlateObs& obs, int class_id, double now, int armor_num) {
    class_id_ = class_id;
    armor_num_ = armor_num;
    last_yaw_ = obs.z[3];
    bank_.init(obs, armor_num);
    bank_.update({obs});
    state_ = TrackState::kDetecting;
    hits_ = 1;
    misses_ = 0;
    last_update_time_ = now;
    pred_hist_.clear();
    samples_ = 0;
    ok_ = true;
  }

  // 每帧调用：predict → 关联 → update → 状态机
  // returns true 表示这一帧有观测被吸收
  bool step(double now, const std::vector<PlateObs>& candidates, int armor_num) {
    (void)armor_num;   // 环模型会用到；板级模型不需要（保留参数以统一接口）
    const double dt = (last_step_time_ > 0.0) ? std::max(1e-4, now - last_step_time_) : 1.0 / 60.0;
    last_step_time_ = now;
    bank_.predict(dt);

    // ---- 关联：普通匹配 + 换板匹配 ----
    // 关键：同一台车的板每隔 step_yaw = 360°/armor_num 就有一块新的转进视野。
    // 换板瞬间"观测方位"会跳整数个 step_yaw，所以要用"残余方位"判断，而不是硬比原始残差。
    const Eigen::Vector3d p_pred = bank_.plate_position();
    const double sigma = std::max(0.02, bank_.position_sigma());
    const double step_yaw = kTwoPi / static_cast<double>(armor_num_);
    const double pos_gate_jump = 2.0 * kRingRadius + kPlateJumpMargin;   // 换板允许的位移量级

    int best = -1;          // 选中的候选
    bool best_is_jump = false;
    int best_k = 0;         // 换了几块板
    double best_d2 = 1e18;
    double best_rel = 0.0;  // 归一化新息（用于机动检测）
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Eigen::Vector3d dpos = candidates[i].z.head<3>() - p_pred;
      const double d2 = dpos.squaredNorm() / (sigma * sigma + kSigmaPosXy * kSigmaPosXy);
      const double dyaw = wrap_pi(candidates[i].z[3] - bank_.yaw());
      const int k = static_cast<int>(std::lround(dyaw / step_yaw));       // 最近的整块板数
      const double resid_yaw = wrap_pi(dyaw - static_cast<double>(k) * step_yaw);

      const bool normal_ok = (d2 <= kGateChi2) && (std::fabs(resid_yaw) <= kYawGateRad);
      // 换板匹配：必须真的"跳了整数块板"（k≠0），否则只是位置野值，不能当换板处理
      const bool jump_ok = (k != 0) && (misses_ >= kJumpMinMisses) &&
                           (dpos.norm() >= kPlateJumpMinDist) &&
                           (dpos.norm() <= pos_gate_jump) &&
                           (std::fabs(resid_yaw) <= kYawJumpTolRad) &&
                           (std::fabs(static_cast<double>(k)) * step_yaw <= kPi + 1e-6);
      if (!normal_ok && !jump_ok) continue;

      // 普通匹配优先（优先级远高于换板），同类里取马氏距离最小的
      const double score = normal_ok ? d2 : (1e6 + d2);
      if (score < best_d2) {
        best_d2 = score;
        best = static_cast<int>(i);
        best_is_jump = !normal_ok;
        best_k = k;
        best_rel = std::sqrt(d2);
      }
    }

    bool absorbed = false;
    if (best >= 0) {
      PlateObs o = candidates[static_cast<size_t>(best)];
      o.z[3] = unwrap_near(bank_.yaw(), o.z[3]);   // 朝向连续化
      if (best_is_jump) {
        // 换板：把观测点对齐到同一刚体的另一块板上（位置/朝向重置，速度保留）
        plate_index_ += best_k;
        if (armor_num_ > 0) {
          plate_index_ %= armor_num_;
          if (plate_index_ < 0) plate_index_ += armor_num_;
        }
        o.plate_index = plate_index_;
        bank_.reset_position(o);
        ++jumps_;                                   // 统计用（可在状态行里看到）
        // 换板后，之前压入的预测指向的是"旧那块板"，与新板不可比 → 全部作废，否则误差统计与画面上的预测框都会出现假尖峰。
        pred_hist_.clear();
      }
      last_yaw_ = o.z[3];
      bank_.update({o});
      absorbed = true;
      ++hits_;
      misses_ = 0;
      last_update_time_ = now;
      last_obs_ = o;

      // 机动自适应：归一化新息（相对于预测不确定度）持续偏大 → 放大 Q
      const double rel = best_rel;
      maneuver_ema_ = (1.0 - kManeuverEmaAlpha) * maneuver_ema_ + kManeuverEmaAlpha * rel;
      const double scale = std::clamp(maneuver_ema_ / kManeuverThreshold, 1.0, kManeuverQMax);
      bank_.set_q_scale(scale);
    } else {
      ++misses_;
      // 长时间没观测：逐步放大 Q，为"重新捕获时快速跟上"做准备
      if (misses_ > 3) bank_.set_q_scale(std::min(kManeuverQMax, bank_q_scale_hint()));
    }

    // 角速度平滑：每帧都更新一次，掉检时也继续按原速率平滑
    bank_.smooth_yaw_rate(dt);

    // ---- 状态机
    switch (state_) {
      case TrackState::kDetecting:
        if (absorbed) {
          if (hits_ >= kDetectFrames) state_ = TrackState::kTracking;
        } else if (misses_ >= 2) {
          state_ = TrackState::kLost;
        }
        break;
      case TrackState::kTracking:
        if (!absorbed) state_ = TrackState::kTempLost;
        break;
      case TrackState::kTempLost:
        if (absorbed) {
          state_ = TrackState::kTracking;
        } else if (misses_ >= kLostFrames) {
          state_ = TrackState::kLost;
        }
        break;
      case TrackState::kLost:
        break;
    }
    // 长时间无观测 → 直接判定死亡，由 Manager 回收
    if (state_ == TrackState::kLost || (now - last_update_time_) > kMaxTrackAgeSec) {
      if (state_ != TrackState::kLost) state_ = TrackState::kLost;
      if ((now - last_update_time_) > kMaxTrackAgeSec) ok_ = false;
    }
    ++samples_;
    return absorbed;
  }

  // 生成一次"horizon 秒后"的预测：整车环模型一次给出 4 块板
  PredictSample make_prediction(double now, double horizon) const {
    PredictSample s;
    s.t_target = now + horizon;
    s.n = bank_.predict_four_plates(horizon, &s.p, &s.yaw);
    // 主预测：取"最朝向相机"的那块（画箭头与显示用）
    const Eigen::Vector3d cam(0.0, 0.0, bank_.plate_position().z());  // 世界系原点≈相机
    double best = -1e18;
    for (int i = 0; i < s.n; ++i) {
      const Eigen::Vector3d n(std::cos(s.yaw[static_cast<size_t>(i)]),
                              std::sin(s.yaw[static_cast<size_t>(i)]), 0.0);
      const Eigen::Vector3d d = cam - s.p[static_cast<size_t>(i)];
      const double sc = n.dot(d.normalized());
      if (sc > best) { best = sc; s.primary = i; }
    }
    s.p_pred = s.p[static_cast<size_t>(s.primary)];
    s.yaw_pred = s.yaw[static_cast<size_t>(s.primary)];
    return s;
  }
  void push_prediction(const PredictSample& s) {
    pred_hist_.push_back(s);
    while (pred_hist_.size() > static_cast<size_t>(kEvalHistorySize)) pred_hist_.pop_front();
  }
  // 取回"到点"的预测（目标时刻 <= now + tol），并从队列移除
  std::optional<PredictSample> take_due_prediction(double now) {
    // 先丢掉过期的（掉检/丢帧会让队列积压，不清就会一直拿旧预测去比新观测）
    while (!pred_hist_.empty() && pred_hist_.front().t_target < now - kEvalMaxStaleSec)
      pred_hist_.pop_front();
    if (pred_hist_.empty()) return std::nullopt;
    if (pred_hist_.front().t_target <= now + kEvalMatchTolSec) {
      const PredictSample s = pred_hist_.front();
      pred_hist_.pop_front();
      return s;
    }
    return std::nullopt;
  }

  // 距画面中心（用于选板）。由 lesson04_tracking 每帧喂进来。
  void set_center_distance(double d) { center_dist_ = d; }

  // 冷启动/长丢帧时的 Q 提示（与机动自适应共用同一上限）
  static double bank_q_scale_hint() { return kManeuverQMax; }

  int class_id() const { return class_id_; }
  int jumps() const { return jumps_; }
  int plate_index() const { return plate_index_; }
  TrackState state() const { return state_; }
  bool alive() const { return ok_ && state_ != TrackState::kLost; }
  bool tracking() const { return state_ == TrackState::kTracking || state_ == TrackState::kTempLost; }
  double center_distance() const { return center_dist_; }
  double last_update_time() const { return last_update_time_; }
  int misses() const { return misses_; }
  const FilterBank& bank() const { return bank_; }
  FilterBank& bank() { return bank_; }
  int armor_num() const { return armor_num_; }
  const std::optional<PlateObs>& last_obs() const { return last_obs_; }

 private:
  int class_id_ = -1;
  int armor_num_ = 4;
  TrackState state_ = TrackState::kLost;
  int hits_ = 0;
  int misses_ = 0;
  int jumps_ = 0;          // 换板次数（调试/答辩用）
  int plate_index_ = 0;    // 当前跟踪的是这台车的第几块板
  double maneuver_ema_ = 1.0;   // 归一化新息 EMA（机动检测）
  long samples_ = 0;
  bool ok_ = false;
  double last_step_time_ = 0.0;
  double last_update_time_ = 0.0;
  double last_yaw_ = 0.0;
  double center_dist_ = 1e9;
  std::optional<PlateObs> last_obs_;
  FilterBank bank_;
  std::deque<PredictSample> pred_hist_;
};

// ---------------------------------------------------------------- 多目标管理
class TrackerManager {
 public:
  // 主入口：喂本帧所有观测，返回"当前选中的目标"（可能为空）
  TargetTracker* update(double now, std::vector<PlateObs>& observations,
                        const std::vector<double>& center_distances) {
    // ① 每条活着的轨迹都走一遍 predict+关联
    for (auto& tr : tracks_) {
      if (!tr.alive()) continue;
      // 只把"同类编号"的候选交给它（编号是最硬的 ID，讲义 §1.4.5）
      std::vector<PlateObs> mine;
      for (size_t i = 0; i < observations.size(); ++i) {
        if (class_id_of_[i] == tr.class_id()) mine.push_back(observations[i]);
      }
      tr.step(now, mine, tr.armor_num());
    }

    // ② 未被任何轨迹吸收的观测 → 建立新轨迹
    for (size_t i = 0; i < observations.size(); ++i) {
      bool used = false;
      for (auto& tr : tracks_) {
        if (!tr.alive() || tr.class_id() != class_id_of_[i]) continue;
        const Eigen::Vector3d d = observations[i].z.head<3>() - tr.bank().plate_position();
        if (d.norm() < kNewTrackSuppressDist) { used = true; break; }
      }
      if (used) continue;
      if (tracks_.size() >= static_cast<size_t>(kMaxTracks)) {
        // 淘汰最久没更新的
        auto worst = tracks_.begin();
        for (auto it = tracks_.begin(); it != tracks_.end(); ++it)
          if (it->last_update_time() < worst->last_update_time()) worst = it;
        tracks_.erase(worst);
      }
      TargetTracker tr;
      tr.spawn(observations[i], class_id_of_[i], now, armor_num_of_[i]);
      tr.set_center_distance(center_distances.empty() ? 1e9 : center_distances[i]);
      tracks_.push_back(tr);
    }

    // ③ 回收死轨迹
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const TargetTracker& t) { return !t.alive(); }),
                  tracks_.end());

    // ④ 选出当前目标：优先"已在跟踪"，其次离画面中心近
    TargetTracker* best = nullptr;
    double best_score = 1e18;
    for (auto& tr : tracks_) {
      if (!tr.alive()) continue;
      double score = tr.center_distance();
      if (tr.tracking()) score -= 1e6;   // 已锁定的目标优先
      if (score < best_score) {
        best_score = score;
        best = &tr;
      }
    }
    selected_ = best;
    return best;
  }

  // 一帧观测的"编号/装甲板块数"必须先登记，Manager 才知道怎么归类
  void set_observation_meta(const std::vector<int>& class_ids, const std::vector<int>& armor_nums) {
    class_id_of_ = class_ids;
    armor_num_of_ = armor_nums;
  }

  TargetTracker* selected() { return selected_; }
  const TargetTracker* selected() const { return selected_; }
  const std::vector<TargetTracker>& tracks() const { return tracks_; }
  void clear() { tracks_.clear(); selected_ = nullptr; }

 private:
  std::vector<TargetTracker> tracks_;
  std::vector<int> class_id_of_;
  std::vector<int> armor_num_of_;
  TargetTracker* selected_ = nullptr;
};

// ---------------------------------------------------------------- 由类别推装甲板块数
// 4 块：普通步兵/英雄/哨兵；3 块：前哨站；2 块：平衡步兵（大板）
inline int armor_num_from_class(int class_id) {
  if (class_id == 5 || class_id == 12) return 3;                 // BO / RO（前哨站）
  if (class_id == 6 || class_id == 13) return 4;                 // BS / RS（基地，用 4 板模型兜底）
  if (class_id >= 0 && class_id <= 4) return 4;                  // B1..B5
  if (class_id >= 7 && class_id <= 11) return 4;                 // R1..R5
  return 4;
}

}  // namespace l4
// ---------------------------------------------------------------- 跨帧状态
struct TrackingState {
  l4::ArmorSolver solver;              // PnP 求解器（内含上电基准姿态 R_world0）
  bool solver_ready = false;           // 是否已 configure（绘图前防崩）
  l4::TrackerManager manager;          // 多目标轨迹管理器（每条轨迹里含 EKF）

  // 误差统计（"预测 vs 250 ms 后实测"的累计量，讲义 §9.4）
  double pred_err_sum = 0.0;
  double pred_err_sq_sum = 0.0;
  int    pred_err_n = 0;
  double px_err_sum = 0.0;
  double px_err_sq_sum = 0.0;
  int    px_err_n = 0;
  double last_pred_err_mm = -1.0;      // 最近一次 3D 误差（-1 = 还没有，别用 0）
  double last_px_err = -1.0;           // 最近一次像素误差

  // 显示缓冲
  std::array<Armor, 16> last_armors{}; // 最近一帧原始检测（画绿框）
  int last_armor_count = 0;
  std::deque<Eigen::Vector3d> trail;   // 轨迹尾巴（世界系）
  Eigen::Vector3d last_filtered = Eigen::Vector3d::Zero();
  double last_yaw_filtered = 0.0;
  bool has_filtered = false;

  // 打印节流
  double last_print_ms = 0.0;
};
}  // namespace training