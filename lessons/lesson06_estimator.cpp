#include "rm_aim/estimator.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <cmath>
#include <string>

namespace rmaim {
namespace {
constexpr double kPi = 3.14159265358979323846;

bool solve3(const Eigen::Matrix3d& A, const Eigen::Vector3d& b, Eigen::Vector3d& x) {
  Eigen::Matrix3d Ad = A + Eigen::Matrix3d::Identity() * (1e-9 * (A.trace() + 1.0));
  x = Ad.ldlt().solve(b);
  return x.allFinite();
}
}  // namespace

// ===================== 整车 EKF（半径/高度/角速度都在状态里）=====================
// 状态 x = [cx, cy, cz, vx, vy, vz, yaw, vyaw, r, h]（世界系/云台原点，z 上）
//   (cx,cy,cz) 车体中心；v 中心速度；yaw 车体朝向；vyaw 角速度；r 装甲环半径；h 板相对中心的高度
// 观测 z = [az, el, range]（由单块装甲板的检出给出；range 由框高反推，噪声大但足以让深度可观测）
// 板位置：p_k = c + r·[cos(yaw+kπ/2), sin(yaw+kπ/2), 0] + [0,0,h]
// 关联：把每个检出指派给"预测方位角/俯仰角最近"的那块板（投影式多板关联）。
struct VehicleEkf {
  static constexpr int N = 10;
  using VecN = Eigen::Matrix<double, N, 1>;
  using MatN = Eigen::Matrix<double, N, N>;
  VecN x = VecN::Zero();
  MatN P = MatN::Identity();
  bool inited = false;
  double t = 0.0;

  static constexpr double kArmorPer = 1.5707963267948966;

  Eigen::Vector3d plate_pos(const VecN& s, int k) const {
    const double a = s[6] + k * kArmorPer;
    return Eigen::Vector3d(s[0] + s[8] * std::cos(a), s[1] + s[8] * std::sin(a), s[2] + s[9]);
  }
  Eigen::Vector3d plate_normal(const VecN& s, int k) const {
    const double a = s[6] + k * kArmorPer;
    return Eigen::Vector3d(std::cos(a), std::sin(a), 0.0);
  }
  // 预测观测（相对云台原点）
  Eigen::Vector3d obs_model(const VecN& s, int k) const {
    const Eigen::Vector3d p = plate_pos(s, k);
    const double rng = std::max(0.3, p.norm());
    return Eigen::Vector3d(std::atan2(p.y(), p.x()), std::asin(std::clamp(p.z() / rng, -1.0, 1.0)), rng);
  }

  void predict(double dt) {
    if (!inited || dt <= 0.0) return;
    VecN s = x;
    s[0] += s[3] * dt; s[1] += s[4] * dt; s[2] += s[5] * dt;
    s[6] += s[7] * dt;
    MatN F = MatN::Identity();
    F(0, 3) = F(1, 4) = F(2, 5) = dt;
    F(6, 7) = dt;
    // 过程噪声：中心加速度 2 m/s²、角加速度 8 rad/s²、半径/高度随机游走
    MatN Q = MatN::Zero();
    const double qa = 2.0, qyaw = 8.0;
    for (int i = 0; i < 3; ++i) {
      Q(i, i) = 0.25 * std::pow(dt, 4) * qa * qa;
      Q(i, i + 3) = Q(i + 3, i) = 0.5 * std::pow(dt, 3) * qa * qa;
      Q(i + 3, i + 3) = dt * dt * qa * qa;
    }
    Q(6, 6) = 0.25 * std::pow(dt, 4) * qyaw * qyaw;
    Q(6, 7) = Q(7, 6) = 0.5 * std::pow(dt, 3) * qyaw * qyaw;
    Q(7, 7) = dt * dt * qyaw * qyaw;
    Q(8, 8) = 1e-4 * dt;
    Q(9, 9) = 1e-5 * dt;
    x = s;
    P = F * P * F.transpose() + Q;
    t += dt;
  }

  // 用一次检出更新（k 为指派到的板编号）；返回残差范数（用于门控）
  double update(const Observation& o, int k, double sigma_az, double sigma_el, double sigma_r) {
    const Eigen::Vector3d h = obs_model(x, k);
    Eigen::Vector3d r(o.az - h.x(), o.el - h.y(), o.range - h.z());
    while (r.x() > M_PI) r.x() -= 2 * M_PI;
    while (r.x() < -M_PI) r.x() += 2 * M_PI;
    Eigen::Matrix<double, 3, N> H;
    const double eps = 1e-6;
    for (int j = 0; j < N; ++j) {
      VecN s2 = x;
      s2[j] += eps;
      const Eigen::Vector3d h2 = obs_model(s2, k);
      H(0, j) = (h2.x() - h.x()) / eps;
      H(1, j) = (h2.y() - h.y()) / eps;
      H(2, j) = (h2.z() - h.z()) / eps;
    }
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = sigma_az * sigma_az;
    R(1, 1) = sigma_el * sigma_el;
    R(2, 2) = sigma_r * sigma_r;
    const Eigen::Matrix3d S = H * P * H.transpose() + R;
    const Eigen::Matrix<double, N, 3> K = P * H.transpose() * S.inverse();
    x += K * r;
    P = (MatN::Identity() - K * H) * P;
    P = 0.5 * (P + P.transpose());
    return r.norm();
  }

  void init_from(const Observation& o, double r0, double h0) {
    // 假设该板正对相机：中心 = 板位 + r·(朝相机方向的单位向量)
    const Eigen::Vector3d dir(std::cos(o.el) * std::cos(o.az), std::cos(o.el) * std::sin(o.az),
                              std::sin(o.el));
    const Eigen::Vector3d p = o.range * dir;
    Eigen::Vector3d to_cam = -p;
    if (to_cam.norm() < 1e-6) to_cam = Eigen::Vector3d(0, 1, 0);
    to_cam.normalize();
    const Eigen::Vector3d c = p + r0 * to_cam;
    x.setZero();
    x[0] = c.x(); x[1] = c.y(); x[2] = c.z();
    x[6] = std::atan2(to_cam.y(), to_cam.x());   // 该板的朝向 = 指向相机
    x[7] = 0.0;
    x[8] = r0;
    x[9] = 0.0;
    P.setIdentity();
    P.diagonal() << 0.30, 0.30, 0.15, 1.0, 1.0, 0.3, 0.8, 3.0, 0.05, 0.05;
    inited = true;
    t = o.t;
  }
};

struct Estimator::EkfHolder {
  VehicleEkf ekf;
};

// 用一帧的多板观测驱动整车 EKF：预测 → 投影式关联（每块板算预测 az/el/range，取最近）→ 更新
void Estimator::ekf_step(const std::vector<Observation>& batch) {
  if (!ekf_) ekf_ = std::make_shared<EkfHolder>();
  auto& e = ekf_->ekf;
  const double t0 = batch.front().t;
  if (!e.inited) {
    e.init_from(batch.front(), std::max(0.05, cfg_.ring_radius), 0.0);
    return;
  }
  if (t0 <= e.t) return;
  e.predict(t0 - e.t);
  // 多板关联：每个检出指派到预测最近的一块板，并做卡方门控
  std::vector<bool> plate_used(4, false);
  int matched = 0;
  for (const auto& o : batch) {
    int best_k = -1;
    double best_d = 1e9;
    for (int k = 0; k < 4; ++k) {
      const Eigen::Vector3d h = e.obs_model(e.x, k);
      double daz = o.az - h.x();
      while (daz > M_PI) daz -= 2 * M_PI;
      while (daz < -M_PI) daz += 2 * M_PI;
      const double d = std::fabs(daz) + std::fabs(o.el - h.y());
      if (d < best_d) { best_d = d; best_k = k; }
    }
    if (best_k < 0 || best_d > 0.20) continue;   // ≈11° 门限（初始收敛期需要宽一些）
    if (plate_used[best_k]) continue;
    ++matched;
    plate_used[best_k] = true;
    const double sigma_az = 0.006, sigma_el = 0.006, sigma_r = 0.35;   // 观测噪声
    e.update(o, best_k, sigma_az, sigma_el, sigma_r);
  }
  e.t = t0;
  ekf_matched_ = matched;
  if (matched == 0) {
    ++ekf_lost_;
    if (ekf_lost_ > 25) {          // 连续 25 帧(≈0.5 s)没关联上 → 重新初始化
      e.init_from(batch.front(), std::max(0.05, cfg_.ring_radius), 0.0);
      e.t = t0;
      ekf_lost_ = 0;
    }
  } else {
    ekf_lost_ = 0;
  }
}

void Estimator::reset() {
  for (auto& t : tracks_) t = PlateTrack{};
  obs_.clear();
  hits_ = 0;
  model_name_ = "none";
  sigma_ = 1e3;
  range_ = 0.0;
}

// 用"短窗口快速初始化"环参数：观测刚切到自转靶时，2.5 s 窗口里还混着旧工况的样本，
// 直接用它算中心会给出一个错得离谱的瞄准点（云台于是来回甩）。这里只用最近 0.6 s。
void Estimator::fast_ring_init(double t_now) {
  double lo = 1e9, hi = -1e9, el_sum = 0.0;
  int m = 0;
  for (const auto& o : obs_) {
    if (o.t < t_now - 0.6) continue;
    lo = std::min(lo, o.az);
    hi = std::max(hi, o.az);
    el_sum += o.el;
    ++m;
  }
  if (m < 3 || !(hi > lo)) return;
  center_az_ = 0.5 * (lo + hi);
  ring_half_width_ = std::clamp(0.5 * (hi - lo), 0.008, 0.30);   // 0.46°~17°
  center_el_ = el_sum / static_cast<double>(m);
}

// 预测持续失灵 → 丢掉历史重新捕获（只保留最后一帧）。
// 依据：本仿真里"换靶/换工况"时会出现一段预测点落在车体外的误判期（tup-vision-2023 描述过同一现象），与其让坏历史继续污染拟合，不如立刻重启（成本≈40 ms，收益是云台不再抖 10 s）。
void Estimator::soft_reset(double t_now) {
  if (!obs_.empty()) {
    Observation keep = obs_.back();
    obs_.clear();
    obs_.push_back(keep);
  }
  for (auto& tr : tracks_) tr.active = false;
  batch_.clear();
  pll_inited_ = false;
  pll_lock_count_ = 0;
  el_cross_ok_ = false;
  el_cross_ema_ = 0.0;
  pair_ok_ = false;
  pair_t_ = -1e9;
  u2_hist_.clear();
  cen_inited_ = false;
  rotation_mode_ = false;
  ring_latched_ = false;
  ring_hw_max_ = 0.0;
  ring_hw_t_ = -1e9;
  ring_enter_t_ = -1e9;
  ring_last_evid_ = -1e9;
  acquire_until_ = t_now + 0.40;
  resid_ema_ = 0.0;
  resid_bad_since_ = -1e9;
  aim_prev_valid_ = false;
  smooth_key_t_ = -1e9;
  hits_ = 0;
}

void Estimator::add(const Observation& o) {
  // ---- 预测失灵监控：用"最近两帧外推"跟新观测比，残差持续偏大就判为换靶/工况切换 ----
  if (obs_.size() >= 2) {
    const Observation& b = obs_[obs_.size() - 1];
    const Observation& a = obs_[obs_.size() - 2];
    const double dt = o.t - b.t, dtab = b.t - a.t;
    if (dt > 1e-3 && dt < 0.035 && dtab > 1e-3 && dtab < 0.06) {   // 观测有>35 ms 空档就别赖模型
      double dv = b.az - a.az;
      const double PI = 3.14159265358979323846, TAU = 2 * PI;
      while (dv > PI) dv -= TAU;
      while (dv < -PI) dv += TAU;
      const double v = dv / dtab;
      double pred = b.az + v * dt;
      double r = o.az - pred;
      while (r > PI) r -= TAU;
      while (r < -PI) r += TAU;
      r = std::fabs(r);
      // 快速起跳、慢速回落：真正的换靶会持续几十毫秒的大残差，换板只是一两帧尖峰
      resid_ema_ = (r > resid_ema_) ? (0.7 * resid_ema_ + 0.3 * r) : (0.97 * resid_ema_ + 0.03 * r);
      // 环工况里"换板"本身就会产生大残差，所以阈值放宽（否则会误判成换靶 → 反复重置 → 云台抖）；
      // 平移靶的半遮挡也会造成大残差 → 4° 阈值 + 0.26 s 持续才判失灵（实测 3°/0.22 s 会误重置）
      const double thr = rotation_mode_ ? 0.14 : 0.070;   // ≈8° / 4°
      if (resid_ema_ > thr) {
        if (resid_bad_since_ < 0.0) resid_bad_since_ = o.t;
        else if (o.t - resid_bad_since_ > (rotation_mode_ ? 0.35 : 0.26)) {
          soft_reset(o.t); obs_.push_back(o); ++hits_; return;
        }
      } else {
        resid_bad_since_ = -1e9;
      }
    }
  }
  obs_.push_back(o);
  if (obs_.size() > 500) obs_.pop_front();
  ++hits_;
}

void Estimator::add_batch(const std::vector<Observation>& batch) {
  if (batch.empty()) return;
  const double t = batch.front().t;
  if (std::fabs(t - last_batch_t_) < 1e-9) return;   // 同一帧，跳过（控制循环会重复喂）
  last_batch_t_ = t;
  batch_ = batch;
  update_tracks(batch, t);
  // 记录 u²（u = 本帧最大方位偏移 / hw）—— 它是"环相位"的可观测量，且与"哪块板"无关
  if (center_az_ != 0.0 && ring_half_width_ > 1e-4) {
    double dmax = 0.0;
    for (const auto& o : batch) {
      double d = o.az - center_az_;
      while (d > 3.14159265358979323846) d -= 2 * 3.14159265358979323846;
      while (d < -3.14159265358979323846) d += 2 * 3.14159265358979323846;
      dmax = std::max(dmax, std::fabs(d));
    }
    const double u = std::clamp(dmax / ring_half_width_, 0.0, 1.2);
    if (u > 0.05) {
      u2_hist_.emplace_back(t, u * u);
      while (u2_hist_.size() > 400) u2_hist_.pop_front();
      while (!u2_hist_.empty() && t - u2_hist_.front().first > 3.5) u2_hist_.pop_front();
    }
  }
  // ---- 两板联合估计环中心/半宽（只在自转工况用；静止/平移不受影响）----
  if (rotation_mode_ && batch.size() >= 2) {
    // 取"正对度"最高的两块（正对的板方位读数最可信）
    int i1 = -1, i2 = -1;
    double w1 = -1, w2 = -1;
    for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
      if (batch[i].range < 0.4) continue;
      if (batch[i].w_px > w1) { w2 = w1; i2 = i1; w1 = batch[i].w_px; i1 = i; }
      else if (batch[i].w_px > w2) { w2 = batch[i].w_px; i2 = i; }
    }
    if (i1 >= 0 && i2 >= 0 && pair_hw_ > 1e-3) {
      const double a1 = batch[i1].az, a2 = batch[i2].az;
      double u = pair_ok_ ? pair_az0_ : center_az_;
      const double target = pair_hw_ * pair_hw_;
      for (int it = 0; it < 3; ++it) {                 // 牛顿迭代解 (a1−u)²+(a2−u)²=hw²
        const double d1 = a1 - u, d2 = a2 - u;
        const double f = d1 * d1 + d2 * d2 - target;
        const double df = -2.0 * (d1 + d2);
        if (std::fabs(df) < 1e-9) break;
        u -= f / df;
        if (std::fabs(f / df) < 1e-6) break;
      }
      const double hw_meas = std::sqrt(0.5 * ((a1 - u) * (a1 - u) + (a2 - u) * (a2 - u)));
      const double resid = std::fabs(((a1 - u) * (a1 - u) + (a2 - u) * (a2 - u)) - target);
      if (std::isfinite(u) && hw_meas > 0.010 && hw_meas < 0.25 && resid < 0.10 * target) {
        if (!pair_ok_) { pair_az0_ = u; pair_hw_ = hw_meas; pair_ok_ = true; }
        else {
          double du = u - pair_az0_;
          while (du > 3.14159265358979323846) du -= 2 * 3.14159265358979323846;
          while (du < -3.14159265358979323846) du += 2 * 3.14159265358979323846;
          pair_az0_ += 0.08 * du;
          pair_hw_ += 0.08 * (hw_meas - pair_hw_);
        }
        pair_t_ = t;
      }
    }
  }

  if (cfg_.ring_strategy == "ekf") ekf_step(batch);
}

// 逐板关联：把每帧的观测按"预测位置最近"分配给已有 track；配不上就新建 track。
// 这一步是"换板不再污染历史"的关键 —— 每块物理装甲板各自有一条连续轨迹。
void Estimator::update_tracks(const std::vector<Observation>& batch, double t) {
  const double kPi = 3.14159265358979323846;
  auto wrap = [](double a) {
    const double PI = 3.14159265358979323846, TAU = 2 * PI;
    while (a > PI) a -= TAU;
    while (a < -PI) a += TAU;
    return a;
  };
  bool used[kMaxTracks] = {false, false, false, false};
  for (const auto& o : batch) {
    int best = -1;
    double best_d = 1e9;
    for (int i = 0; i < kMaxTracks; ++i) {
      if (!tracks_[i].active) continue;
      const double dt = t - tracks_[i].t;
      if (dt > 3.0) continue;
      const double pa = tracks_[i].az + tracks_[i].vaz * dt;
      const double pe = tracks_[i].el + tracks_[i].vel * dt;
      const double d = std::fabs(wrap(o.az - pa)) + std::fabs(o.el - pe) + 0.6 * std::fabs(dt);
      if (d < best_d) { best_d = d; best = i; }
    }
    const double gate = 0.015;   // ≈0.86°：环上两块板在过零附近会靠得很近，门太宽会互相串台
    if (best >= 0 && best_d < gate && !used[best]) {
      auto& tr = tracks_[best];
      const double dt = std::max(1e-4, t - tr.t);
      tr.vaz = 0.6 * tr.vaz + 0.4 * wrap(o.az - tr.az) / dt;
      tr.vel = 0.6 * tr.vel + 0.4 * (o.el - tr.el) / dt;
      tr.az = o.az; tr.el = o.el; tr.t = t; tr.w = o.w_px;
      tr.w_max = std::max(tr.w_max, o.w_px);
      tr.h_t[tr.h_i] = t;                       // 方位历史（最小二乘用）
      tr.h_az[tr.h_i] = o.az;
      tr.h_i = (tr.h_i + 1) % PlateTrack::kHistN;
      if (tr.h_n < PlateTrack::kHistN) ++tr.h_n;
      ++tr.n; tr.misses = 0;
      used[best] = true;
    } else {
      // 新建/复用一条最"旧"的 track
      int slot = -1;
      for (int i = 0; i < kMaxTracks; ++i) {
        if (!tracks_[i].active) { slot = i; break; }
      }
      if (slot < 0) {
        double worst = -1;
        for (int i = 0; i < kMaxTracks; ++i) {
          if (used[i]) continue;
          const double age = t - tracks_[i].t + 0.2 * tracks_[i].misses;
          if (age > worst) { worst = age; slot = i; }
        }
      }
      if (slot >= 0) {
        auto& tr = tracks_[slot];
        tr.active = true;
        tr.az = o.az; tr.el = o.el; tr.t = t; tr.t_first = t; tr.w = o.w_px; tr.w_max = o.w_px;
        tr.vaz = 0; tr.vel = 0; tr.n = 1; tr.misses = 0;
        tr.h_n = 1; tr.h_i = 1; tr.h_t[0] = t; tr.h_az[0] = o.az;
        used[slot] = true;
      }
    }
  }
  for (int i = 0; i < kMaxTracks; ++i) {
    if (tracks_[i].active && !used[i]) {
      ++tracks_[i].misses;
      if (tracks_[i].misses > 200) tracks_[i].active = false;   // 长时间没配到就作废
    }
  }
  (void)kPi;
}

// 最小二乘拟合方位斜率：取最近 min(h_n,16) 个样本，要求时间跨度 ≥15 ms。
// 为什么不用 EMA：过零点附近板上角速度最大（hw·ω≈8°/s），EMA 被关键点/框中心抖动污染，外推 0.15 s 就会产生 ~1° 误差，正好等于板半宽 → 一半的弹打空
bool Estimator::fit_track_line(const PlateTrack& tr, double& a, double& b) {
  const int N = std::min(tr.h_n, 16);
  if (N < 4) return false;
  const int HN = PlateTrack::kHistN;
  double t_new = -1e18, t_old = 1e18;
  for (int k = 0; k < N; ++k) {
    const int idx = ((tr.h_i - 1 - k) % HN + HN) % HN;
    const double tk = tr.h_t[idx];
    if (!std::isfinite(tk) || tk <= 0.0) continue;
    t_new = std::max(t_new, tk);
    t_old = std::min(t_old, tk);
  }
  if (!(t_new > t_old) || t_new - t_old < 0.015) return false;
  double st = 0, sz = 0, stt = 0, stz = 0, n = 0;
  for (int k = 0; k < N; ++k) {
    const int idx = ((tr.h_i - 1 - k) % HN + HN) % HN;
    const double tk = tr.h_t[idx];
    if (!std::isfinite(tk) || tk <= 0.0) continue;
    const double dt = tk - t_new;
    st += dt; sz += tr.h_az[idx]; stt += dt * dt; stz += tr.h_az[idx] * dt; n += 1;
  }
  if (n < 4) return false;
  const double den = n * stt - st * st;
  if (std::fabs(den) < 1e-12) return false;
  b = (n * stz - st * sz) / den;
  a = (sz - b * st) / n - b * t_new;
  return std::isfinite(a) && std::isfinite(b);
}

void Estimator::viz_faces(double face[4], double plate_deg[4], double& phase) const {
  // 可视化用：把"观测到的方位偏移"反解成机械角 θ（θ=180° 表示该板正对相机）。
  // 关键：必须用几何半宽hw_geo = atan(R/D)，而不是观测半宽 hw_obs ——
  const double kR2D = 180.0 / 3.14159265358979323846;
  const double R = std::max(0.05, cfg_.ring_radius);
  // 非自转工况（静止/平移）也用同一套几何：以"最近一次观测"为目标中心参考
  const double D = std::max(0.5, rotation_mode_ ? ring_range_med_
                                               : (obs_.empty() ? 3.0 : obs_.back().range));
  const double ref_az = rotation_mode_ ? center_az_ : (obs_.empty() ? 0.0 : obs_.back().az);
  const double hw_geo = std::atan2(R, D);
  double th_best = M_PI, best_face = -1.0;
  int k = 0;
  for (int i = 0; i < kMaxTracks && k < 4; ++i) {
    const auto& tr = tracks_[i];
    if (!tr.active) continue;
    const double f = (tr.w_max > 1e-6) ? std::clamp(tr.w / tr.w_max, 0.0, 1.0) : 0.0;
    face[k] = f;
    double d = tr.az - ref_az;
    while (d > M_PI) d -= 2 * M_PI;
    while (d < -M_PI) d += 2 * M_PI;
    plate_deg[k] = d * kR2D;
    if (f > best_face) {
      best_face = f;
      // δ ≈ hw_geo·sin θ  ⇒  θ = π − asin(δ/hw_geo)（近侧分支，正对时 δ=0 → θ=π）
      th_best = M_PI - std::asin(std::clamp(d / hw_geo, -1.0, 1.0));
    }
    ++k;
  }
  for (; k < 4; ++k) { face[k] = 0.0; plate_deg[k] = 0.0; }
  if (!rotation_mode_) th_best = M_PI;    // 静止/平移：正对相机那块板按 θ=180° 画
  phase = th_best * kR2D;      // 机械角（度）
}

void Estimator::track_stats(int& n_active, double& longest_span, int& longest_n) const {
  n_active = 0; longest_span = 0; longest_n = 0;
  for (int i = 0; i < kMaxTracks; ++i) {
    if (!tracks_[i].active) continue;
    ++n_active;
    const double span = tracks_[i].t - tracks_[i].t_first;
    if (span > longest_span) { longest_span = span; longest_n = tracks_[i].n; }
  }
}

void Estimator::prune(double now) {
  while (!obs_.empty() && now - obs_.front().t > 3.0) obs_.pop_front();
}

// ---------------- CV：u,v 随时间线性 ----------------
Estimator::Fit Estimator::fit_cv(double t_now) const {
  Fit f;
  f.name = "cv";
  f.t_ref = t_now;
  if (obs_.size() < 4) return f;
  const double t_lo = t_now - 0.35;
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d bu = Eigen::Vector3d::Zero(), bv = bu, br = bu;
  int n = 0;
  for (const auto& o : obs_) {
    if (o.t < t_lo) continue;
    const double dt = o.t - t_now;
    const Eigen::Vector3d row(1.0, dt, 0.0);
    A += row * row.transpose();
    bu += row * o.az;
    bv += row * o.el;
    br += row * o.range;
    ++n;
  }
  if (n < 4) return f;
  Eigen::Vector3d cu, cv, cr;
  if (!solve3(A, bu, cu) || !solve3(A, bv, cv) || !solve3(A, br, cr)) return f;
  f.cu = cu;
  f.cv = cv;
  f.r0 = cr[0];
  f.omega = 0.0;
  // 留出：用 0.12 s 之前的样本做验证
  double err = 0.0;
  int cnt = 0;
  for (const auto& o : obs_) {
    if (o.t > t_now - 0.12 || o.t < t_lo) continue;
    const double dt = o.t - t_now;
    const double du = (cu[0] + cu[1] * dt) - o.az;
    const double dv = (cv[0] + cv[1] * dt) - o.el;
    err += du * du + dv * dv;
    ++cnt;
  }
  f.holdout = cnt ? std::sqrt(err / cnt) : 1e9;   // 弧度
  f.ok = cnt >= 2;
  return f;
}

// ---------------- HARM：u,v = a + b cos(ωt) + c sin(ωt) ----------------
Estimator::Fit Estimator::fit_harm(double t_now) const {
  Fit f;
  f.name = "harm";
  f.t_ref = t_now;
  if (obs_.size() < 28) return f;
  const double t_lo = t_now - 2.2;
  const double t_split = t_now - 0.12;
  std::vector<double> ft, ht;
  std::vector<double> fu, fv, fr, hu, hv;
  for (const auto& o : obs_) {
    if (o.t < t_lo) continue;
    const double dt = o.t - t_now;
    if (o.t <= t_split) { ft.push_back(dt); fu.push_back(o.az); fv.push_back(o.el); fr.push_back(o.range); }
    else { ht.push_back(dt); hu.push_back(o.az); hv.push_back(o.el); }
  }
  if (ft.size() < 24 || ht.size() < 3) return f;
  if (-ft.front() < 1.0) return f;   // 至少要覆盖 1 s，否则周期不可辨识

  double best_sse = 1e18, best_w = 0.0;
  Eigen::Vector3d best_u = Eigen::Vector3d::Zero(), best_v = best_u;
  for (double w = 0.2; w <= 9.6; w += 0.04) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d bu = Eigen::Vector3d::Zero(), bv = bu;
    for (size_t i = 0; i < ft.size(); ++i) {
      const Eigen::Vector3d row(1.0, std::cos(w * ft[i]), std::sin(w * ft[i]));
      A += row * row.transpose();
      bu += row * fu[i];
      bv += row * fv[i];
    }
    Eigen::Vector3d cu, cv;
    if (!solve3(A, bu, cu) || !solve3(A, bv, cv)) continue;
    double sse = 0.0;
    for (size_t i = 0; i < ft.size(); ++i) {
      const double c = std::cos(w * ft[i]), sn = std::sin(w * ft[i]);
      const double du = cu[0] + cu[1] * c + cu[2] * sn - fu[i];
      const double dv = cv[0] + cv[1] * c + cv[2] * sn - fv[i];
      sse += du * du + dv * dv;
    }
    if (sse < best_sse) { best_sse = sse; best_w = w; best_u = cu; best_v = cv; }
  }
  if (best_w <= 0.0) return f;

  // 显著性：振幅必须明显大于残差，否则是"把噪声拟合成正弦"
  const double amp = std::max(std::hypot(best_u[1], best_u[2]), std::hypot(best_v[1], best_v[2]));
  const double noise = std::sqrt(best_sse / std::max(1.0, 2.0 * ft.size() - 6.0));
  if (amp < 2.5 * noise || amp < 1.2e-4) return f;   // 幅度不显著（<0.007°）或太小的"周期"拒绝

  double rsum = 0;
  for (double r : fr) rsum += r;
  f.r0 = rsum / static_cast<double>(fr.size());
  f.omega = best_w;
  f.cu = best_u;
  f.cv = best_v;

  double err = 0.0;
  for (size_t i = 0; i < ht.size(); ++i) {
    const double c = std::cos(best_w * ht[i]), sn = std::sin(best_w * ht[i]);
    const double du = (best_u[0] + best_u[1] * c + best_u[2] * sn) - hu[i];
    const double dv = (best_v[0] + best_v[1] * c + best_v[2] * sn) - hv[i];
    err += du * du + dv * dv;
  }
  f.holdout = std::sqrt(err / static_cast<double>(ht.size()));
  f.ok = true;
  return f;
}

// 俯仰摆动的预测（池化谐波拟合的结果）
double Estimator::el_predict(double t_hit, double t_now) const {
  if (!el_fit_ok_ || el_fit_rms_ > 0.008) return center_el_;   // 拟合不可信就退回均值
  const double w = std::max(0.2, cfg_.ring_omega);
  const double dt = t_hit - t_now;
  return el_fit_[0] + el_fit_[1] * std::cos(w * dt) + el_fit_[2] * std::sin(w * dt);
}

Eigen::Vector2d Estimator::eval(const Fit& f, double t) const {
  const double dt = t - f.t_ref;
  if (f.omega > 0.0) {
    const double c = std::cos(f.omega * dt), s = std::sin(f.omega * dt);
    return Eigen::Vector2d(f.cu[0] + f.cu[1] * c + f.cu[2] * s,
                           f.cv[0] + f.cv[1] * c + f.cv[2] * s);
  }
  return Eigen::Vector2d(f.cu[0] + f.cu[1] * dt, f.cv[0] + f.cv[1] * dt);
}

Prediction Estimator::predict(double t_hit) {
  Prediction out;
  if (obs_.size() < 4) return out;
  const double kPi = 3.14159265358979323846;
  const double t_now = obs_.back().t;
  auto wrap_pi = [](double a) {
    const double PI = 3.14159265358979323846, TAU = 2 * PI;
    while (a > PI) a -= TAU;
    while (a < -PI) a += TAU;
    return a;
  };

  // ============ 1) 窗口统计：环中心 / 摆动半宽 / 自转工况判据 ============
  {
    // 环中心 = 2.5 s 窗口内观测方位角的中点；半宽 = 半径（与旋转相位无关）
    double lo = 1e9, hi = -1e9, el_sum = 0.0;
    int m = 0;
    for (const auto& o : obs_) {
      if (o.t < t_now - 2.5) continue;
      lo = std::min(lo, o.az);
      hi = std::max(hi, o.az);
      el_sum += o.el;
      ++m;
    }
    if (m >= 10 && hi > lo) {
      const double c_new = 0.5 * (lo + hi);
      const double h_new = std::clamp(0.5 * (hi - lo), 0.008, 0.30);
      if (rotation_mode_ && t_now > acquire_until_ && ring_half_width_ > 1e-4) {
        // 稳态：瞄准点在环工况下本质是常量 → 强平滑，避免跟着噪声/离群帧跳（这就是云台抖的主因）
        double dc = c_new - center_az_;
        const double PI = 3.14159265358979323846, TAU = 2 * PI;
        while (dc > PI) dc -= TAU;
        while (dc < -PI) dc += TAU;
        center_az_ += 0.06 * dc;
        ring_half_width_ += 0.06 * (h_new - ring_half_width_);
        center_el_ = 0.9 * center_el_ + 0.1 * (el_sum / m);
      } else {
        center_az_ = c_new;
        ring_half_width_ = h_new;
        center_el_ = el_sum / m;
      }
    }
    // 判据：① 方位角抖动（静止靶仅 0.02°） ② 3 s 内方位跨度 < 0.20 rad ③ 穿越中心 ≥3 次
    std::vector<double> azs;
    for (const auto& o : obs_) {
      if (o.t < t_now - 1.3) continue;
      azs.push_back(o.az);
    }
    double az_std = 0.0, az_span = 0.0;
    int sign_changes = 0;
    if (azs.size() >= 12) {
      double mean = 0.0;
      for (double a : azs) mean += a;
      mean /= static_cast<double>(azs.size());
      double var = 0.0;
      for (double a : azs) var += (a - mean) * (a - mean);
      az_std = std::sqrt(var / static_cast<double>(azs.size()));
      int last = 0;
      for (double a : azs) {
        const int sg = (a - center_az_ > 0.0) ? 1 : -1;
        if (last != 0 && sg != last) ++sign_changes;
        last = sg;
      }
      double lo3 = 1e9, hi3 = -1e9;
      for (const auto& o : obs_) {
        if (o.t < t_now - 3.0) continue;
        lo3 = std::min(lo3, o.az);
        hi3 = std::max(hi3, o.az);
      }
      az_span = (hi3 > lo3) ? (hi3 - lo3) : 0.0;
    }
    // 距离中位数：自转靶分支必须用它算重力下坠补偿。
    // （曾经直接用上一次的 range_，环模式刚接管时它是 0 → 下坠补偿失效 → 全部打低）
    {
      std::vector<double> rs;
      for (const auto& o : obs_) {
        if (o.t < t_now - 1.3 || o.range < 0.4) continue;
        rs.push_back(o.range);
      }
      if (!rs.empty()) {
        std::nth_element(rs.begin(), rs.begin() + rs.size() / 2, rs.end());
        ring_range_med_ = rs[rs.size() / 2];
      }
    }
    // ---- 俯仰摆动的池化谐波拟合 ----
    // 实测：自转靶的 obs el 以 2π/ω ≈ 3.1 s 周期摆动 ±1.5°，且所有板共用同一个摆动
    // （周期就是整车角速度周期）→ 因此可以把窗口内所有板的 el 混在一起拟合，无需板号关联。
    // el(t) = el0 + a·cos(ω·t) + b·sin(ω·t)
    {
      const double w = std::max(0.2, cfg_.ring_omega);
      double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      double B[3] = {0, 0, 0};
      int n = 0;
      for (const auto& o : obs_) {
        if (o.t < t_now - 2.6) continue;
        const double dt = o.t - t_now;
        const double b[3] = {1.0, std::cos(w * dt), std::sin(w * dt)};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) A[i][j] += b[i] * b[j];
          B[i] += b[i] * o.el;
        }
        ++n;
      }
      if (n >= 20) {
        // 3x3 高斯消元
        double M[3][4];
        for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) M[i][j] = A[i][j]; M[i][3] = B[i]; }
        bool ok = true;
        for (int c = 0; c < 3 && ok; ++c) {
          int piv = c;
          for (int r = c + 1; r < 3; ++r) if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
          if (std::fabs(M[piv][c]) < 1e-9) { ok = false; break; }
          if (piv != c) for (int k = 0; k < 4; ++k) std::swap(M[c][k], M[piv][k]);
          for (int r = 0; r < 3; ++r) {
            if (r == c) continue;
            const double f = M[r][c] / M[c][c];
            for (int k = c; k < 4; ++k) M[r][k] -= f * M[c][k];
          }
        }
        if (ok) {
          el_fit_[0] = M[0][3] / M[0][0];
          el_fit_[1] = M[1][3] / M[1][1];
          el_fit_[2] = M[2][3] / M[2][2];
          // 残差 → 是否可信
          double sse = 0.0;
          for (const auto& o : obs_) {
            if (o.t < t_now - 2.6) continue;
            const double dt = o.t - t_now;
            const double pred = el_fit_[0] + el_fit_[1] * std::cos(cfg_.ring_omega * dt) +
                                el_fit_[2] * std::sin(cfg_.ring_omega * dt);
            sse += (o.el - pred) * (o.el - pred);
          }
          el_fit_rms_ = std::sqrt(sse / n);
          el_fit_ok_ = true;
        }
      }
    }

    // ---- 观测半宽的"可见性偏差"修正 ---
    // 我们只可能看到"正对的那一半"板，所以观测到的方位跨度天然小于几何半宽
    // hw_geo = atan(R/D)。实测 hw_obs≈3.0~3.7° vs hw_geo≈4.5°（偏小 ~25%）。
    // 任何用 hw 做的相位/几何反解都会被这个偏差带偏，所以这里把 hw 夹到 [0.55, 1.0]·hw_geo。
    if (ring_half_width_ > 1e-4 && ring_range_med_ > 0.4) {
      const double hw_geo = std::atan2(std::max(0.05, cfg_.ring_radius), ring_range_med_);
      if (hw_geo > 2e-3) ring_half_width_ = std::clamp(ring_half_width_, 0.55 * hw_geo, 1.0 * hw_geo);
    }

    // ---- 两板联合估计优先：比 min/max 窗口稳，做一致性校验后采信 ----
    if (pair_ok_ && t_now - pair_t_ < 1.0 && ring_half_width_ > 1e-4) {
      const double PI2 = 3.14159265358979323846, TAU2 = 2 * PI2;
      double dc = pair_az0_ - center_az_;
      while (dc > PI2) dc -= TAU2;
      while (dc < -PI2) dc += TAU2;
      if (std::fabs(dc) < 0.15) {                      // 与窗口估计一致才采信
        center_az_ += 0.35 * dc;
        if (pair_hw_ > 0.4 * ring_half_width_ && pair_hw_ < 2.0 * ring_half_width_)
          ring_half_width_ = 0.7 * ring_half_width_ + 0.3 * pair_hw_;
      }
    }

    // ---- 过零处俯仰的回归截距（比窗口均值更准：均值会被 ±1.5° 的斜视伪影摊平）----
    {
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      int n = 0;
      for (const auto& o : obs_) {
        if (o.t < t_now - 2.0) continue;
        const double dx = o.az - center_az_;
        sx += dx; sy += o.el; sxx += dx * dx; sxy += dx * o.el;
        ++n;
      }
      if (n > 30) {
        const double den = n * sxx - sx * sx;
        if (std::fabs(den) > 1e-12) {
          const double k = (n * sxy - sx * sy) / den;
          const double b = (sy - k * sx) / n;          // δaz=0 处的 el —— 即过零时刻的俯仰
          if (std::isfinite(b)) el_at_center_ = b;
        }
      }
    }

    // ---- 斜视伪影补偿：实测"俯仰读数"随方位偏移线性变化（板越斜，检测框心越偏），这是检测几何伪影、不是目标真实运动。用窗口内的线性回归 el ≈ el0 + k·δaz 学出来，再把瞄准点的 el 按它补偿，能把"竖直方向的白白丢失"补回来。
    {
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      int n = 0;
      for (const auto& o : obs_) {
        if (o.t < t_now - 2.0) continue;
        const double dx = o.az - center_az_;
        if (std::fabs(dx) < 1e-6) continue;
        sx += dx; sy += o.el; sxx += dx * dx; sxy += dx * o.el;
        ++n;
      }
      if (n > 30) {
        const double den = n * sxx - sx * sx;
        if (std::fabs(den) > 1e-12) obl_k_ = (n * sxy - sx * sy) / den;
        obl_k_ = std::clamp(obl_k_, -2.0, 2.0);
      }
    }

    // 自转工况判据（实测最稳的一组）：
    //   ① 方位角抖动 > 0.23°（静止靶仅 0.02°）  ② 3 s 内方位跨度 < 0.20 rad（平移靶必 >11°）
    //   ③ 2.5 s 内穿越中心 ≥3 次（平移靶只在行程两端慢下来）
    // 注：单独用"锁相环残差"不行——环模型允许 ±90° 跳板，任何 4 重对称的周期运动都能被"锁上"。
    // 本仿真的自转靶实测：方位几乎不变（−90.0°±0.05°），俯仰以 3.1 s 周期摆动 ±1.5°。
    // 因此判据改用"俯仰抖动"，否则自转工况永远识别不出来（这也是之前 ring 策略一直不生效的根因）。
    double el_mean = 0.0;
    int en = 0;
    for (const auto& o : obs_) {
      if (o.t < t_now - 1.3) continue;
      el_mean += o.el;
      ++en;
    }
    double el_std = 0.0;
    if (en > 8) {
      el_mean /= en;
      double var = 0.0;
      for (const auto& o : obs_) {
        if (o.t < t_now - 1.3) continue;
        var += (o.el - el_mean) * (o.el - el_mean);
      }
      el_std = std::sqrt(var / en);
    }
    const bool detect_now = cfg_.ring_mode_enable &&
                            ((el_std > 0.004 && az_span < 0.25) ||
                             (az_std > 0.004 && az_span < 0.20 && sign_changes >= 3));
    if (detect_now) { ring_hold_until_ = t_now + 0.4; ring_last_evid_ = t_now; }
    const bool was_ring = rotation_mode_;
    if (!was_ring && detect_now) {
      // 进入自转工况：立刻切换、用 0.6 s 短窗口快速初始化环参数、并进入 0.45 s 收敛期
      ring_enter_t_ = t_now;
      acquire_until_ = std::max(acquire_until_, t_now + 0.30);
      fast_ring_init(t_now);
    }
    // 滞后 + 锁存 + 几何合理性校验：
    //   进入 = 有自转证据 且 摆动半宽足够大（几何上像"环"）
    //   保持 = 锁存（否则判据一抖云台就甩）
    //   退出 = ① 出现明确平移证据（3 s 方位跨度 >0.35 rad 且几乎无俯仰抖动）
    //          ② 或摆动半宽塌缩（说明根本不是环，而是被启动瞬态误判进来的——静止靶就是这样）
    // ring_hw_max_ = 摆动半宽的"记忆最大值"（缓慢衰减）：进入初期窗口里可能只看到半个摆动，用记忆最大值避免"刚进去又因为 hw 变小而退出"的抖振。
    if (t_now - ring_hw_t_ > 0.3) {
      ring_hw_t_ = t_now;
      ring_hw_max_ = std::max(ring_half_width_, ring_hw_max_ * 0.94);
    }
    const bool ring_geom_ok = ring_hw_max_ > 0.026;        // >1.5°
    const bool exit_evidence = ((az_span > 0.35) && (el_std < 0.003)) || !ring_geom_ok;
    if (exit_evidence) ring_latched_ = false;
    if (detect_now && ring_geom_ok) ring_latched_ = true;
    rotation_mode_ = cfg_.ring_mode_enable && ring_latched_;
    if (rotation_mode_ && ring_hw_max_ < 0.02) ring_latched_ = false;   // 双保险
    if (rotation_mode_ != was_ring) aim_prev_valid_ = false;   // 切换瞬间不做平滑，避免拖慢收敛
    (void)az_std; (void)az_span; (void)sign_changes;
  }

  // ============ 1.2) 环相位锁相环（始终运行，用于自校验 + 选板预测）============
  {
    const double hw = std::max(1e-4, ring_half_width_);
    const double w = std::max(0.2, cfg_.ring_omega);
    if (!obs_.empty()) {
      const double delta = obs_.back().az - center_az_;
      const double psi_c = std::acos(std::clamp(delta / hw, -1.0, 1.0));
      const double t_o = obs_.back().t;
      if (!pll_inited_) {
        pll_psi_ = psi_c; pll_t_ = t_o; pll_omega_ = w; pll_inited_ = true;
      } else {
        const double psi_pred = pll_psi_ + pll_omega_ * (t_o - pll_t_);
        // 只在"同一块板"的两支（±ψ）与整周（2π）里选最近的分支；不允许 ±90° 换板——否则任何 4 重对称的周期运动都能被"锁上"，相位信息被抹掉。
        double best = psi_pred, best_e = 1e9;
        for (int k = -4; k <= 4; ++k) {
          for (int r = 0; r < 2; ++r) {
            const double cand = (r ? -psi_c : psi_c) + k * 2 * kPi;
            const double e = std::fabs(wrap_pi(cand - psi_pred));
            if (e < best_e) { best_e = e; best = cand; }
          }
        }
        if (best_e < 0.35) {                    // 正常跟踪：小幅修正
          pll_psi_ += 0.5 * wrap_pi(best - psi_pred);
          pll_lock_count_ = std::min(pll_lock_count_ + 1, 200);
        } else {                                 // 换板/丢检后重新开始
          pll_psi_ = psi_c;
          pll_lock_count_ = 0;
        }
        pll_resid_ema_ = 0.9 * pll_resid_ema_ + 0.1 * best_e;
        pll_t_ = t_o;
        pll_omega_ = w;
      }
    }
  }

  // ============ 1.5) 3D 环模型拟合（反小陀螺）============
  bool ring3d_ok = false;
  double r3_cx = 0, r3_cy = 0, r3_cz = 0, r3_R = 0, r3_phi = 0, r3_w = 0;
  if (rotation_mode_) {
    const double t_lo = t_now - 1.8;
    std::vector<Eigen::Vector3d> pts;
    std::vector<double> ts;
    for (const auto& o : obs_) {
      if (o.t < t_lo || o.pos.norm() < 0.3) continue;
      pts.push_back(o.pos);
      ts.push_back(o.t - t_now);
    }
    if (pts.size() >= 20 && -ts.front() > 0.8) {
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const auto& p : pts) mean += p;
      mean /= static_cast<double>(pts.size());
      const double R0 = std::max(0.05, cfg_.ring_radius);
      double cx = mean.x(), cy = mean.y();
      for (int it = 0; it < 6; ++it) {          // 已知半径，只拟合圆心（Gauss-Newton）
        Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        for (const auto& p : pts) {
          const double dx = p.x() - cx, dy = p.y() - cy;
          const double r = std::sqrt(dx * dx + dy * dy) + 1e-9;
          const Eigen::Vector2d J(-dx / r, -dy / r);
          A += J * J.transpose();
          b += J * (r - R0);
        }
        if (std::fabs(A.determinant()) < 1e-12) break;
        const Eigen::Vector2d d = A.ldlt().solve(-b);
        cx += d.x(); cy += d.y();
        if (d.norm() < 1e-5) break;
      }
      std::vector<double> phis(pts.size());
      double prev = 0.0, unwrap = 0.0;
      for (size_t i = 0; i < pts.size(); ++i) {
        const double a = std::atan2(pts[i].y() - cy, pts[i].x() - cx);
        if (i == 0) { prev = a; unwrap = a; }
        else {
          double d = a - prev;
          while (d > kPi) d -= 2 * kPi;
          while (d < -kPi) d += 2 * kPi;
          unwrap += d; prev = a;
        }
        phis[i] = unwrap;
      }
      double st = 0, sp = 0, stt = 0, stp = 0;
      const double n = static_cast<double>(ts.size());
      for (size_t i = 0; i < ts.size(); ++i) { st += ts[i]; sp += phis[i]; stt += ts[i]*ts[i]; stp += ts[i]*phis[i]; }
      const double den = n * stt - st * st;
      if (std::fabs(den) > 1e-9) {
        const double w = (n * stp - st * sp) / den;
        const double phi0 = (sp - w * st) / n;
        double sse = 0.0;
        for (const auto& p : pts) {
          const double r = std::hypot(p.x() - cx, p.y() - cy);
          sse += (r - R0) * (r - R0);
        }
        const double r_rms = std::sqrt(sse / n);
        const double phase_span = std::fabs(phis.back() - phis.front());
        if (std::fabs(w) > 0.3 && std::fabs(w) < 9.5 && r_rms < 0.6 * R0 && phase_span > 0.8) {
          ring3d_ok = true;
          r3_cx = cx; r3_cy = cy; r3_cz = mean.z(); r3_R = R0;
          r3_phi = phi0; r3_w = w;
        }
      }
    }
  }

  // ============ 2) 自转靶的三套瞄准策略 ============
  if (rotation_mode_) {
    // 方位摆动半宽的物理值 = R/d（用距离中位数算）。观测到的方位范围会偏小，
    // 因为被选中的总是"较正对"的那些板（它们的方位本来就更接近中心）。
    const double hw = std::max(1e-4, ring_half_width_);   // 只用观测到的摆动半宽
    const double w = std::max(0.1, cfg_.ring_omega);
    // ---- (a) track3d：3D 环模型，瞄准命中时刻最正对的那块板 ----
    if (cfg_.ring_strategy == "track3d" && ring3d_ok) {
      const double ph = r3_phi + r3_w * (t_hit - t_now);
      double best_az = 0, best_face = -1, best_rng = 0, best_el = 0;
      for (int k = 0; k < 4; ++k) {
        const double a = ph + k * kPi * 0.5;
        const Eigen::Vector3d p(r3_cx + r3_R * std::cos(a), r3_cy + r3_R * std::sin(a), r3_cz);
        const double rng = p.norm();
        if (rng < 0.5) continue;
        const Eigen::Vector3d nrm(std::cos(a), std::sin(a), 0.0);
        const double face = std::fabs(nrm.dot(-p / rng));
        if (face > best_face) {
          best_face = face;
          best_az = std::atan2(p.y(), p.x());
          best_el = std::asin(std::clamp(p.z() / rng, -1.0, 1.0));
          best_rng = rng;
        }
      }
      out.az = best_az;
      out.el = best_el;
      out.range = std::max(0.5, best_rng > 0.5 ? best_rng : ring_range_med_);
      range_ = out.range;
      out.model_sigma = 0.2 / 57.2958;
      out.model_name = "ring3d";
      out.valid = true;
      model_name_ = "ring3d";
      sigma_ = out.model_sigma;
      return out;
    }

    // ---- (b) track：环相位 θ(t)=θ0+ωt 跟踪（换板=θ±90°）----
    if (cfg_.ring_strategy == "track") {
      if (!obs_.empty()) {
        const double a = std::asin(std::clamp((obs_.back().az - center_az_) / hw, -1.0, 1.0));
        if (!ring_inited_ || std::fabs(t_now - ring_t_) > 1.0) {
          ring_theta_ = a; ring_t_ = t_now; ring_inited_ = true;
        } else {
          const double theta_pred = ring_theta_ + w * (t_now - ring_t_);
          double best = theta_pred, best_err = 1e9;
          for (int k = -2; k <= 3; ++k) {
            for (int refl = 0; refl < 2; ++refl) {
              const double base = (refl ? (kPi - a) : a) + k * 2 * kPi;
              for (int mm = 0; mm < 4; ++mm) {
                const double cand = base + mm * kPi * 0.5;
                const double e = std::fabs(wrap_pi(cand - theta_pred));
                if (e < best_err) { best_err = e; best = cand; }
              }
            }
          }
          ring_theta_ = (best_err < 0.7) ? best : theta_pred;
          ring_t_ = t_now;
        }
      }
      if (ring_inited_) {
        const double theta_h = ring_theta_ + w * (t_hit - ring_t_);
        double best_k = 0.0, best_d = 1e9;
        for (int k = 0; k < 4; ++k) {
          const double th = theta_h + k * kPi * 0.5;
          const double dist = std::fabs(std::remainder(th, kPi));
          if (dist < best_d) { best_d = dist; best_k = k * kPi * 0.5; }
        }
        out.az = center_az_ + hw * std::sin(theta_h + best_k);
        out.el = center_el_ + cfg_.ring_el_offset_deg * kPi / 180.0;
        out.range = std::max(0.5, ring_range_med_);
        range_ = out.range;
        out.model_sigma = 0.15 / 57.2958;
        out.model_name = "ringtrk";
        out.valid = true;
        model_name_ = "ringtrk";
        sigma_ = out.model_sigma;
        return out;
      }
    }

    // ---- (b2) timed：用已锁定的环相位，瞄准命中时刻"最正对"的那块板 ----
    if (cfg_.ring_strategy == "timed" && pll_inited_) {
      // --- 命中时刻选板：找 |cos ψ| 最大（最正对）的一块 ---
      const double psi_h = pll_psi_ + pll_omega_ * (t_hit - pll_t_);
      double best_face = -1.0, best_az = center_az_;
      for (int k = 0; k < 4; ++k) {
        const double psi_k = psi_h + k * kPi * 0.5;
        const double face = std::fabs(std::sin(psi_k));      // 见 docs：face=|sinψ|
        if (face > best_face) {
          best_face = face;
          best_az = center_az_ + hw * std::cos(psi_k);       // 该板在命中时刻的方位
        }
      }
      out.az = best_az;
      out.el = center_el_ + cfg_.ring_el_offset_deg * kPi / 180.0;
      out.range = std::max(0.5, ring_range_med_);
      range_ = out.range;
      out.model_sigma = 0.15 / 57.2958;
      out.model_name = "ringpll";
      out.face_on = best_face;          // 交给火控：不够正对就别开火
      out.valid = true;
      model_name_ = "ringpll";
      sigma_ = out.model_sigma;
      return out;
    }

    // ---- (a0) follow：锁定一块板持续跟踪，快转过去时交接给下一块 ----
    // 思路（来自讨论）：不要每帧都重新挑"最宽的板"，而是锁定一块持续追；
    // 等它转得越来越斜（框宽变窄）／快要消失时，交接给刚出现的相邻板，如此轮转。
    if (cfg_.ring_strategy == "follow") {
      // 交接判据：当前跟踪的板"框宽掉到本 track 历史最大值的 70% 以下"= 正在转过去，此时优先交给另一块刚出现的板（这就是"追板→交接"的核心）。
      int best = -1;
      double best_score = -1e9;
      for (int i = 0; i < kMaxTracks; ++i) {
        const auto& tr = tracks_[i];
        if (!tr.active || tr.misses > 4) continue;
        if (tr.t < t_now - 0.25) continue;               // 太旧的不跟
        const double facing = (tr.w_max > 1.0) ? tr.w / tr.w_max : 0.0;   // 1=完全正对
        double score = tr.w * 1.0 + 40.0 * facing - 15.0 * tr.misses +
                       (tr.t > t_now - 0.12 ? 25.0 : 0.0);
        if (facing < 0.70) score -= 30.0;                // 正在转走 → 主动降权，准备交接
        if (score > best_score) { best_score = score; best = i; }
      }
      if (best >= 0) {
        const auto& tr = tracks_[best];
        const double dt = std::clamp(t_hit - tr.t, 0.0, 0.25);
        const double lead_lim = 0.6 * hw;
        const double daz = std::clamp(tr.vaz * dt, -lead_lim, lead_lim);
        const double del = std::clamp(tr.vel * dt, -0.01, 0.01);
        out.az = tr.az + daz;
        out.el = tr.el + del + cfg_.ring_el_offset_deg * kPi / 180.0;
        out.range = std::max(0.5, ring_range_med_);
        range_ = out.range;
        out.model_sigma = 0.2 / 57.2958;
        out.model_name = "ringfol";
        out.face_on = std::clamp((tr.w_max > 1.0 ? tr.w / tr.w_max : 0.0), 0.0, 1.0);
        out.valid = true;
        model_name_ = "ringfol";
        sigma_ = out.model_sigma;
        return out;
      }
    }

    // ---- (a00) phase2：相位二次拟合（无需板号关联）----
    // 关键观察：4 块板在环上相隔 90°，所以"本帧最大方位偏移" u = max|δ|/hw 满足
    //     u² = (1+cos(2ψ))/2 ，其中 ψ(t) = ψ0 + ωt 是环相位。
    // 于是 u²(t) 是一个**频率已知(2ω)**的纯正弦 → 用线性最小二乘就能把 ψ0 拟合出来，
    // 而且完全不需要知道"这一帧看到的是哪块板"（板号只差 90°，而 u 对 90° 是周期的）。
    if (cfg_.ring_strategy == "phase2" && u2_hist_.size() >= 25) {
      const double w = std::max(0.2, cfg_.ring_omega);
      const double t_ref = t_now;
      // 拟合 u² = A + B·cos(2ω·dt) + C·sin(2ω·dt)，dt = t - t_ref
      double M[3][4] = {{0}};
      int n = 0;
      for (const auto& p2 : u2_hist_) {
        const double dt = p2.first - t_ref;
        const double b[3] = {1.0, std::cos(2 * w * dt), std::sin(2 * w * dt)};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) M[i][j] += b[i] * b[j];
          M[i][3] += b[i] * p2.second;
        }
        ++n;
      }
      bool ok = true;
      for (int c = 0; c < 3 && ok; ++c) {
        int piv = c;
        for (int r = c + 1; r < 3; ++r) if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
        if (std::fabs(M[piv][c]) < 1e-12) { ok = false; break; }
        if (piv != c) for (int j = 0; j < 4; ++j) std::swap(M[c][j], M[piv][j]);
        for (int r = 0; r < 3; ++r) {
          if (r == c) continue;
          const double f = M[r][c] / M[c][c];
          for (int j = c; j < 4; ++j) M[r][j] -= f * M[c][j];
        }
      }
      if (ok) {
        const double B = M[1][3] / M[1][1];
        const double C = M[2][3] / M[2][2];
        double rms = 0.0;
        const double twopsi0 = std::atan2(-C, B);     // 2ψ0
        for (const auto& p2 : u2_hist_) {
          const double dt = p2.first - t_ref;
          const double pred = 0.5 + 0.5 * std::cos(twopsi0 + 2 * w * dt);
          rms += (p2.second - pred) * (p2.second - pred);
        }
        ph_rms_ = std::sqrt(rms / std::max(1, n));
        ph_ok_ = ph_rms_ < 0.08 && std::hypot(B, C) > 0.12;   // 振幅够大且残差小
        std::snprintf(ph_diag_, sizeof(ph_diag_), "ph2 amp=%.2f rms=%.3f ok=%d n=%d hw=%.2f°",
                      std::hypot(B, C), ph_rms_, ph_ok_ ? 1 : 0, static_cast<int>(u2_hist_.size()),
                      hw * 57.2958);
        ph_B_ = B; ph_C_ = C;
        if (ph_ok_) {
          const double psi0 = 0.5 * twopsi0;
          const double psi_h = psi0 + w * (t_hit - t_ref);
          // 候选：四块板的方位偏移（相对环中心）
          double best_abs = 1e9, best_d = 0.0;
          for (int k = 0; k < 4; ++k) {
            const double d = hw * std::cos(psi_h + k * kPi * 0.5);
            if (std::fabs(d) < best_abs) { best_abs = std::fabs(d); best_d = d; }
          }
          // 符号分支：与最近一次观测的 δ 同号（连续性）
          double d_obs = 0.0;
          if (!batch_.empty()) {
            double mn = 1e9;
            for (const auto& o : batch_) {
              const double d = o.az - center_az_;
              if (std::fabs(d) < std::fabs(mn)) mn = d;
            }
            d_obs = mn;
          }
          if (d_obs * best_d < 0.0) best_d = -best_d;
          out.az = center_az_ + best_d;
          out.el = center_el_ + obl_k_ * best_d + cfg_.ring_el_offset_deg * kPi / 180.0;
          out.range = std::max(0.5, ring_range_med_);
          range_ = out.range;
          out.model_sigma = 0.2 / 57.2958;
          out.model_name = "ringph2";
          out.valid = true;
          model_name_ = "ringph2";
          sigma_ = out.model_sigma;
          return out;
        }
      }
    }

    // ---- (b000) ekf：整车 EKF + 投影式多板关联 ----
    if (cfg_.ring_strategy == "ekf") {
      // 健康门控：整车 EKF 只有在"本帧至少关联上 2 块板"时才可信；
      // 否则（初始化不足/发散）直接退回 dwell，避免把云台甩飞（实测过）。
      if (ekf_ && ekf_->ekf.inited && ekf_matched_ >= 1) {
        // 外推到命中时刻，选"最正对"的那块板
        Estimator::EkfHolder tmp = *ekf_;
        tmp.ekf.predict(std::max(0.0, t_hit - tmp.ekf.t));
        double best_face = -1.0, best_az = center_az_, best_el = center_el_, best_rng = 0.0;
        for (int k = 0; k < 4; ++k) {
          const Eigen::Vector3d p = tmp.ekf.plate_pos(tmp.ekf.x, k);
          const Eigen::Vector3d n = tmp.ekf.plate_normal(tmp.ekf.x, k);
          const double rng = p.norm();
          if (rng < 0.5) continue;
          const double face = n.dot(-p / rng);          // 正对相机 → +1
          if (face > best_face) {
            best_face = face;
            best_az = std::atan2(p.y(), p.x());
            best_el = std::asin(std::clamp(p.z() / rng, -1.0, 1.0));
            best_rng = rng;
          }
        }
        if (best_rng > 0.5) {
          out.az = best_az;
          out.el = best_el + cfg_.ring_el_offset_deg * kPi / 180.0;
          out.range = best_rng;
          range_ = out.range;
          out.model_sigma = 0.25 / 57.2958;
          out.model_name = "ringekf";
          out.face_on = (best_face + 1.0) * 0.5;
          out.valid = true;
          model_name_ = "ringekf";
          sigma_ = out.model_sigma;
          return out;
        }
      }
    }

    // ---- (b00) grid：整车级联合拟合（相位网格搜索 + 多板指派）----
    // 关键点：不依赖深度。用"环上 4 块板 90° 分布 + 角速度 ω 已知"这一结构，
    // 对相位 φ0 做一维网格搜索，把当前帧所有检出**指派**到 4 个预测位置里最近的那个，
    // 取总残差最小者。这样即使每块板只可见 0.1~0.6 s，也能靠"多板联合"锁住整车相位。
    if (cfg_.ring_strategy == "grid" && !batch_.empty() && hw > 1e-4) {
      const double w = std::max(0.2, cfg_.ring_omega);
      const double t_now3 = batch_.front().t;
      double best_phi = grid_phi_, best_err = 1e18;
      const int kGrid = 60;
      for (int gi = 0; gi < kGrid; ++gi) {
        const double phi = -kPi + 2 * kPi * gi / kGrid;
        double err = 0.0;
        for (const auto& o : batch_) {
          const double d = wrap_pi(o.az - center_az_);
          double best = 1e9;
          for (int k = 0; k < 4; ++k) {
            const double pred = hw * std::cos(phi + k * kPi * 0.5);
            const double e = std::fabs(wrap_pi(d - pred));
            if (e < best) best = e;
          }
          err += best * best;
        }
        if (err < best_err) { best_err = err; best_phi = phi; }
      }
      grid_phi_ = best_phi;
      grid_err_ = std::sqrt(best_err / static_cast<double>(batch_.size()));
      // 只有残差足够小才认为是"环运动"（静止/平移靶不会满足）
      if (grid_err_ < 0.012) {          // ≈0.7°
        // 预测：命中时刻哪块板最正对（该板位于环中心方位附近）
        const double ph = grid_phi_ + w * (t_hit - t_now3);
        double best_face = -1, best_az = center_az_;
        for (int k = 0; k < 4; ++k) {
          const double a = ph + k * kPi * 0.5;
          const double face = std::sin(a);         // 正对相机的板 sinψ≈+1
          if (face > best_face) { best_face = face; best_az = center_az_ + hw * std::cos(a); }
        }
        out.az = best_az;
        out.el = center_el_ + cfg_.ring_el_offset_deg * kPi / 180.0;
        out.range = std::max(0.5, ring_range_med_);
        range_ = out.range;
        out.model_sigma = 0.2 / 57.2958;
        out.model_name = "ringgrid";
        out.face_on = (best_face + 1) * 0.5;
        out.valid = true;
        model_name_ = "ringgrid";
        sigma_ = out.model_sigma;
        return out;
      }
    }

    // ---- (b0) zoh：零阶保持（瞄准"当前看到的位置"，不加提前量）----
    // 实测依据：对"往复摆动"的目标，提前量估计的噪声大于它带来的收益
    // （逐板跟踪+全提前 10.3% vs 零阶保持 21.3%）。
    if (cfg_.ring_strategy == "zoh" && !obs_.empty()) {
      out.az = obs_.back().az;
      out.el = cfg_.use_el_harmonic ? el_predict(t_hit, t_now) : obs_.back().el;
      out.range = std::max(0.5, ring_range_med_);
      range_ = out.range;
      out.model_sigma = 0.3 / 57.2958;
      out.model_name = "ringzoh";
      out.valid = true;
      model_name_ = "ringzoh";
      sigma_ = out.model_sigma;
      return out;
    }

    // ---- (b3) center：多板联合 + 跟踪"最正对的那块板"（离环中心最近的板）----
    // 依据：正对相机的板其方位偏移最小（δ=hw·cos ψ，正对时 ψ≈90°⇒δ≈0），
    //       而且它在环中心位置以 hw·ω 的角速度扫过 ⇒ 必须加提前量。
    // 用「多板联合」：每帧对所有板的 |δ| 排序，取最小的那块（正对板），
    // 再对该板的 δ 做时序拟合得到 δ̇，用 δ(t_hit)=δ+δ̇·Δt 预测。
    if (cfg_.ring_strategy == "center" && !batch_.empty()) {
      double dmin = 1e9, dsum = 0.0;
      int n_valid = 0;
      for (const auto& o : batch_) {
        const double d = wrap_pi(o.az - center_az_);
        dsum += d;
        ++n_valid;
        if (std::fabs(d) < std::fabs(dmin)) dmin = d;
      }
      // 一致性检查：多板时 |δ| 的最大/最小应当满足 u²+v²≈hw²（环上 4 板 90° 分布）
      double dmax = 0.0;
      for (const auto& o : batch_) {
        const double d = wrap_pi(o.az - center_az_);
        if (std::fabs(d) > std::fabs(dmax)) dmax = d;
      }
      const double uv = std::hypot(dmin, dmax);
      const bool uv_ok = (n_valid < 2) || (uv > 0.55 * hw && uv < 1.5 * hw);
      (void)dsum;
      if (uv_ok && std::isfinite(dmin)) {
        // 记录该板的 δ 序列（换板时 δ 会突跳 → 用预测值判断）
        const double dt = (t_now - cen_t_);
        if (cen_inited_ && dt > 1e-4 && dt < 0.5 &&
            std::fabs(wrap_pi(dmin - (cen_d_ + cen_v_ * dt))) < 0.05) {
          const double v_meas = wrap_pi(dmin - cen_d_) / dt;
          cen_v_ = 0.7 * cen_v_ + 0.3 * std::clamp(v_meas, -6.0 * hw, 6.0 * hw);
        } else {
          cen_v_ = 0.0;   // 换板/丢检 → 速度清零重估
        }
        cen_d_ = dmin;
        cen_t_ = t_now;
        cen_inited_ = true;
        const double lead = std::clamp(cen_v_ * (t_hit - t_now), -0.5 * hw, 0.5 * hw);
        out.az = center_az_ + dmin + lead;
        out.el = center_el_ + cfg_.ring_el_offset_deg * kPi / 180.0;
        out.range = std::max(0.5, ring_range_med_);
        range_ = out.range;
        out.model_sigma = 0.15 / 57.2958;
        out.model_name = "ringctr";
        out.valid = true;
        model_name_ = "ringctr";
        sigma_ = out.model_sigma;
        return out;
      }
    }

    // ---- 过零瞬间的俯仰直采（比窗口均值干净：窗口里混着 ±1.5° 的斜视伪影）----
    // 板正在过零（|δaz| 小、且正对）时读数最接近真值；把它做 EMA，作为瞄准用的 el。
    {
      const double ymax = 0.35 * std::max(1e-4, hw);
      for (const auto& tr : tracks_) {
        if (!tr.active || tr.n < 3) continue;
        const double face = (tr.w_max > 1e-6) ? std::clamp(tr.w / tr.w_max, 0.0, 1.0) : 0.0;
        if (face < 0.80) continue;
        if (t_now - tr.t > 0.08) continue;
        if (std::fabs(wrap_pi(tr.az - center_az_)) > ymax) continue;
        if (!el_cross_ok_) { el_cross_ema_ = tr.el; el_cross_ok_ = true; }
        else el_cross_ema_ = 0.85 * el_cross_ema_ + 0.15 * tr.el;
      }
    }

    // ---- (c) dwell（默认）：守方位角驻留点 ----
    out.az = center_az_ + cfg_.ring_dwell_scale * hw + cfg_.ring_az_offset_deg * kPi / 180.0;
    const double el_corr = obl_k_ * (cfg_.ring_dwell_scale * hw);
    // 注：实测"用池化谐波预测俯仰"反而更差（16.2 → 9.0/7.1/3.2），
    // 因为自转工况下观测到的 ±1.5° 俯仰摆动主要是"云台持续运动时的姿态补偿残差"，
    // 不是目标真实运动。这里仍用窗口均值，反而更稳。
    // 注：曾试过用"过零处俯仰回归截距"替代窗口均值，实测反而更差（docs/07 §27），故保留均值
    out.el = (el_cross_ok_ ? el_cross_ema_ : center_el_) + el_corr +
             cfg_.ring_el_offset_deg * kPi / 180.0;
    out.range = std::max(0.5, ring_range_med_);
    range_ = out.range;
    out.model_sigma = 0.2 / 57.2958;
    out.model_name = "ring";

    // ================= ★ 命中时刻对齐门控=================
    // 旧门控比的是"预测点 vs 云台实际角"，而驻留点是常量 → 门控基本恒开 →
    // 以 8.4 发/s 连续开火，只有落进"板正好扫过瞄准点"那 0.41 占空比的弹才命中（P(hit)=15%）。
    // 这里改成直接回答"这一枪会不会打中"：把每条看得见的逐板轨迹外推到子弹到达时刻，看它是否落在瞄准点 ±(板半宽×正对度) 之内。环上过零点附近二阶误差 <0.15°，足够。
    // ---- 可视化：命中时刻"最该打的那块板"（与门控无关，始终计算）----
    {
      const double rng_v = std::max(0.8, ring_range_med_);
      double best = -1.0;
      for (const auto& tr : tracks_) {
        if (!tr.active || tr.n < 2) continue;
        const double face = (tr.w_max > 1e-6) ? std::clamp(tr.w / tr.w_max, 0.0, 1.0) : 0.0;
        if (t_now - tr.t > 0.30) continue;
        double fa = 0.0, fb = 0.0;
        double az_hit = tr.az + tr.vaz * (t_hit - tr.t);
        if (fit_track_line(tr, fa, fb)) az_hit = fa + fb * t_hit;
        if (face > best) {
          best = face;
          out.plate_az_hit = az_hit;
          out.plate_el_hit = tr.el;
          out.plate_face = face;
          out.plate_hit_ok = true;
        }
      }
      (void)rng_v;
    }

    const bool mode_gated = (cfg_.ring_fire_mode == "impact");
    if (mode_gated) {
      const double rng_g = std::max(0.8, ring_range_med_);
      const double half_w = 0.5 * cfg_.plate_width_small / rng_g;   // 板半宽（rad）
      const double min_tol = cfg_.ring_gate_min_tol_deg * kPi / 180.0;
      bool any = false;
      double best_margin = -1e9, best_tol = 0.0, best_face = 0.0, best_err = 1e9;
      for (const auto& tr : tracks_) {
        if (!tr.active || tr.n < 3) continue;
        const double face = (tr.w_max > 1e-6) ? std::clamp(tr.w / tr.w_max, 0.0, 1.0) : 0.0;
        if (face < cfg_.ring_gate_face_min) continue;              // 侧视的板不算
        const double dt = t_hit - tr.t;
        if (dt < -0.06 || dt > 0.35) continue;                     // 观测太旧：不许凭空外推
        if (t_now - tr.t > 0.35) continue;                         // 超过 350 ms 没看见 → 不可信
        // ★ 外推到命中时刻：优先用最小二乘斜率（比 EMA 稳一个量级），并夹到物理合理范围
        double az_hit = tr.az + tr.vaz * dt;
        double fa = 0.0, fb = 0.0;
        if (fit_track_line(tr, fa, fb)) {
          const double vphys = hw * cfg_.ring_omega;   // 过零点附近的角速度（上限量级）
          fb = std::clamp(fb, -1.6 * vphys, 1.6 * vphys);
          az_hit = fa + fb * t_hit;
        }
        const double err = std::fabs(wrap_pi(az_hit - out.az));
        // 容差随正对度收缩：侧视时板在方位上的有效宽度本来就会变窄
        const double tol = std::max(min_tol, half_w * cfg_.ring_gate_tol_scale * std::max(0.45, face));
        const double margin = tol - err;
        if (!any || margin > best_margin) {
          best_margin = margin; best_tol = tol; best_face = face; best_err = err; any = true;
        }
      }
      out.fire_window = any && best_margin > 0.0;
      out.fire_margin = any ? best_margin : -1.0;
      out.fire_tol = best_tol;
      out.fire_face = best_face;
      (void)best_err;
    }

    out.valid = true;
    model_name_ = "ring";
    sigma_ = out.model_sigma;
    return out;
  }

  // ============ 3) 常规工况：CV / HARM ============
  Fit cv = fit_cv(t_now);
  Fit ha = fit_harm(t_now);
  const Fit* best = nullptr;
  if (cv.ok) best = &cv;
  if (cfg_.force_model == "harm") {
    if (ha.ok) best = &ha;
  } else if (cfg_.force_model == "cv") {
    // 只允许线性模型（消融用）
  } else if (ha.ok && ha.omega > 0.55 && ha.omega < 4.2) {
    // 只要拟合出一个"物理上合理的往复周期"（0.55~4.2 rad/s ⇔ 1.5~11 s），就优先用谐波模型：
    // 平移靶的往复、旋转靶上单板的回旋都落在这个区间；静止靶会被 fit_harm 的振幅显著性检验直接拒掉（amp 不足以超过噪声），不会误用。
    best = &ha;
  } else if (ha.ok && (!best || ha.holdout < best->holdout * 0.95)) {
    // 谐波模型只要"不差于"线性模型就优先选它：防"把噪声当周期"的任务
    // 由 fit_harm() 里的振幅显著性检验承担（amp > 4×residual）。
    best = &ha;
  }
  std::snprintf(diag_, sizeof(diag_), "cv=%s(%.3f°) harm=%s(%.3f°,w=%.2f) amp=%.1fpx noise=%.1fpx n=%zu",
                cv.ok ? "ok" : "no", cv.ok ? cv.holdout * 57.2958 : -1.0,
                ha.ok ? "ok" : "no", ha.ok ? ha.holdout * 57.2958 : -1.0, ha.omega,
                dbg_amp_, dbg_noise_, obs_.size());
  if (!best) return out;
  model_name_ = best->name;
  sigma_ = best->holdout;
  const Eigen::Vector2d ae = eval(*best, t_hit);
  out.az = ae.x();
  out.el = ae.y();
  out.range = std::max(0.3, best->r0);
  range_ = out.range;
  out.model_sigma = best->holdout;
  out.model_name = best->name;
  out.valid = ae.allFinite();
  return out;
}

bool ballistic_solution(double d, double h, double v0, double g, double& theta, double& t_f) {
  d = std::max(0.05, d);
  const double a = g * d * d / (2.0 * v0 * v0);
  const double disc = d * d - 4.0 * a * (a + h);
  if (disc <= 0.0) return false;
  const double tan_theta = (d - std::sqrt(disc)) / (2.0 * a);
  theta = std::atan(tan_theta);
  const double c = std::cos(theta);
  if (std::fabs(c) < 1e-6) return false;
  t_f = d / (v0 * c);
  return true;
}

}  // namespace rmaim
