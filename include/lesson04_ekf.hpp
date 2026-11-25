#pragma once
// ============================================================================
// lesson04_ekf.hpp —— 卡尔曼滤波内核 + 两种运动模型 + IMM
// ----------------------------------------------------------------------------
//
// 坐标系：世界系（IMU 稳定参考系），x 前 / y 左 / z 上，单位 m、rad。
//
// 两种运动模型：
//   PlateEkf（默认，最强）：状态 = 板中心位置/速度/加速度 + 板朝向/角速度，11 维。
//                           位置与朝向是"解耦"的两块，朝向噪声不会污染位置。
//   RingEkf （进阶）      ：状态 = 车体中心/速度/加速度 + 朝向/角速度 + 环半径，12 维。
//                           能表达"板绕车心转"（自转靶、跳板），但目标正对相机时aw 只能通过 r 弱耦合到位置，r 与车心 x 数学上不可分。
// ============================================================================

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "lesson04_config.hpp"
#include "types.hpp"   // 只为复用 training::kPi（避免两处定义 pi 引发命名歧义）

namespace training {
namespace l4 {

// ---------------------------------------------------------------- 小工具

constexpr double kTwoPi = 2.0 * kPi;

// 把角度折到 (-pi, pi]
inline double wrap_pi(double a) {
  a = std::fmod(a + kPi, kTwoPi);
  if (a < 0.0) a += kTwoPi;
  return a - kPi;
}

// 把 b 折算到"离 a 最近"的等价角：返回 a + wrap(b-a)
inline double unwrap_near(double a, double b) { return a + wrap_pi(b - a); }

// ---------------------------------------------------------------- 状态布局
// 板级模型（PlateEkf）11 维
enum PlateIndex {
  PX = 0, PVX = 1, PAX = 2,
  PY = 3, PVY = 4, PAY = 5,
  PZ = 6, PVZ = 7, PAZ = 8,
  PYAW = 9, PVYAW = 10,
  PN = 11
};

// 环模型（RingEkf）12 维
enum RingIndex {
  RX = 0, RVX = 1, RAX = 2,
  RY = 3, RVY = 4, RAY = 5,
  RZ = 6, RVZ = 7, RAZ = 8,
  RYAW = 9, RVYAW = 10,
  RR = 11,
  RN = 12
};

// ---------------------------------------------------------------- 一维平移块
// 每个平移轴都是一个 (p, v, a) 的三阶块。把它的数学写一次，两种模型共用。
struct Axis3 {
  // F 的 3x3 块：p' = p + v·dt + a·dt²/2, v' = v + a·dt, a' = a（CA）或衰减（CV）
  static Eigen::Matrix3d jacobian(double dt, bool use_accel_model) {
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 1) = dt;
    F(0, 2) = 0.5 * dt * dt;
    F(1, 2) = dt;
    F(2, 2) = use_accel_model ? 1.0 : std::exp(-dt / kCvAccelTau);
    return F;
  }
  // Q 的 3x3 块：jerk 白噪声模型（§6.7 的推导结果）
  static Eigen::Matrix3d noise(double dt, double sigma_accel) {
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
    Eigen::Matrix3d Q;
    Q << dt5 / 20.0, dt4 / 8.0, dt3 / 6.0,
         dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0,
         dt3 / 6.0,  dt2 / 2.0, dt;
    return Q * (sigma_accel * sigma_accel);
  }
};

// ---------------------------------------------------------------- EKF 内核
// 只做"给定 F/H/Q/R 的预测与更新"，不含任何业务逻辑。
class EkfCore {
 public:
  void init(int n, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0) {
    n_ = n;
    x_ = x0;
    P_ = P0;
    I_ = Eigen::MatrixXd::Identity(n, n);
    nis_ = 0.0;
    ok_ = false;
  }
  void predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q) {
    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q;
    P_ = 0.5 * (P_ + P_.transpose());
  }
  // 返回 false = 被门限拒绝（NIS 太大 / 数值异常）
  // allow_inflate：门限拒绝时把 R 放大 16 倍再试一次（防"门控饥饿"）
  //   为什么需要它：模型与被跟踪的运动不匹配时（例如用匀速模型跟正弦机动）， 新息会系统性偏大，严格门限会导致"永远不更新 → 自由发散"。
  //   放大 R 等价于"这一帧我只信一点点"，既不会把野值当真，也不会饿死。
  bool update(const Eigen::VectorXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
              double gate_chi2, bool allow_inflate = true) {
    if (try_update(y, H, R, gate_chi2)) return true;
    if (!allow_inflate) return false;
    return try_update(y, H, R * 16.0, gate_chi2 * 4.0);
  }
  bool try_update(const Eigen::VectorXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
                  double gate_chi2) {
    ok_ = false;
    const Eigen::MatrixXd S = H * P_ * H.transpose() + R;
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) return false;
    nis_ = y.transpose() * llt.solve(y);
    if (!(nis_ == nis_)) return false;                       // NaN 检查
    if (gate_chi2 > 0.0 && nis_ > gate_chi2) return false;   // 马氏距离门限
    const Eigen::MatrixXd PHt = P_ * H.transpose();
    const Eigen::MatrixXd K = llt.solve(PHt.transpose()).transpose();
    x_ += K * y;
    const Eigen::MatrixXd IKH = I_ - K * H;
    P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();  // Joseph 形式
    P_ = 0.5 * (P_ + P_.transpose());
    ok_ = true;
    return true;
  }
  double nis() const { return nis_; }
  bool last_update_ok() const { return ok_; }
  Eigen::MatrixXd innovation_cov(const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) const {
    return H * P_ * H.transpose() + R;
  }
  // 高斯对数似然（IMM 打分用）
  static double log_likelihood(const Eigen::VectorXd& y, const Eigen::MatrixXd& S) {
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) return -1e9;
    const double m = static_cast<double>(y.size());
    const double quad = y.transpose() * llt.solve(y);
    double logdet = 0.0;
    const auto L = llt.matrixL();
    for (int i = 0; i < L.rows(); ++i) logdet += 2.0 * std::log(L(i, i));
    return -0.5 * (quad + logdet + m * std::log(kTwoPi));
  }

  const Eigen::VectorXd& x() const { return x_; }
  const Eigen::MatrixXd& P() const { return P_; }
  void set(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) { x_ = x; P_ = P; }
  int n() const { return n_; }

 private:
  int n_ = 0;
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd I_;
  double nis_ = 0.0;
  bool ok_ = false;
};

// ---------------------------------------------------------------- 观测结构
// 一块装甲板的观测：位置 + 朝向（世界系）。plate_index 只有环模型用。
struct PlateObs {
  Eigen::Vector4d z = Eigen::Vector4d::Zero();  // [x, y, z, yaw]
  int plate_index = 0;
};

// 统一观测噪声（两种模型共用）
inline Eigen::Matrix4d plate_measurement_noise() {
  Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
  R(0, 0) = R(1, 1) = kSigmaPosXy * kSigmaPosXy;
  R(2, 2) = kSigmaPosZ * kSigmaPosZ;
  R(3, 3) = kSigmaYaw * kSigmaYaw;
  return R;
}

// ---------------------------------------------------------------- 板级模型（默认）
// 运动模型参数：同一套代码，不同参数 = 不同的"动力学假设"
struct PlateModelParams {
  double sigma_accel = kAccelNoise;   // 加速度噪声 1σ，m/s²
  bool hold_accel = true;             // true=CA（加速度保持） false=CV（加速度衰减）
};

class PlateEkf {
 public:
  static constexpr int N = PN;
  PlateModelParams params{};
  // 机动自适应：过程噪声缩放系数（1 = 正常；>1 = 目标在急变，模型不值得信）
  double q_scale = 1.0;

  void init(const Eigen::Vector4d& z, int /*armor_num*/ = 4) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(N);
    x0[PX] = z[0];
    x0[PY] = z[1];
    x0[PZ] = z[2];
    x0[PYAW] = z[3];
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(N, N);
    P0(PX, PX) = P0(PY, PY) = kSigmaPosXy * kSigmaPosXy;
    P0(PZ, PZ) = kSigmaPosZ * kSigmaPosZ;
    P0(PYAW, PYAW) = kSigmaYaw * kSigmaYaw;
    P0(PVX, PVX) = P0(PVY, PVY) = 2.0 * 2.0;  // 初速度未知 → 大方差
    P0(PVZ, PVZ) = 1.0;
    P0(PAX, PAX) = P0(PAY, PAY) = P0(PAZ, PAZ) = 9.0;
    P0(PVYAW, PVYAW) = 1.0;
    core_.init(N, x0, P0);
  }

  void predict(double dt, bool use_accel_model) {
    const bool use = use_accel_model && params.hold_accel;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(N, N);
    for (int k = 0; k < 3; ++k) F.block<3, 3>(3 * k, 3 * k) = Axis3::jacobian(dt, use);
    F(PYAW, PVYAW) = dt;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);
    const double sa = params.sigma_accel * q_scale;
    for (int k = 0; k < 3; ++k) Q.block<3, 3>(3 * k, 3 * k) = Axis3::noise(dt, sa);
    const double qy = kYawAccelNoise * kYawAccelNoise;
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
    Q(PYAW, PYAW) = qy * dt4 / 4.0;
    Q(PYAW, PVYAW) = qy * dt3 / 2.0;
    Q(PVYAW, PYAW) = qy * dt3 / 2.0;
    Q(PVYAW, PVYAW) = qy * dt2;
    core_.predict(F, Q);
  }

  bool update(const PlateObs& o, double gate_chi2) {
    const Eigen::VectorXd y = residual_vector(o);
    const Eigen::MatrixXd H = jacobian_h(o);
    const Eigen::MatrixXd R = measurement_noise();
    return core_.update(y, H, R, gate_chi2);
  }

  Eigen::Vector4d predict_observation(const PlateObs&) const {
    const Eigen::VectorXd& x = core_.x();
    return Eigen::Vector4d(x[PX], x[PY], x[PZ], x[PYAW]);
  }
  Eigen::MatrixXd jacobian_h(const PlateObs&) const {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, N);
    H(0, PX) = 1.0;
    H(1, PY) = 1.0;
    H(2, PZ) = 1.0;
    H(3, PYAW) = 1.0;
    return H;
  }
  Eigen::MatrixXd measurement_noise() const { return plate_measurement_noise(); }
  Eigen::VectorXd residual_vector(const PlateObs& o) const {
    Eigen::VectorXd y = o.z - predict_observation(o);
    y[3] = wrap_pi(y[3]);
    return y;
  }
  Eigen::MatrixXd residual_cov(const PlateObs& o) const {
    return core_.innovation_cov(jacobian_h(o), measurement_noise());
  }
  // 换板重置：位置与朝向对齐新观测，速度保留，位置/朝向方差放大
  void reset_position(const Eigen::Vector4d& z, double inflate) {
    Eigen::VectorXd x = core_.x();
    Eigen::MatrixXd P = core_.P();
    x[PX] = z[0];
    x[PY] = z[1];
    x[PZ] = z[2];
    x[PYAW] = z[3];
    P(PX, PX) *= inflate;
    P(PY, PY) *= inflate;
    P(PZ, PZ) *= inflate;
    P(PYAW, PYAW) *= inflate;
    core_.set(x, P);
  }

  EkfCore& core() { return core_; }
  const EkfCore& core() const { return core_; }
  const Eigen::VectorXd& x() const { return core_.x(); }
  const Eigen::MatrixXd& P() const { return core_.P(); }
  void set(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) { core_.set(x, P); }

  Eigen::Vector3d position() const {
    const Eigen::VectorXd& xx = core_.x();
    return Eigen::Vector3d(xx[PX], xx[PY], xx[PZ]);
  }
  Eigen::Vector3d velocity() const {
    const Eigen::VectorXd& xx = core_.x();
    return Eigen::Vector3d(xx[PVX], xx[PVY], xx[PVZ]);
  }
  double yaw() const { return core_.x()[PYAW]; }
  double vyaw() const { return core_.x()[PVYAW]; }

 private:
  EkfCore core_;
};

// ---------------------------------------------------------------- 环模型（进阶）
struct RingModel {
  int armor_num = 4;
  double radius = kRingRadius;  // 环半径（默认常数）

  double plate_yaw(double yaw, int i) const {
    return yaw + static_cast<double>(i) * kTwoPi / static_cast<double>(armor_num);
  }
  Eigen::Vector4d observe(const Eigen::VectorXd& x, int i) const {
    const double psi = plate_yaw(x[RYAW], i);
    const double r = x[RR];
    return Eigen::Vector4d(x[RX] + r * std::cos(psi), x[RY] + r * std::sin(psi), x[RZ], psi);
  }
  Eigen::MatrixXd jacobian_h(const Eigen::VectorXd& x, int i) const {
    const double psi = plate_yaw(x[RYAW], i);
    const double r = x[RR];
    const double s = std::sin(psi), c = std::cos(psi);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, RN);
    H(0, RX) = 1.0;
    H(0, RYAW) = -r * s;
    H(0, RR) = c;
    H(1, RY) = 1.0;
    H(1, RYAW) = r * c;
    H(1, RR) = s;
    H(2, RZ) = 1.0;
    H(3, RYAW) = 1.0;
    return H;
  }
  Eigen::VectorXd init_from_armor(const Eigen::Vector4d& z, int i) const {
    const double psi0 = z[3] - static_cast<double>(i) * kTwoPi / static_cast<double>(armor_num);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(RN);
    x[RX] = z[0] - radius * std::cos(psi0);
    x[RY] = z[1] - radius * std::sin(psi0);
    x[RZ] = z[2];
    x[RYAW] = psi0;
    x[RR] = radius;
    return x;
  }
};

class RingEkf {
 public:
  static constexpr int N = RN;
  double q_scale = 1.0;

  void init(const Eigen::Vector4d& z, int armor_num) {
    model_.armor_num = armor_num;
    const Eigen::VectorXd x0 = model_.init_from_armor(z, 0);
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(N, N);
    P0(RX, RX) = P0(RY, RY) = 0.3 * 0.3;
    P0(RZ, RZ) = 0.3 * 0.3;
    P0(RVX, RVX) = P0(RVY, RVY) = 2.0 * 2.0;
    P0(RVZ, RVZ) = 1.0;
    P0(RAX, RAX) = P0(RAY, RAY) = P0(RAZ, RAZ) = 9.0;
    P0(RYAW, RYAW) = 0.5 * 0.5;
    P0(RVYAW, RVYAW) = 1.0;
    P0(RR, RR) = kRadiusPriorSigma * kRadiusPriorSigma;
    core_.init(N, x0, P0);
  }

  void predict(double dt, bool use_accel_model) {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(N, N);
    for (int k = 0; k < 3; ++k) F.block<3, 3>(3 * k, 3 * k) = Axis3::jacobian(dt, use_accel_model);
    F(RYAW, RVYAW) = dt;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);
    const double sa = (use_accel_model ? kAccelNoise : kCvAccelNoise) * q_scale;
    for (int k = 0; k < 3; ++k) Q.block<3, 3>(3 * k, 3 * k) = Axis3::noise(dt, sa);
    const double qy = kYawAccelNoise * kYawAccelNoise;
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
    Q(RYAW, RYAW) = qy * dt4 / 4.0;
    Q(RYAW, RVYAW) = qy * dt3 / 2.0;
    Q(RVYAW, RYAW) = qy * dt3 / 2.0;
    Q(RVYAW, RVYAW) = qy * dt2;
    Q(RR, RR) = kRadiusNoise * kRadiusNoise * dt;
    core_.predict(F, Q);
    clamp_radius();
  }

  bool update(const PlateObs& o, double gate_chi2) {
    const Eigen::VectorXd y = residual_vector(o);
    const Eigen::MatrixXd H = model_.jacobian_h(core_.x(), o.plate_index);
    const Eigen::MatrixXd R = measurement_noise();
    const bool ok = core_.update(y, H, R, gate_chi2);
    clamp_radius();
    return ok;
  }

  Eigen::Vector4d predict_observation(const PlateObs& o) const {
    return model_.observe(core_.x(), o.plate_index);
  }
  Eigen::MatrixXd jacobian_h(const PlateObs& o) const {
    return model_.jacobian_h(core_.x(), o.plate_index);
  }
  Eigen::MatrixXd measurement_noise() const { return plate_measurement_noise(); }
  Eigen::VectorXd residual_vector(const PlateObs& o) const {
    Eigen::VectorXd y = o.z - predict_observation(o);
    y[3] = wrap_pi(y[3]);
    return y;
  }
  Eigen::MatrixXd residual_cov(const PlateObs& o) const {
    return core_.innovation_cov(jacobian_h(o), measurement_noise());
  }
  // 换板重置：以新板的观测重建状态（车心/半径/角速度取当前值的合理延续）
  void reset_position(const Eigen::Vector4d& z, int plate_index, double inflate) {
    Eigen::VectorXd x = model_.init_from_armor(z, plate_index);
    Eigen::MatrixXd P = core_.P();
    P *= inflate;
    core_.set(x, P);
    clamp_radius();
  }

  EkfCore& core() { return core_; }
  const EkfCore& core() const { return core_; }
  const Eigen::VectorXd& x() const { return core_.x(); }
  const Eigen::MatrixXd& P() const { return core_.P(); }
  void set(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) { core_.set(x, P); }
  const RingModel& model() const { return model_; }

 private:
  void clamp_radius() {
    Eigen::VectorXd x = core_.x();
    const double r = std::clamp(x[RR], kRadiusMin, kRadiusMax);
    if (r != x[RR]) {
      x[RR] = r;
      Eigen::MatrixXd P = core_.P();
      P.row(RR) *= 0.5;
      P.col(RR) *= 0.5;
      core_.set(x, P);
    }
  }
  RingModel model_;
  EkfCore core_;
};

// ---------------------------------------------------------------- IMM（CV/CA 双模型）
// 两个模型共享同一状态空间，差别只在：加速度是"衰减"（CV）还是"保持"（CA），
// 以及加速度噪声大小。IMM 用模型概率加权：
//   · 匀速段 → CV 概率高 → 平滑、不抖
//   · 机动段 → CA 概率高 → 跟得上
// 讲义 §10.5 有完整推导与实测量化。
template <class Filter>
class ImmFilter {
 public:
  void init(const Eigen::Vector4d& z, int armor_num) {
    for (int j = 0; j < 2; ++j) {
      ekf_[j].init(z, armor_num);
      ekf_[j].params = params_[j];
    }
    const double stay = 1.0 - 1.0 / (kImmSwitchDwell * 60.0 + 1.0);
    Pi_ << stay, 1.0 - stay, 1.0 - stay, stay;
    mu_ << 0.5, 0.5;
  }

  void predict(double dt) {
    Eigen::MatrixXd cbar(2, 2);
    for (int j = 0; j < 2; ++j) {
      double s = 0.0;
      for (int i = 0; i < 2; ++i) s += Pi_(i, j) * mu_(i);
      cbar(j, 0) = std::max(s, 1e-12);
    }
    Eigen::VectorXd xm[2];
    Eigen::MatrixXd Pm[2];
    for (int j = 0; j < 2; ++j) {
      xm[j] = Eigen::VectorXd::Zero(Filter::N);
      for (int i = 0; i < 2; ++i) xm[j] += (Pi_(i, j) * mu_(i) / cbar(j, 0)) * ekf_[i].x();
      Pm[j] = Eigen::MatrixXd::Zero(Filter::N, Filter::N);
      for (int i = 0; i < 2; ++i) {
        const double w = Pi_(i, j) * mu_(i) / cbar(j, 0);
        const Eigen::VectorXd d = ekf_[i].x() - xm[j];
        Pm[j] += w * (ekf_[i].P() + d * d.transpose());
      }
      ekf_[j].set(xm[j], Pm[j]);
      ekf_[j].predict(dt, j == 1);  // j==1 是 CA 模型
    }
    mu_ = cbar.col(0);
  }

  // 一帧可能有多个观测（多块板）；返回是否有任何一个通过了门限
  bool update(const std::vector<PlateObs>& obs) {
    double lik[2] = {0.0, 0.0};
    bool any = false;
    for (int j = 0; j < 2; ++j) {
      for (const auto& o : obs) {
        const Eigen::VectorXd y = ekf_[j].residual_vector(o);
        const Eigen::MatrixXd S = ekf_[j].residual_cov(o);
        lik[j] += EkfCore::log_likelihood(y, S);
        any = ekf_[j].update(o, kGateChi2) || any;
      }
    }
    const double lmax = std::max(lik[0], lik[1]);
    double w[2] = {mu_(0) * std::exp(lik[0] - lmax), mu_(1) * std::exp(lik[1] - lmax)};
    double s = w[0] + w[1];
    if (!(s > 0.0)) { w[0] = w[1] = 0.5; s = 1.0; }
    mu_(0) = std::max(w[0] / s, kImmProbFloor);
    mu_(1) = std::max(w[1] / s, kImmProbFloor);
    mu_ /= (mu_(0) + mu_(1));
    return any;
  }

  Eigen::VectorXd combined_x() const {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(Filter::N);
    for (int j = 0; j < 2; ++j) x += mu_(j) * ekf_[j].x();
    return x;
  }
  Eigen::MatrixXd combined_P() const {
    const Eigen::VectorXd x = combined_x();
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Filter::N, Filter::N);
    for (int j = 0; j < 2; ++j) {
      const Eigen::VectorXd d = ekf_[j].x() - x;
      P += mu_(j) * (ekf_[j].P() + d * d.transpose());
    }
    return P;
  }
  // 机动自适应：把同一个 Q 缩放系数应用到两个子模型
  void set_q_scale(double s) {
    for (int j = 0; j < 2; ++j) ekf_[j].q_scale = s;
  }
  // 换板重置：两个子模型都对齐到新观测（速度保留）
  void reset_position(const Eigen::Vector4d& z, double inflate) {
    for (int j = 0; j < 2; ++j) ekf_[j].reset_position(z, inflate);
  }
  double prob_ca() const { return mu_(1); }
  const Filter& model(int j) const { return ekf_[j]; }
  PlateModelParams params_[2] = {{kLowDynAccelNoise, false}, {kAccelNoise, true}};  // 低动态/CV、高动态/CA

 private:
  Filter ekf_[2];
  Eigen::Matrix2d Pi_;
  Eigen::Vector2d mu_;
};

// ---------------------------------------------------------------- 预测（含环形修正）
// 直线外推：用于匀速/平移靶
inline Eigen::Vector3d predict_position_linear(const Eigen::VectorXd& x, double dt, bool use_accel) {
  const Eigen::Vector3d p(x[PX], x[PY], x[PZ]);
  const Eigen::Vector3d v(x[PVX], x[PVY], x[PVZ]);
  const Eigen::Vector3d a(x[PAX], x[PAY], x[PAZ]);
  if (use_accel) return p + v * dt + 0.5 * a * dt * dt;
  return p + v * dt;
}

// 环形外推：目标自转时板绕车心画弧（§6.11）。yaw 是板的外法向朝向。
inline Eigen::Vector3d predict_position_arc(const Eigen::Vector3d& p_plate, double yaw, double vyaw,
                                            double dt, double r) {
  const Eigen::Vector3d n(std::cos(yaw), std::sin(yaw), 0.0);
  const Eigen::Vector3d center = p_plate - r * n;
  const double dpsi = vyaw * dt;
  const Eigen::Vector3d rel = p_plate - center;
  const Eigen::Vector3d rotated(std::cos(dpsi) * rel.x() - std::sin(dpsi) * rel.y(),
                                std::sin(dpsi) * rel.x() + std::cos(dpsi) * rel.y(), rel.z());
  return center + rotated;
}

}  // namespace l4
}  // namespace training

