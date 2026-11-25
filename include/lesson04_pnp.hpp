// ===== 文件名：include/lesson04_pnp.hpp =====
#pragma once
// ============================================================================
// lesson04_pnp.hpp —— 从 4 个像素角点到"世界系位姿"（第四讲第①②步）
// ----------------------------------------------------------------------------
// 依赖：OpenCV(calib3d/core) + Eigen3 + 仓库的 types.hpp（CameraCalibration/Armor）
// 讲义第 04 章逐行讲解：对象点尺寸、求解器选择、消歧、重投影门控、坐标变换。
//
// 坐标系约定（附录 B）：
//   cam   : x 右 / y 下 / z 前（OpenCV 光学系）
//   gimbal: x 前 / y 左 / z 上
//   world : 上电时刻的 gimbal（IMU 稳定系），x 前 / y 左 / z 上
//   armor : x 右 / y 上 / z 板外法向（朝相机一侧）
// ============================================================================

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lesson04_config.hpp"
#include "lesson04_ekf.hpp"   // kPi / wrap_pi
#include "types.hpp"          // training::Armor / CameraCalibration

namespace training {
namespace l4 {

// ---------------------------------------------------------------- PnP 结果
struct PnpObservation {
  bool ok = false;
  Eigen::Vector3d p_cam = Eigen::Vector3d::Zero();      // 板中心（相机系）
  Eigen::Matrix3d R_cam_armor = Eigen::Matrix3d::Identity();
  Eigen::Vector3d n_cam = Eigen::Vector3d::Zero();      // 板外法向（相机系）
  double reproj_px = 1e9;                               // 平均重投影误差
  double distance = 0.0;                                // 板中心到相机距离
  // 世界系（滤波器的观测量）
  Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
  double yaw_world = 0.0;      // 板外法向的偏航角，连续量
  double yaw_obs = 0.0;        // 本次观测的原始偏航角（-pi..pi）
};

// ---------------------------------------------------------------- 求解器
// 一个对象只管三件事：内参/畸变、相机↔云台外参、上电时刻的姿态基准。
class ArmorSolver {
 public:
  // calibration: 仓库给的相机标定；imu_q: 本帧姿态（用于把结果转到世界系）
  void configure(const CameraCalibration& calibration, const Eigen::Quaterniond& imu_q) {
    camera_matrix_ = calibration.matrix;
    dist_coeffs_ = calibration.distortion;
    if (dist_coeffs_.empty()) dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
    R_cam2gimbal_ = calibration.camera_to_gimbal;
    t_cam2gimbal_ = calibration.camera_to_gimbal_translation;
    R_imu_ = imu_q.normalized().toRotationMatrix();
    if (!world0_set_) {                     // 上电基准：只在第一帧记录一次
      R_world0_ = R_imu_;
      world0_set_ = true;
    }
  }

  // 主函数：解一块装甲板
  PnpObservation solve(const Armor& armor, double last_yaw_world) {
    PnpObservation out;
    const double half_w = (armor.large_armor ? kArmorWidthLarge : kArmorWidthSmall) * 0.5;
    const double half_h = (armor.large_armor ? kArmorHeightLarge : kArmorHeightSmall) * 0.5;
    const std::array<cv::Point3f, 4> obj = object_points(half_w, half_h);

    std::vector<cv::Point2f> img(armor.corners.begin(), armor.corners.end());

    // ① 求两个候选解（平面位姿天生有二义性，见讲义 §4.6）
    std::vector<cv::Mat> rvecs, tvecs;
    try {
      cv::solvePnPGeneric(obj, img, camera_matrix_, dist_coeffs_, rvecs, tvecs, false,
                          cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception&) {
      return out;
    }
    if (rvecs.empty()) return out;

    // ② 消歧：法向必须朝向相机 + 重投影误差最小
    int best = pick_solution(obj, img, rvecs, tvecs);
    if (best < 0) return out;
    cv::Mat rvec = rvecs[best].clone();
    cv::Mat tvec = tvecs[best].clone();

    // ③ 可选精化（LM）。IPPE 已经很准，精化主要救"角点噪声很大"的帧
    if (kEnableRefineLm) {
      try {
        cv::solvePnPRefineLM(obj, img, camera_matrix_, dist_coeffs_, rvec, tvec);
      } catch (const cv::Exception&) {
        // 精化失败就用 IPPE 的解，不算错误
      }
    }

    // ④ 重投影误差（工程上唯一"不需要真值"的质量指标）
    out.reproj_px = reprojection_error(obj, img, rvec, tvec);
    if (!(out.reproj_px <= kReprojGatePx)) return out;   // NaN 也会被这里拦下

    // ⑤ 取出位姿
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) out.R_cam_armor(r, c) = R_cv.at<double>(r, c);
    out.p_cam << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    out.n_cam = out.R_cam_armor.col(2);                  // armor 系 z 轴 = 外法向
    out.distance = out.p_cam.norm();
    if (!(out.distance >= kMinDistanceM && out.distance <= kMaxDistanceM)) return out;

    // ⑥ 相机系 → 云台系 → 世界系
    const Eigen::Vector3d p_gimbal = R_cam2gimbal_ * out.p_cam + t_cam2gimbal_;
    out.p_world = R_world0_.transpose() * (R_imu_ * p_gimbal);
    Eigen::Vector3d n_world = R_world0_.transpose() * (R_imu_ * (R_cam2gimbal_ * out.n_cam));
    // 朝向取"外法向在水平面的方位角"，并把 ±pi 折成连续量（讲义 §4.8）
    out.yaw_obs = std::atan2(n_world.y(), n_world.x());
    out.yaw_world = unwrap_near(last_yaw_world, out.yaw_obs);
    out.ok = true;
    return out;
  }

  // 世界系 → 像素：把估计/预测的板画回图像上（讲义 §8.4）
  // half_w/half_h 是板半宽/半高；yaw 是外法向朝向（世界系）
  std::array<cv::Point2f, 4> world_to_pixel(const Eigen::Vector3d& p_world, double yaw,
                                           double half_w, double half_h) const {
    std::array<cv::Point2f, 4> px{};
    // 装甲板坐标系的三根轴（世界系）
    const Eigen::Vector3d z_axis(std::cos(yaw), std::sin(yaw), 0.0);
    const Eigen::Vector3d y_axis(0.0, 0.0, 1.0);
    const Eigen::Vector3d x_axis = y_axis.cross(z_axis);
    Eigen::Matrix3d R_world_armor;
    R_world_armor.col(0) = x_axis;
    R_world_armor.col(1) = y_axis;
    R_world_armor.col(2) = z_axis;
    // 世界 → 云台 → 相机
    // 世界 → 云台：p_gimbal = R_imu^T R_world0 p_world（由 p_world = R_world0^T R_imu p_gimbal 反解）
    const Eigen::Matrix3d R_gimbal_world = R_imu_.transpose() * R_world0_;
    const Eigen::Matrix3d R_cam_world = R_cam2gimbal_.transpose() * R_gimbal_world;
    const Eigen::Vector3d t_cam_world = -R_cam2gimbal_.transpose() * t_cam2gimbal_;
    const std::array<Eigen::Vector2d, 4> local = {
        Eigen::Vector2d(-half_w, half_h), Eigen::Vector2d(half_w, half_h),
        Eigen::Vector2d(half_w, -half_h), Eigen::Vector2d(-half_w, -half_h)};
    std::vector<cv::Point3f> obj;
    for (const auto& l : local) {
      const Eigen::Vector3d pc = R_cam_world * (p_world + R_world_armor * Eigen::Vector3d(l.x(), l.y(), 0.0)) +
                                 t_cam_world;
      obj.emplace_back(static_cast<float>(pc.x()), static_cast<float>(pc.y()),
                       static_cast<float>(pc.z()));
    }
    std::vector<cv::Point2f> proj;
    const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::projectPoints(obj, rvec, tvec, camera_matrix_, dist_coeffs_, proj);
    for (int i = 0; i < 4; ++i) px[i] = proj[i];
    return px;
  }

  // 板尺寸（画图用）
  static void armor_half_size(bool large, double& half_w, double& half_h) {
    half_w = (large ? kArmorWidthLarge : kArmorWidthSmall) * 0.5;
    half_h = (large ? kArmorHeightLarge : kArmorHeightSmall) * 0.5;
  }

  bool world0_ready() const { return world0_set_; }
  const Eigen::Matrix3d& R_world0() const { return R_world0_; }

 private:
  // 对象点：顺序必须与 Armor::corners 一致（左上、右上、右下、左下）
  // 装甲板坐标系：x 右 / y 上 / z 外法向。注意 y 向上 —— 与 IPPE 文档要求一致。
  static std::array<cv::Point3f, 4> object_points(double half_w, double half_h) {
    return {cv::Point3f(static_cast<float>(-half_w), static_cast<float>(half_h), 0.f),
            cv::Point3f(static_cast<float>(half_w), static_cast<float>(half_h), 0.f),
            cv::Point3f(static_cast<float>(half_w), static_cast<float>(-half_h), 0.f),
            cv::Point3f(static_cast<float>(-half_w), static_cast<float>(-half_h), 0.f)};
  }

  double reprojection_error(const std::array<cv::Point3f, 4>& obj,
                            const std::vector<cv::Point2f>& img, const cv::Mat& rvec,
                            const cv::Mat& tvec) const {
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, camera_matrix_, dist_coeffs_, proj);
    double s = 0.0;
    for (size_t i = 0; i < proj.size(); ++i) s += cv::norm(proj[i] - img[i]);
    return s / static_cast<double>(proj.size());
  }

  // 返回 -1 表示"两个解都不合理"
  int pick_solution(const std::array<cv::Point3f, 4>& obj, const std::vector<cv::Point2f>& img,
                    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs) const {
    if (rvecs.size() == 1) {
      const double rep = reprojection_error(obj, img, rvecs[0], tvecs[0]);
      return (rep <= kReprojGatePx) ? 0 : -1;
    }
    int best = -1;
    double best_score = 1e18;
    for (size_t i = 0; i < rvecs.size(); ++i) {
      cv::Mat R_cv;
      cv::Rodrigues(rvecs[i], R_cv);
      const Eigen::Vector3d n(R_cv.at<double>(0, 2), R_cv.at<double>(1, 2), R_cv.at<double>(2, 2));
      const Eigen::Vector3d t(tvecs[i].at<double>(0), tvecs[i].at<double>(1),
                              tvecs[i].at<double>(2));
      // 法向应指向相机（相机系里相机在原点，所以法向应与 -t 同向）
      const double cosine = n.dot(-t.normalized());
      const double rep = reprojection_error(obj, img, rvecs[i], tvecs[i]);
      // 打分：重投影误差为主，法向朝向为辅（朝向反了直接重罚）
      const double score = rep + ((cosine < 0.0) ? 50.0 : 0.0) - 0.01 * cosine;
      if (score < best_score) {
        best_score = score;
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  Eigen::Matrix3d R_cam2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_cam2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_imu_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_world0_ = Eigen::Matrix3d::Identity();
  bool world0_set_ = false;
};

}  // namespace l4
}  // namespace training
