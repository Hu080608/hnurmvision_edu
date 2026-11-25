// ============================================================================
// 第四讲：位姿解算 + EKF 跟踪 + 250 ms 预测
// ----------------------------------------------------------------------------
// 本文件把讲义第 04~09 章接到仓库骨架上，只做三件事：
//   1) lesson04_tracking()：每帧把 Armor 观测变成 Target（PnP → 坐标变换 → EKF → 预测）
//   2) lesson04_display() ：把「检测 / 滤波 / 250 ms 预测」画到图像上
//   3) 打印统计行         ：录屏与答辩时要的数据
// ============================================================================

// 第四讲：位姿、坐标与基础跟踪
//
// 输入格式：
//   armors     : const std::vector<Armor>&，第三讲输出的本帧装甲板观测。
//   timestamp  : TimePoint，本帧图像时刻；只能与其他 TimePoint 求时间差。
//   imu_q      : const Eigen::Quaterniond&，图像时刻的归一化姿态四元数。
//   calibration: const CameraCalibration&：
//                matrix=3x3 CV_64F 相机内参；distortion=1x5 CV_64F 畸变；
//                camera_to_gimbal=相机到云台的 3x3 旋转矩阵；
//                camera_to_gimbal_translation=相机到云台平移，单位 m。
//   state      : TrackingState&，仅本讲使用的跨帧状态；函数需要原地更新。
// 输出格式：
//   std::optional<Target>。有目标时：
//   - position : IMU 稳定参考系中、相对云台原点的目标向量，单位 m；
//   - velocity : 同一参考坐标系中的目标速度，单位 m/s。
//   本课程统一使用 x 向前、y 向左、z 向上的右手坐标系。
//   无有效观测、PnP 失败或跟踪未建立时返回 std::nullopt。

#include "lessons.hpp"

#include <opencv2/freetype.hpp>   // 中文文字（cv::putText 的内置字体只有 ASCII）
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "tracking_state.hpp"

namespace training {
namespace {

using namespace training::l4;

// steady_clock 的时间点 → 秒（只用来给状态机一条单调递增的 double 时间轴）
double to_seconds(const TimePoint& t) {
  return std::chrono::duration<double>(t.time_since_epoch()).count();
}

cv::Scalar color_for_state(TrackState s) {
  switch (s) {
    case TrackState::kTracking: return cv::Scalar(0, 255, 0);
    case TrackState::kTempLost: return cv::Scalar(0, 200, 255);
    case TrackState::kDetecting: return cv::Scalar(255, 200, 0);
    case TrackState::kLost: return cv::Scalar(0, 0, 255);
  }
  return cv::Scalar(255, 255, 255);
}

// 画一个闭合四边形
void draw_quad(cv::Mat& img, const std::array<cv::Point2f, 4>& q, const cv::Scalar& color,
               int thickness) {
  for (int k = 0; k < 4; ++k)
    cv::line(img, q[static_cast<size_t>(k)], q[static_cast<size_t>((k + 1) % 4)], color, thickness,
             cv::LINE_AA);
}

// ---------------------------------------------------------------- 中文文字
// cv::putText 的内置字体只有 ASCII，写中文会变成一串问号，所以这里用 OpenCV 的 freetype 模块
// （需要在 CMakeLists 的 find_package(OpenCV COMPONENTS ...) 里加 freetype）。
// 字体找不到时自动退回英文 putText —— 保证换一台机器也绝不会崩。
inline cv::Ptr<cv::freetype::FreeType2> ft_handle() {
  static bool tried = false;
  static cv::Ptr<cv::freetype::FreeType2> ft;
  if (!tried) {
    tried = true;
    const char* fonts[] = {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                           "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"};
    for (const char* f : fonts) {
      auto h = cv::freetype::createFreeType2();
      try {
        h->loadFontData(f, 0);
        ft = h;
        break;
      } catch (const cv::Exception&) {
        // 这个字体不存在/加载失败，试下一个
      }
    }
    std::printf("[显示] 中文字体：%s\n", ft ? "已加载" : "没找到，画面退回英文");
  }
  return ft;
}

// 画一行中文（px = 字高像素）
inline void put_text_cn(cv::Mat& img, const std::string& txt, cv::Point org, int px,
                        const cv::Scalar& color, int thickness = 1) {
  if (auto ft = ft_handle()) {
    ft->putText(img, txt, org, px, color, thickness, cv::LINE_AA, true);
  } else {
    cv::putText(img, txt, org, cv::FONT_HERSHEY_SIMPLEX, px / 26.0, color, thickness, cv::LINE_AA);
  }
}

// 状态名（中文）。终端日志仍用英文 state_name()，方便 grep。
inline const char* state_name_cn(TrackState s) {
  switch (s) {
    case TrackState::kLost: return "丢失";
    case TrackState::kDetecting: return "检测中";
    case TrackState::kTracking: return "跟踪中";
    case TrackState::kTempLost: return "暂时丢失";
  }
  return "未知";
}

// 检测类别号 → 可读标签（"蓝3号" / "红5号" / "蓝前哨站" …）。
// 三个容易混的概念，面板上只显示第一个：
//   class_id    ：检测网络的类别号（0~4 = 蓝 1~5 号，7~11 = 红 1~5 号，5/12 前哨站，6/13 基地）
//   armor_num   ：这台车上有**几块**装甲板（3 或 4，见 armor_num_from_class）——不是编号
//   plate_index ：环模型里"第几块"（0~3，关联/预测内部用，画面不显示）
inline const char* class_tag_cn(int cls) {
  static char buf[24];
  if (cls >= 0 && cls <= 4) {
    std::snprintf(buf, sizeof(buf), "蓝%d号", cls + 1);
  } else if (cls >= 7 && cls <= 11) {
    std::snprintf(buf, sizeof(buf), "红%d号", cls - 6);
  } else if (cls == 5 || cls == 12) {
    return (cls == 5) ? "蓝前哨站" : "红前哨站";
  } else if (cls == 6 || cls == 13) {
    return (cls == 6) ? "蓝基地" : "红基地";
  } else {
    std::snprintf(buf, sizeof(buf), "类别%d", cls);
  }
  return buf;
}

}  // namespace

// ---------------------------------------------------------------- 主函数
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
                                       TimePoint timestamp, const Eigen::Quaterniond& imu_q,
                                       const CameraCalibration& calibration,
                                       TrackingState& state) {
  // ===== 实现区 BEGIN =====
  if (!calibration.valid()) return std::nullopt;

  // ① 每帧刷新求解器（内参/外参可能改；世界系基准姿态只在第一帧记录一次）
  state.solver.configure(calibration, imu_q);
  state.solver_ready = true;

  const double now = to_seconds(timestamp);

  // ② 本帧所有装甲板 → 世界系观测（只收 1~5 号数字板；基地/前哨站不参与自瞄）
  std::vector<PlateObs> observations;
  std::vector<int> class_ids;
  std::vector<int> armor_nums;
  std::vector<double> center_dist;
  observations.reserve(armors.size());
  const double cx = calibration.matrix.at<double>(0, 2);
  const double cy = calibration.matrix.at<double>(1, 2);
  for (const Armor& a : armors) {
    const bool digit = (a.class_id >= 0 && a.class_id <= 4) || (a.class_id >= 7 && a.class_id <= 11);
    if (!digit) continue;
    const PnpObservation ob = state.solver.solve(a, state.last_yaw_filtered);
    if (!ob.ok) continue;
    PlateObs po;
    po.z = Eigen::Vector4d(ob.p_world.x(), ob.p_world.y(), ob.p_world.z(), ob.yaw_world);
    po.plate_index = 0;   // 环模型会在 TargetTracker 里按 yaw 差自动纠正
    observations.push_back(po);
    class_ids.push_back(a.class_id);
    armor_nums.push_back(armor_num_from_class(a.class_id));
    cv::Point2f c(0.f, 0.f);
    for (const auto& p : a.corners) c += p;
    c *= 0.25f;
    center_dist.push_back(cv::norm(c - cv::Point2f(static_cast<float>(cx),
                                                   static_cast<float>(cy))));
    if (kVerbosePnp)
      std::printf("[PnP] cls=%2d dist=%.3f m reproj=%.2f px world=(%.3f, %.3f, %.3f) yaw=%.1f°\n",
                  a.class_id, ob.distance, ob.reproj_px, ob.p_world.x(), ob.p_world.y(),
                  ob.p_world.z(), ob.yaw_world * 180.0 / kPi);
  }
  state.last_armor_count =
      static_cast<int>(std::min<size_t>(armors.size(), state.last_armors.size()));
  for (int i = 0; i < state.last_armor_count; ++i)
    state.last_armors[static_cast<size_t>(i)] = armors[static_cast<size_t>(i)];

  // ③ 交给多目标管理器（内部完成 predict / 关联 / 状态机）
  state.manager.set_observation_meta(class_ids, armor_nums);
  TargetTracker* tr = state.manager.update(now, observations, center_dist);
  if (tr == nullptr) {
    state.has_filtered = false;
    state.trail.clear();
    return std::nullopt;
  }

  // ④ 记录滤波结果（供误差配对与显示使用）
  const Eigen::Vector3d p_filt = tr->bank().plate_position();
  state.last_filtered = p_filt;
  state.last_yaw_filtered = tr->bank().yaw();
  state.has_filtered = true;
  state.trail.push_back(p_filt);
  while (state.trail.size() > 90) state.trail.pop_front();   // ≈1.5 s @60FPS

  // ⑤ 250 ms 预测 + 误差配对（作业要求⑥）
  // 只在"这一帧确实看到了靶"时记录预测样本：纯滑行帧（掉检）不参与统计，否则统计里混进了"用几帧前的状态外推"的样本，看起来像预测不准
  if (tr->tracking() && !observations.empty())
    tr->push_prediction(tr->make_prediction(now, kPredictHorizonSec));
  if (const auto due = tr->take_due_prediction(now)) {
    // 同时要求"这一帧也有观测"：否则拿几帧前的观测当参照，统计就不是"预测 vs 实测"了
    if (const auto& obs = (!observations.empty() ? tr->last_obs() : std::nullopt)) {
      const Eigen::Vector3d p_true = obs->z.head<3>();
      // 整车环模型：4 块板都预测了，误差取"离实测最近的那块预测"（看到哪块都能对上）
      state.last_pred_err_mm = due->min_distance_to(p_true) * 1000.0;
      state.pred_err_sum += state.last_pred_err_mm;
      state.pred_err_sq_sum += state.last_pred_err_mm * state.last_pred_err_mm;
      ++state.pred_err_n;

      double hw = 0.0, hh = 0.0;
      ArmorSolver::armor_half_size(false, hw, hh);

      // 参照框：本帧"同编号"的实测角点（没有就退回用观测位姿反投影）
      std::array<cv::Point2f, 4> px_true{};
      bool have_true = false;
      for (int i = 0; i < state.last_armor_count; ++i) {
        if (state.last_armors[static_cast<size_t>(i)].class_id == tr->class_id()) {
          px_true = state.last_armors[static_cast<size_t>(i)].corners;
          have_true = true;
          break;
        }
      }
      if (!have_true) px_true = state.solver.world_to_pixel(p_true, obs->z[3], hw, hh);
      cv::Point2f ct(0.f, 0.f);
      for (const auto& q : px_true) ct += q;
      ct *= 0.25f;

      // 预测框：4 块板里挑"中心离实测框最近"的那个（同样避免"比错了板"）
      std::array<cv::Point2f, 4> px_pred{};
      double best_px = 1e18;
      for (int i = 0; i < due->n; ++i) {
        const auto cand = state.solver.world_to_pixel(due->p[static_cast<size_t>(i)],
                                                      due->yaw[static_cast<size_t>(i)], hw, hh);
        cv::Point2f cc(0.f, 0.f);
        for (const auto& q : cand) cc += q;
        cc *= 0.25f;
        const double d = cv::norm(cc - ct);
        if (d < best_px) {
          best_px = d;
          px_pred = cand;
        }
      }
      cv::Point2f cp(0.f, 0.f);
      for (const auto& q : px_pred) cp += q;
      cp *= 0.25f;
      state.last_px_err = cv::norm(cp - ct);
      state.px_err_sum += state.last_px_err;
      state.px_err_sq_sum += state.last_px_err * state.last_px_err;
      ++state.px_err_n;
    }
  }

  // ⑥ 状态行（每 kPrintEveryMs 打一次，避免 I/O 拖慢主循环）
  const double now_ms = now * 1000.0;
  if (now_ms - state.last_print_ms >= kPrintEveryMs) {
    state.last_print_ms = now_ms;
    const double rms_mm =
        (state.pred_err_n > 0) ? std::sqrt(state.pred_err_sq_sum / state.pred_err_n) : -1.0;
    const double rms_px =
        (state.px_err_n > 0) ? std::sqrt(state.px_err_sq_sum / state.px_err_n) : -1.0;
    std::printf(
        "[L4] %-9s cls=%2d pos=(%6.3f,%6.3f,%6.3f) v=(%6.2f,%6.2f,%6.2f) yaw=%7.1f° "
        "sigma=%.0fmm | 250ms误差: 3D RMS=%.1fmm 像素 RMS=%.1fpx (n=%d)\n",
        state_name(tr->state()), tr->class_id(), p_filt.x(), p_filt.y(), p_filt.z(),
        tr->bank().plate_velocity().x(), tr->bank().plate_velocity().y(),
        tr->bank().plate_velocity().z(), state.last_yaw_filtered * 180.0 / kPi,
        tr->bank().position_sigma() * 1000.0, rms_mm, rms_px, state.pred_err_n);
    std::fflush(stdout);
  }

  // ⑦ 输出给第五讲
  Target target;
  target.position = p_filt;
  target.velocity = tr->bank().plate_velocity();
  return target;
  // ===== 实现区 END =====
}

// ---------------------------------------------------------------- 可视化
// 颜色：绿=检测  青=EKF 结果  黄粗=250 ms 主预测  （橙细=其余 3 块，默认关闭）  红=轨迹尾巴
void lesson04_display(cv::Mat image, const TrackingState& state) {
  if (image.empty() || !state.solver_ready) return;

  // ① 原始检测（绿）
  for (int i = 0; i < state.last_armor_count; ++i)
    draw_quad(image, state.last_armors[static_cast<size_t>(i)].corners, cv::Scalar(0, 255, 0), 1);

  const TargetTracker* tr = state.manager.selected();
  if (tr == nullptr || !state.has_filtered) return;

  double hw = 0.0, hh = 0.0;
  ArmorSolver::armor_half_size(false, hw, hh);
  const Eigen::Vector3d p_w = state.last_filtered;
  const double yaw_w = state.last_yaw_filtered;

  // ② EKF 结果（青）
  const auto px_f = state.solver.world_to_pixel(p_w, yaw_w, hw, hh);
  draw_quad(image, px_f, cv::Scalar(255, 255, 0), 2);

  // ③ 250 ms 预测（黄）：只画"最朝向相机"的那块主预测（粗黄框 + 位移箭头）。
  //    其余 3 块板的橙色细框默认**关闭**（画面更干净、录屏更好看）；
  //    想看"4 块板全覆盖"的效果，把 kDrawOtherPlates 改成 true。
  constexpr bool kDrawOtherPlates = false;
  if (tr->tracking()) {
    const PredictSample ps = tr->make_prediction(0.0, kPredictHorizonSec);
    std::array<cv::Point2f, 4> px_primary{};
    bool have_primary = false;
    for (int i = 0; i < ps.n; ++i) {
      const auto px_i = state.solver.world_to_pixel(ps.p[static_cast<size_t>(i)],
                                                    ps.yaw[static_cast<size_t>(i)], hw, hh);
      if (i == ps.primary) {
        px_primary = px_i;
        have_primary = true;
        draw_quad(image, px_i, cv::Scalar(0, 255, 255), 2);
      } else if (kDrawOtherPlates) {
        draw_quad(image, px_i, cv::Scalar(0, 165, 255), 1);
      }
    }
    if (have_primary)
      cv::arrowedLine(image, cv::Point(static_cast<int>(px_f[0].x), static_cast<int>(px_f[0].y)),
                      cv::Point(static_cast<int>(px_primary[0].x), static_cast<int>(px_primary[1].y)),
                      cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.25);
  }

  // ⑤ 轨迹尾巴（红）：把历史位置逐个投回图像
  std::vector<cv::Point> trail_px;
  trail_px.reserve(state.trail.size());
  for (const auto& p : state.trail) {
    const auto quad = state.solver.world_to_pixel(p, yaw_w, hw, hh);
    cv::Point2f c(0.f, 0.f);
    for (const auto& q : quad) c += q;
    c *= 0.25f;
    trail_px.emplace_back(static_cast<int>(c.x), static_cast<int>(c.y));
  }
  for (size_t i = 1; i < trail_px.size(); ++i)
    cv::line(image, trail_px[i - 1], trail_px[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

  // ⑥ 左下角信息面板
  const double rms3d =
      (state.pred_err_n > 0) ? std::sqrt(state.pred_err_sq_sum / state.pred_err_n) : -1.0;
  const double rmspx =
      (state.px_err_n > 0) ? std::sqrt(state.px_err_sq_sum / state.px_err_n) : -1.0;
  const cv::Scalar c = color_for_state(tr->state());
  char line1[200], line2[200], line3[200];
  std::snprintf(line1, sizeof(line1), "状态=%s  目标=%s（cls=%d）  样本=%d",
                state_name_cn(tr->state()), class_tag_cn(tr->class_id()), tr->class_id(),
                state.pred_err_n);
  if (state.last_pred_err_mm >= 0.0) {
    std::snprintf(line2, sizeof(line2), "250ms 预测误差=%.0f mm（均方根 %.0f mm）",
                  state.last_pred_err_mm, rms3d);
    std::snprintf(line3, sizeof(line3), "像素误差=%.1f px（均方根 %.1f px）", state.last_px_err,
                  rmspx);
  } else {
    std::snprintf(line2, sizeof(line2), "250ms 预测误差=--（均方根 %.0f mm）", rms3d);
    std::snprintf(line3, sizeof(line3), "像素误差=--（均方根 %.1f px）", rmspx);
  }
  const int y0 = image.rows - 18;
  put_text_cn(image, line1, cv::Point(10, y0 - 78), 26, c);
  put_text_cn(image, line2, cv::Point(10, y0 - 42), 26, c);
  put_text_cn(image, line3, cv::Point(10, y0 - 6), 26, c);

  // 图例（中文）
  put_text_cn(image, "绿=检测  青=滤波  黄=250ms预测  红=轨迹", cv::Point(10, 30), 24,
              cv::Scalar(255, 255, 255));
}

}  // namespace training
