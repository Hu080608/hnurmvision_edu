// 主程序：采集 → 检测 → 位姿 → 估计/预测 → 弹道 → 火控（三线程流水线）
//
// 数据流：
//   [IO 线程]  ROS 回调只做像素拷贝，保留最新一帧
//   [检测线程] 取最新帧 → letterbox → ONNX/TensorRT → 角点 → PnP → 世界系观测
//   [控制线程] 观测入估计器 → 预测命中时刻 → 角度/弹道 → 安全限幅 → 10 Hz 连续开火
#include "rm_aim/config.hpp"
#include "rm_aim/detector.hpp"
#include "rm_aim/lightbar.hpp"
#include "rm_aim/viz.hpp"
#include "rm_aim/estimator.hpp"
#include "rm_aim/io.hpp"
#include "io_context.hpp"    // 仓库讲义 IO（兼容线程用，只为保持原有控制台输出）
#include "lessons.hpp"       // lesson03/04/05

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <deque>
#include <mutex>
#include <vector>
#include <thread>

namespace rmaim {
bool rmaim_verbose() {
  static const bool v = std::getenv("RMAIM_VERBOSE") != nullptr;
  return v;
}
}  // namespace rmaim

namespace {
inline Eigen::Quaterniond quat_from_rpy_deg_local(double roll, double pitch, double yaw) {
  const double kRad = 3.14159265358979323846 / 180.0;
  const Eigen::AngleAxisd qz(yaw * kRad, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd qy(pitch * kRad, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd qx(roll * kRad, Eigen::Vector3d::UnitX());
  return qz * qy * qx;
}

constexpr double kPi = 3.14159265358979323846;
using rmaim::rmaim_verbose;
std::atomic<bool> g_stop{false};
std::atomic<long long> g_win_total{0}, g_win_open{0};   // 命中时刻门控的开/关统计
std::atomic<double> g_fps{0.0}, g_det_pct{0.0};         // 供可视化显示
// 门控阻塞原因统计（1 Hz 打印一次，用来定位"射速为什么上不去"）
struct GateStats {
  std::mutex mu;
  long long n = 0, no_settled = 0, no_sigma = 0, no_range = 0, no_fresh = 0, no_face = 0, no_window = 0;
  std::vector<double> yaw_errs;
  void add(bool settled, bool sigma_ok, bool range_ok, bool fresh_ok, bool face_ok, bool window_ok,
           double yaw_err) {
    std::lock_guard<std::mutex> lk(mu);
    ++n;
    if (!settled) ++no_settled;
    if (!sigma_ok) ++no_sigma;
    if (!range_ok) ++no_range;
    if (!fresh_ok) ++no_fresh;
    if (!face_ok) ++no_face;
    if (!window_ok) ++no_window;
    yaw_errs.push_back(yaw_err);
    if (yaw_errs.size() > 4000) yaw_errs.erase(yaw_errs.begin(), yaw_errs.begin() + 2000);
  }
  void report() {
    std::lock_guard<std::mutex> lk(mu);
    if (n < 50) return;
    std::sort(yaw_errs.begin(), yaw_errs.end());
    auto q = [&](double p) {
      if (yaw_errs.empty()) return 0.0;
      return yaw_errs[static_cast<size_t>(p * (yaw_errs.size() - 1))];
    };
    if (rmaim_verbose()) std::printf("[gate] n=%lld 阻塞: settled=%lld sigma=%lld range=%lld fresh=%lld face=%lld window=%lld "
                "| |yaw_des-fb| p50=%.3f p95=%.3f deg\n",
                n, no_settled, no_sigma, no_range, no_fresh, no_face, no_window, q(0.5), q(0.95));
    n = no_settled = no_sigma = no_range = no_fresh = no_face = no_window = 0;
    yaw_errs.clear();
  }
};
static GateStats g_gate;
void on_signal(int) { g_stop = true; }

double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

Eigen::Matrix3d rot_z(double r) {
  Eigen::Matrix3d m;
  m << std::cos(r), -std::sin(r), 0, std::sin(r), std::cos(r), 0, 0, 0, 1;
  return m;
}
Eigen::Matrix3d rot_y(double r) {
  Eigen::Matrix3d m;
  m << std::cos(r), 0, std::sin(r), 0, 1, 0, -std::sin(r), 0, std::cos(r);
  return m;
}
Eigen::Matrix3d rot_x(double r) {
  Eigen::Matrix3d m;
  m << 1, 0, 0, 0, std::cos(r), -std::sin(r), 0, std::sin(r), std::cos(r);
  return m;
}

// 反馈角（度）→ 云台到世界系的旋转：R = Rz(yaw)·Ry(pitch)·Rx(roll)
Eigen::Matrix3d imu_rot(double yaw_deg, double pitch_deg, double roll_deg) {
  const double k = kPi / 180.0;
  return rot_z(yaw_deg * k) * rot_y(pitch_deg * k) * rot_x(roll_deg * k);
}

double wrap180(double a) {
  while (a > 180.0) a -= 360.0;
  while (a < -180.0) a += 360.0;
  return a;
}

// 分阶段延迟统计（毫秒）：采集→到达、到达→检完、检完→决策
struct StageStats {
  std::mutex mu;
  std::vector<double> transport, detect, decide, total;
  void add_transport(double v) { std::lock_guard<std::mutex> lk(mu); transport.push_back(v); }
  void add_detect(double v) { std::lock_guard<std::mutex> lk(mu); detect.push_back(v); }
  void add_decide(double v) { std::lock_guard<std::mutex> lk(mu); decide.push_back(v); }
  void add_total(double v) { std::lock_guard<std::mutex> lk(mu); total.push_back(v); }
  void add(double t, double d, double c) {
    std::lock_guard<std::mutex> lk(mu);
    transport.push_back(t); detect.push_back(d); decide.push_back(c); total.push_back(t + d + c);
  }
  static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    size_t i = static_cast<size_t>(p * (s.size() - 1));
    return s[i];
  }
  void report(const char* tag) {
    std::vector<double> tr, de, dc, to;
    {
      std::lock_guard<std::mutex> lk(mu);
      tr.swap(transport); de.swap(detect); dc.swap(decide); to.swap(total);
    }
    if (to.empty()) return;
    if (rmaim_verbose()) std::printf("[lat] %s 到达→检完 p50=%.1f p95=%.1f | 检完→决策 p50=%.2f p95=%.2f | "
                "到达→决策 p50=%.1f p95=%.1f ms (n=%zu)\n",
                tag, pct(de, 0.5), pct(de, 0.95), pct(dc, 0.5), pct(dc, 0.95),
                pct(to, 0.5), pct(to, 0.95), to.size());
  }
};
static StageStats g_stage;
static double g_clock_offset = 1e18;   // steady ≈ ros + offset（用最小值法估计）

struct SharedObs {
  std::mutex mu;
  rmaim::Observation obs;
  bool fresh = false;
  std::vector<rmaim::Observation> batch;   // 本帧所有机器人装甲板
  int batch_n = 0;                         // 本帧 batch 里的板数（诊断）
  double cap_stamp = 0.0;    // 本帧采集时刻（ROS 时间）
  double detect_done = 0.0;  // 检出完成时刻（steady）
  double arrive_steady = 0.0; // 本帧到达（回调）时刻（steady）
  double obs_arrive = 0.0;    // 与 shared.obs 同一帧的到达时刻
  // 统计
  double det_ms = 0, pnp_ms = 0, last_frame_age_ms = 0;
  int frames = 0, detected = 0;
  double conf = 0, face = 0;
  int cls = -1;
  cv::Mat last_image;
};

// 发布指令 + 延迟分解统计。
// 只在"新帧驱动的那一拍"记录样本：控制循环是 2 kHz，而图像是 ~124 Hz，
// 若每拍都记，样本会被同一帧的重复决策稀释。
void publish_and_stat(rmaim::Io& io, const rmaim::Command& cmd, SharedObs& shared,
                      StageStats& stage, bool count_stat) {
  const double t_pub = now_sec();
  io.publish(cmd);
  if (!count_stat) return;
  double det_done = 0.0, arrive = 0.0;
  {
    std::lock_guard<std::mutex> lk(shared.mu);
    det_done = shared.detect_done;
    arrive = shared.obs_arrive;
  }
  if (det_done > 0.0) stage.add_decide((t_pub - det_done) * 1000.0);
  if (arrive > 0.0) stage.add_total((t_pub - arrive) * 1000.0);
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  const std::string cfg_path = argc > 1 ? argv[1] : "configs/simulation.yaml";


  rmaim::Config cfg;
  try {
    const training::Config scfg = training::load_config(cfg_path);
    const auto& m = scfg.camera_matrix;             // 行优先 9 个数
    cfg.camera.fx = m[0];
    cfg.camera.fy = m[4];
    cfg.camera.cx = m[2];
    cfg.camera.cy = m[5];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) cfg.camera.R_camera2gimbal(r, c) = scfg.camera_to_gimbal[r * 3 + c];
    cfg.camera.t_camera2gimbal = Eigen::Vector3d(scfg.camera_to_gimbal_translation[0],
                                                 scfg.camera_to_gimbal_translation[1],
                                                 scfg.camera_to_gimbal_translation[2]);
    cfg.camera.dist.assign(scfg.distort_coeffs.begin(), scfg.distort_coeffs.end());
    cfg.display = scfg.display;
  } catch (const std::exception& e) {
    std::printf("[fatal] 读取 %s 失败：%s\n", cfg_path.c_str(), e.what());
    return 2;
  }
  // ---- 可视化开关（不需要改代码/重编译）----
  //   AIM_VIZ=1              → 弹 aim_viz 四区块窗口
  //   AIM_VIZ_SAVE=<目录>    → 只把画布周期存成 PNG（不弹窗，适合评测/写文档）
  if (std::getenv("AIM_VIZ")) cfg.viz.enable = true;
  if (const char* vd = std::getenv("AIM_VIZ_SAVE")) cfg.viz.save_dir = vd;

  if (rmaim_verbose())
    std::printf("[cfg] fx=%.2f cx=%.1f cy=%.1f conf=%.2f pitch_zero=%.2f calib=%.2f gate=%.2f\n",
                cfg.camera.fx, cfg.camera.cx, cfg.camera.cy, cfg.detector.conf,
                cfg.aim.pitch_zero_deg, cfg.aim.pitch_calib_deg, cfg.aim.fire_angle_gate_deg);

  rclcpp::init(argc, argv);
  rmaim::Io io;
  io.start();

  // ---- 兼容旧控制台输出的开关与状态（只为打印 [IO]/[armor]/[L4]，不参与控制）----
  const bool legacy_logs = (std::getenv("RMAIM_NO_LEGACY_LOG") == nullptr);
  training::TrackingState tracking_legacy;
  training::CameraCalibration legacy_calib;
  for (const char* cand : {"configs/simulation.yaml",
                           "/home/hu/hnurmvision_edu/configs/simulation.yaml"}) {
    try { legacy_calib = training::load_calibration(training::load_config(cand)); break; }
    catch (const std::exception&) {}
  }

  rmaim::Detector detector(cfg.detector);
  rmaim::LightBarDetector lbar(cfg.lightbar);
  rmaim::AimViz viz(cfg.viz);
  rmaim::Estimator est(cfg.aim);
  SharedObs shared;

  double last_sel_u = -1.0, last_sel_v = -1.0;
  bool have_sel = false;   // 是否已有锁定位置（用于 ROI 检测）
  bool have_last_obs = false;
  int switch_count = 0;
  std::atomic<int> lbar_frames{0}, lbar_plates{0}, lbar_added{0};
  std::deque<double> range_hist;
  double med_range = 0.0;

  // ---------- 检测线程 ----------
  std::thread det_thread([&] {
    while (!g_stop) {
      rmaim::Frame frame;
      if (!io.take_frame(frame)) {
        std::this_thread::sleep_for(std::chrono::microseconds(300));
        continue;
      }
      const double t_frame = frame.stamp_sec;
      const auto fb_now = io.feedback_at(t_frame);   // 该帧时刻的云台反馈（供兼容日志）
      const double t_arrive = frame.stamp_sec;   // 回调时刻（steady）
      {
        std::lock_guard<std::mutex> lk(shared.mu);
        shared.arrive_steady = t_arrive;
      }
      if (frame.bgr.cols < 64 || frame.bgr.rows < 64) continue;   // 换仿真器后可能出现异常小图
      // ---- ROI 二次检测：以上次锁定的板为中心裁一块小图（等效放大），提高检出率 ----
      cv::Mat work = frame.bgr;
      int ox = 0, oy = 0;
      static int frame_no = 0;
      ++frame_no;
      const bool want_roi = cfg.detector.roi_enable && have_sel &&
                            (cfg.detector.roi_full_every <= 1 || frame_no % cfg.detector.roi_full_every != 0);
      if (want_roi) {
        const int W = frame.bgr.cols, H = frame.bgr.rows;
        const int cw = std::min(W, std::max(128, cfg.detector.roi_w));
        const int ch = std::min(H, std::max(128, cfg.detector.roi_h));
        int x0 = static_cast<int>(std::lround(last_sel_u - cw * 0.5));
        int y0 = static_cast<int>(std::lround(last_sel_v - ch * 0.5));
        x0 = std::clamp(x0, 0, W - cw);
        y0 = std::clamp(y0, 0, H - ch);
        work = frame.bgr(cv::Rect(x0, y0, cw, ch));
        ox = x0;
        oy = y0;
      }
      std::vector<rmaim::Detection> dets;
      try {
        dets = detector.detect(work);
        if (dets.empty() && (ox || oy)) {          // ROI 里没找到 → 立即回退全图
          dets = detector.detect(frame.bgr);
          ox = oy = 0;
        } else if (ox || oy) {
          for (auto& d : dets) {                   // ROI 坐标 → 原图坐标
            for (auto& kp : d.corners) { kp.x += static_cast<float>(ox); kp.y += static_cast<float>(oy); }
            d.box.x += static_cast<float>(ox);
            d.box.y += static_cast<float>(oy);
          }
        }
      } catch (const std::exception& e) {
        static int err_n = 0;
        if (++err_n % 100 == 1) if (rmaim_verbose()) std::printf("[warn] 检测异常：%s\n", e.what());
        continue;
      }
      const double t_after_det = now_sec();

      if (!dets.empty()) {
        // ---- 逐块做 PnP，再按「正对程度 + 置信度 + 连续性」挑一块 ----
        struct Cand {
          double u = 0, v = 0, w = 0, range = 0, conf = 0, score = -1e9;
          double az = 0, el = 0;
          Eigen::Vector3d dir = Eigen::Vector3d::UnitX();
          int cls = -1;
        };
        std::vector<Cand> cands;
        const double fx = cfg.camera.fx;
        // 该帧时刻的云台姿态（时间对齐！否则快照与反馈差 20~30 ms → 世界角漂移）
        const rmaim::Feedback fb_f = io.feedback_at(t_frame);
        const Eigen::Matrix3d R_w = imu_rot(fb_f.yaw_deg, fb_f.pitch_deg, fb_f.roll_deg);
        for (const auto& det : dets) {
          // 类别过滤：前哨站/基地的装甲不算自瞄目标（否则会被牵引去打场地建筑）
          if (std::find(cfg.detector.ignore_classes.begin(), cfg.detector.ignore_classes.end(),
                        det.cls) != cfg.detector.ignore_classes.end()) {
            continue;
          }
          const double w_px = det.box.width;
          if (w_px < cfg.detector.min_box_w_px) continue;
          const cv::Point2f c = det.center();
          const double plate_h = det.large ? cfg.aim.plate_height_large : cfg.aim.plate_height_small;
          const double h_px = std::max(6.0f, det.box.height);
          const double range = fx * plate_h / static_cast<double>(h_px);
          if (!std::isfinite(range) || range < 0.4 || range > 25.0) continue;
          const Eigen::Vector3d ray_cam((c.x - cfg.camera.cx) / fx, (c.y - cfg.camera.cy) / fx, 1.0);
          Eigen::Vector3d dir_w = (R_w * (cfg.camera.R_camera2gimbal * ray_cam)).normalized();
          if (!dir_w.allFinite() || dir_w.norm() < 1e-6) continue;

          Cand c2;
          c2.u = c.x; c2.v = c.y; c2.w = w_px; c2.range = range; c2.conf = det.conf; c2.cls = det.cls;
          c2.az = std::atan2(dir_w.y(), dir_w.x());
          c2.el = std::asin(std::clamp(dir_w.z(), -1.0, 1.0));
          c2.dir = dir_w;
          // 选板规则：正对的板在图上最宽（apparent width ∝ face-on），
          // 因此以"框宽"为主项；连续性只作为小权重，避免同一块板被高频抢占。
          double cont = 0.0;
          if (last_sel_u >= 0) {
            const double du = c2.u - last_sel_u, dv = c2.v - last_sel_v;
            cont = 0.55 * std::exp(-std::sqrt(du * du + dv * dv) / 60.0);
          }
          c2.score = 1.0 * (w_px / 50.0) + 0.35 * c2.conf + cont;
          cands.push_back(c2);
        }
        double w_max_frame = 0.0;
        for (const auto& det : dets) {
          const bool ignored = std::find(cfg.detector.ignore_classes.begin(),
                                         cfg.detector.ignore_classes.end(),
                                         det.cls) != cfg.detector.ignore_classes.end();
          if (!ignored) w_max_frame = std::max(w_max_frame, static_cast<double>(det.box.width));
        }
        // 多板联合估计：用【宽松门槛】把本帧所有可能的装甲板打包（不进主选板路径）
        {
          std::vector<rmaim::Observation> batch;
          batch.reserve(dets.size() + cands.size());
          for (const auto& det : dets) {
            const bool ignored = std::find(cfg.detector.ignore_classes.begin(),
                                           cfg.detector.ignore_classes.end(),
                                           det.cls) != cfg.detector.ignore_classes.end();
            if (ignored) continue;
            if (det.conf < cfg.detector.conf_loose) continue;
            if (det.box.width < cfg.detector.min_box_w_loose) continue;
            const double plate_h = det.large ? cfg.aim.plate_height_large : cfg.aim.plate_height_small;
            const double h_px = std::max(6.0f, det.box.height);
            const double rng = fx * plate_h / static_cast<double>(h_px);
            if (!std::isfinite(rng) || rng < 0.4 || rng > 25.0) continue;
            const cv::Point2f c = det.center();
            const Eigen::Vector3d ray_cam((c.x - cfg.camera.cx) / fx, (c.y - cfg.camera.cy) / fx, 1.0);
            const Eigen::Vector3d dir_w = (R_w * (cfg.camera.R_camera2gimbal * ray_cam)).normalized();
            if (!dir_w.allFinite()) continue;
            rmaim::Observation ob;
            ob.t = t_frame;
            ob.az = std::atan2(dir_w.y(), dir_w.x());
            ob.el = std::asin(std::clamp(dir_w.z(), -1.0, 1.0));
            ob.range = rng;
            ob.w_px = det.box.width;
            ob.w_max = w_max_frame;
            ob.pos = dir_w * rng;
            ob.conf = det.conf;
            batch.push_back(ob);
          }
          // ---- ① 灯条级观测：模型置信度不足时，用"灯条配对"补板 ----
          if (cfg.lightbar.enable) {
            const auto bar_plates = lbar.detect(frame.bgr);   // 全图坐标，与 ROI 无关
            lbar_frames.fetch_add(1);
            int added = 0;
            for (const auto& bp : bar_plates) {
              lbar_plates.fetch_add(1);
              // 模型已经检出的同一块板 → 跳过（模型带类别/关键点，质量更高）
              bool dup = false;
              for (const auto& d : dets) {
                const bool ignored = std::find(cfg.detector.ignore_classes.begin(),
                                               cfg.detector.ignore_classes.end(),
                                               d.cls) != cfg.detector.ignore_classes.end();
                if (ignored) continue;
                const cv::Point2f c = d.center();
                const double dd = std::hypot(c.x - bp.cx, c.y - bp.cy);
                if (dd < std::max(18.0, 0.8 * bp.h)) { dup = true; break; }
              }
              if (dup) continue;
              // 灯条高 → 距离：板高 = 灯条高 / bar_h_ratio
              const double h_px = std::max(6.0, static_cast<double>(bp.h));
              const double rng = fx * cfg.aim.plate_height_small * cfg.lightbar.bar_h_ratio / h_px;
              if (!std::isfinite(rng) || rng < 0.4 || rng > 25.0) continue;
              const Eigen::Vector3d ray_cam((bp.cx - cfg.camera.cx) / fx, (bp.cy - cfg.camera.cy) / fx, 1.0);
              const Eigen::Vector3d dir_w = (R_w * (cfg.camera.R_camera2gimbal * ray_cam)).normalized();
              if (!dir_w.allFinite()) continue;
              rmaim::Observation ob;
              ob.t = t_frame;
              ob.az = std::atan2(dir_w.y(), dir_w.x());
              ob.el = std::asin(std::clamp(dir_w.z(), -1.0, 1.0));
              ob.range = rng;
              ob.w_px = bp.w;
              ob.w_max = w_max_frame;
              ob.pos = dir_w * rng;
              ob.conf = cfg.lightbar.conf;
              batch.push_back(ob);
              ++added;
            }
            lbar_added.fetch_add(added);
          }
          if (!batch.empty()) {
            std::lock_guard<std::mutex> lk(shared.mu);
            shared.batch = batch;
            shared.batch_n = static_cast<int>(batch.size());
          }
        }
        // ---- 延迟分解：到达→检完（含预处理/推理/后处理 + batch/灯条）----
        const double t_det_done = now_sec();
        g_stage.add_detect((t_det_done - t_arrive) * 1000.0);
        if (!cands.empty()) {
          auto best_it = std::max_element(cands.begin(), cands.end(),
                                          [](const Cand& a, const Cand& b) { return a.score < b.score; });
          const Cand& best = *best_it;
          last_sel_u = best.u;
          last_sel_v = best.v;
          have_sel = true;
          range_hist.push_back(best.range);
          if (range_hist.size() > 40) range_hist.pop_front();
          {
            std::vector<double> tmp(range_hist.begin(), range_hist.end());
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            med_range = tmp[tmp.size() / 2];
          }
          const double t_after_det = now_sec();
          std::lock_guard<std::mutex> lk(shared.mu);
          shared.obs.t = t_frame;
          shared.obs.az = best.az;
          shared.obs.el = best.el;
          shared.obs.pos = best.dir * best.range;
          shared.obs.range = best.range;
          shared.obs.w_px = best.w;
          shared.obs.w_max = w_max_frame;
          shared.obs.conf = best.conf;
          shared.obs_arrive = t_arrive;
          shared.fresh = true;
          shared.det_ms = (t_after_det - t_frame) * 1000.0;
          shared.pnp_ms = 0.0;
          shared.last_frame_age_ms = (now_sec() - t_frame) * 1000.0;
          shared.conf = best.conf;
          shared.cls = best.cls;
          shared.cap_stamp = frame.stamp_ros;
          shared.detect_done = now_sec();
                  ++shared.detected;
        }
      } else {
        std::lock_guard<std::mutex> lk(shared.mu);
        shared.last_frame_age_ms = (now_sec() - t_frame) * 1000.0;
      }
      {
        std::lock_guard<std::mutex> lk(shared.mu);
        ++shared.frames;
        ++shared.frames;
        // ---------- 兼容旧控制台输出（复用本次检测结果，零额外推理）----------
        // 目的：保持和以前一样的 [IO]/[armor]/[L4] 日志。这些日志只打印，不参与控制。
        if (legacy_logs) {
          static bool banner_done = false;
          if (!banner_done) {
            banner_done = true;
            std::printf("[IO] 已连接：订阅 /image_raw、/vision_receive_data，发布 /vision_send_data\n");
            std::printf("[armor] 调试窗口=lesson03 armor | DISPLAY=%s\n",
                        std::getenv("DISPLAY") ? std::getenv("DISPLAY") : "(未设置，将不开窗)");
            std::printf("[armor] 模型=%s 输入=%dx%d 输出=[1,300,14] 端到端=是 后端=%s\n",
                        cfg.detector.model.c_str(), cfg.detector.input_size, cfg.detector.input_size,
                        detector.provider().c_str());
            std::printf("[armor] 图案分类器=assets/pattern_dos.onnx（S=基地 / O=前哨站）\n");
            std::fflush(stdout);
          }
          // 把 rm_aim 的检出转成讲义结构，喂 lesson04 只为打印 [L4]
          std::vector<training::Armor> armors;
          for (const auto& d : dets) {
            training::Armor a;
            a.corners = d.corners;
            a.class_id = d.cls;
            a.confidence = d.conf;
            a.large_armor = d.large;
            armors.push_back(a);
          }
          const auto imu_q = quat_from_rpy_deg_local(fb_now.roll_deg, fb_now.pitch_deg, fb_now.yaw_deg);
          const training::TimePoint tp =
              training::TimePoint(std::chrono::duration_cast<training::Clock::duration>(
                  std::chrono::duration<double>(t_frame)));
          const auto target = training::lesson04_tracking(
              armors, tp, imu_q, legacy_calib, tracking_legacy);
          if (target) {
            training::lesson05_set_attitude(imu_q, tracking_legacy.solver.R_world0());
            (void)training::lesson05_aiming(*target, tp, fb_now.bullet_speed);
          }
          static double t_io_log = -1e9;
          if (t_frame - t_io_log > 1.0) {
            t_io_log = t_frame;
            std::printf("[IO] yaw=%.2f deg pitch=%.2f deg bullet_speed=%.1f m/s\n",
                        fb_now.yaw_deg, fb_now.pitch_deg, fb_now.bullet_speed);
          }
        }
      }
    }
  });

  // ---------- 控制线程 ----------
  double last_pub = 0.0;
  double yaw_cmd = 0.0, pitch_cmd = 0.0;
  bool have_cmd = false;
  double last_obs_time = -1e9;
  double log_t = 0.0;
  int fire_count = 0, pub_count = 0;
  double sum_age = 0.0;
  int age_n = 0;

  while (!g_stop && rclcpp::ok()) {
   try {
    const double t_now = now_sec();
    bool fresh = false;
    rmaim::Observation obs;
    {
      std::lock_guard<std::mutex> lk(shared.mu);
      if (shared.fresh) {
        obs = shared.obs;
        shared.fresh = false;
        fresh = true;
      }
    }
    static rmaim::Observation last_obs;
    if (fresh) {
      last_obs = obs;
      est.add(obs);
      last_obs_time = obs.t;
      const double age = (t_now - obs.t) * 1000.0;
      sum_age += age;
      ++age_n;
    }
    {
      std::lock_guard<std::mutex> lk(shared.mu);
      if (!shared.batch.empty()) est.add_batch(shared.batch);
    }
    est.prune(t_now);

    auto fb = io.feedback();
    const bool tracking = est.tracking() && (t_now - last_obs_time) * 1000.0 < cfg.aim.lost_timeout_ms;

    // ---- 调试/标定模式：固定 yaw、固定 pitch，只开火（用于标定 pitch 映射与像素→角度关系）----
    static const char* hold_env = std::getenv("AIM_HOLD_PITCH");
    if (hold_env) {
      static double hold_yaw = 1e9;
      if (hold_yaw > 1e8) hold_yaw = fb.valid ? fb.yaw_deg : -90.0;
      rmaim::Command cmd;
      cmd.control = true;
      cmd.fire = true;
      cmd.yaw_deg = hold_yaw;
      cmd.pitch_deg = std::atof(hold_env);
      cmd.target_distance = est.current_range();
      publish_and_stat(io, cmd, shared, g_stage, fresh);
      static double last_dbg = 0;
      if (t_now - last_dbg > 0.5 && est.hits() > 0) {
        last_dbg = t_now;
        if (rmaim_verbose()) std::printf("[hold] pitch=%.2f obs az=%.3f el=%.3f r=%.2f\n", cmd.pitch_deg,
                    last_obs.az * 180 / kPi, last_obs.el * 180 / kPi, last_obs.range);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }

    if (tracking) {
      // ---- 预测命中时刻（迭代两次求自洽的飞行时间）----
      const double v0 = fb.valid ? fb.bullet_speed : cfg.aim.bullet_speed_default;
      const double latency = cfg.latency_fixed_ms / 1000.0 + (age_n ? (sum_age / age_n) / 1000.0 : 0.0);
      double t_hit = t_now + latency + 0.10;
      rmaim::Prediction pred;
      for (int it = 0; it < 2; ++it) {
        pred = est.predict(t_hit);
        if (!pred.valid) break;
        const double d = std::max(0.5, pred.range);
        const double t_f = d / std::max(1.0, v0);
        t_hit = t_now + cfg.aim.lead_scale * (latency + t_f);
        pred = est.predict(t_hit);
      }
      if (pred.valid && fb.valid && std::isfinite(pred.az) && std::isfinite(pred.el)) {
        const double fx = cfg.camera.fx;
        // 目标相对云台原点的视线：用预测的 (az, el) + 中位距离
        double rng = pred.range;
        if (!range_hist.empty()) {
          std::vector<double> tmp(range_hist.begin(), range_hist.end());
          std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
          rng = tmp[tmp.size() / 2];
        }
        rng = std::clamp(rng, 0.6, 12.0);
        const double d = std::max(0.5, rng * std::cos(pred.el));
        const double h = rng * std::sin(pred.el);
        double theta = 0.0, t_f = 0.0;
        const bool ball_ok = rmaim::ballistic_solution(d, h, v0, cfg.aim.gravity, theta, t_f);
        const double theta_deg = (ball_ok ? theta : pred.el) * 180.0 / kPi;

        const double yaw_raw = pred.az * 180.0 / kPi;
        const double pitch_raw = cfg.aim.pitch_zero_deg - theta_deg - cfg.aim.pitch_calib_deg;

        // ---- 抖动抑制 ----
        // 自转工况下瞄准点本质是常量（守驻留点），所以允许强平滑 + 速率限制：
        // 原来它跟着每帧估计跳，估计噪声/洋离群帧直接把云台甩出去 → 换靶后要抖 10 s 才稳。
        static double last_ctrl_t = 0.0;
        static double ring_yaw_des = 0.0, ring_pitch_des = 0.0;
        static bool ring_des_ok = false;
        const double dtc = (last_ctrl_t > 0.0) ? std::clamp(t_now - last_ctrl_t, 1e-4, 0.05) : 0.002;
        last_ctrl_t = t_now;
        double yaw_des = yaw_raw, pitch_des = pitch_raw;
        if (est.rotation_mode()) {
          const bool acq = est.acquiring();
          if (!ring_des_ok) { ring_yaw_des = yaw_raw; ring_pitch_des = pitch_raw; ring_des_ok = true; }
          const double vmax_yaw = acq ? 260.0 : 45.0;     // °/s：收敛期给足，稳态压死
          const double vmax_pit = acq ? 160.0 : 25.0;
          double d = wrap180(yaw_raw - ring_yaw_des);
          d = std::clamp(d, -vmax_yaw * dtc, vmax_yaw * dtc);
          ring_yaw_des = wrap180(ring_yaw_des + d);
          double dp = std::clamp(pitch_raw - ring_pitch_des, -vmax_pit * dtc, vmax_pit * dtc);
          ring_pitch_des += dp;
          yaw_des = ring_yaw_des;
          pitch_des = ring_pitch_des;
        } else {
          // 非环工况：只做速率限制（250°/s），挡住"单帧垃圾预测"把云台甩出去；
          // 平移靶本身的角速度只有 ~5°/s，250°/s 完全够用，不会拖慢对准。
          ring_des_ok = false;
          const double vmax = est.acquiring() ? 600.0 : 250.0;
          static double yaw_des_prev = 0.0, pitch_des_prev = 0.0;
          static double yaw_des_t = -1e9;
          if (t_now - yaw_des_t > 0.2) { yaw_des_prev = yaw_raw; pitch_des_prev = pitch_raw; }
          double d = wrap180(yaw_raw - yaw_des_prev);
          d = std::clamp(d, -vmax * dtc, vmax * dtc);
          yaw_des_prev = wrap180(yaw_des_prev + d);
          double dp = std::clamp(pitch_raw - pitch_des_prev, -vmax * dtc, vmax * dtc);
          pitch_des_prev += dp;
          yaw_des_t = t_now;
          yaw_des = yaw_des_prev;
          pitch_des = pitch_des_prev;
        }

        if (!have_cmd) { yaw_cmd = fb.yaw_deg; pitch_cmd = fb.pitch_deg; have_cmd = true; }
        double dyaw = wrap180(yaw_des - yaw_cmd);
        dyaw = std::clamp(dyaw, -cfg.aim.max_yaw_step_deg, cfg.aim.max_yaw_step_deg);
        yaw_cmd = wrap180(yaw_cmd + dyaw);
        double dpitch = pitch_des - pitch_cmd;
        dpitch = std::clamp(dpitch, -cfg.aim.max_pitch_step_deg, cfg.aim.max_pitch_step_deg);
        pitch_cmd = std::clamp(pitch_cmd + dpitch, -44.0, 44.0);

        // 用"枪口指向 vs 目标方向"的真实误差，而不是"限幅后的指令 vs 反馈"——
        // 后者在目标快速移动时会把大量本该开火的时刻判成"没跟上"。
        // ---- 舵机滞后补偿 ----
        // 原来比的是"期望角 vs 现在反馈"，舵机有滞后 → 快速平移时会误判"没跟上"而停火。
        // 现在比的是"子弹到达时刻枪口会在哪" = 反馈 + 反馈角速度 × 到达时间。
        static double fb_yaw_prev = 0.0, fb_pitch_prev = 0.0, fb_t_prev = -1e9;
        double rate_fb_yaw = 0.0, rate_fb_pitch = 0.0;
        if (fb.valid && fb_t_prev > -1e8) {
          const double dt_fb = t_now - fb_t_prev;
          if (dt_fb > 1e-4 && dt_fb < 0.2) {
            rate_fb_yaw = wrap180(fb.yaw_deg - fb_yaw_prev) / dt_fb;
            rate_fb_pitch = (fb.pitch_deg - fb_pitch_prev) / dt_fb;
          }
        }
        if (fb.valid) { fb_yaw_prev = fb.yaw_deg; fb_pitch_prev = fb.pitch_deg; fb_t_prev = t_now; }
        const double tau_arr = (t_hit > t_now) ? (t_hit - t_now) : 0.0;
        const double comp = cfg.aim.servo_lead_comp ? 1.0 : 0.0;
        const double yaw_err = std::fabs(wrap180(yaw_des - (fb.yaw_deg + comp * rate_fb_yaw * tau_arr)));
        const double pitch_err = std::fabs(pitch_des - (fb.pitch_deg + comp * rate_fb_pitch * tau_arr));
        const bool settled = yaw_err < cfg.aim.fire_angle_gate_deg &&
                             pitch_err < cfg.aim.fire_angle_gate_deg;
        const bool sigma_ok = pred.model_sigma * 180.0 / kPi < cfg.aim.fire_px_gate;  // 弧度→度
        const bool range_ok = rng > 0.5 && rng < cfg.aim.fire_range_gate_m;
        // 观测太旧就不开火：丢检期间预测会越飘越远，继续打只会白掉命中率
        const double obs_age_ms = (t_now - est.last_obs_time()) * 1000.0;
        double det_ms_now = 0.0;
        { std::lock_guard<std::mutex> lk(shared.mu); det_ms_now = shared.det_ms; }
        const bool fresh_ok = obs_age_ms < cfg.aim.fire_obs_age_gate_ms;

        rmaim::Command cmd;
        cmd.control = true;
        const bool face_ok2 = pred.face_on > 0.55;   // 环模型：板不够正对就别打
        // ★ 命中时刻对齐门控：环工况下必须"子弹到达时某块板正好会经过瞄准点"才开火
        const bool window_ok = pred.fire_window;
        if (est.rotation_mode()) { ++g_win_total; if (window_ok) ++g_win_open; }
        // 注：收敛期不需要额外"禁止开火"——瞄准点还没收到目标上时 yaw_err 本来就大，
        // 上面的 settled 门控已经会拦住（实测额外加 !acquiring() 会白丢射速）。
        cmd.fire = settled && sigma_ok && range_ok && fresh_ok && face_ok2 && window_ok;
        g_gate.add(settled, sigma_ok, range_ok, fresh_ok, face_ok2, window_ok,
                   std::fabs(wrap180(yaw_des - fb.yaw_deg)));
        cmd.yaw_deg = yaw_cmd;
        cmd.pitch_deg = pitch_cmd;
        cmd.target_distance = rng;
        publish_and_stat(io, cmd, shared, g_stage, fresh);
        last_pub = t_now;
        ++pub_count;
        if (cmd.fire) ++fire_count;

        // ---------- 可视化----------
        if (cfg.viz.enable || !cfg.viz.save_dir.empty()) {
          const double half_w = 0.5 * cfg.aim.plate_width_small / std::max(0.5, rng) * 180.0 / kPi;
          const double half_h = 0.5 * cfg.aim.plate_height_small / std::max(0.5, rng) * 180.0 / kPi;
          rmaim::VizFrame vf;
          vf.valid = true;
          vf.range = rng;
          vf.half_w_deg = half_w;
          vf.half_h_deg = half_h;
          vf.az_aim = pred.az;
          vf.el_aim = pred.el;
          vf.tracking = tracking;
          vf.fire = cmd.fire;
          vf.acquiring = est.acquiring();
          vf.ring = est.rotation_mode();
          vf.model = est.current_model();
          vf.az0 = est.center_az();
          vf.hw = est.ring_half_width();
          vf.omega = cfg.aim.ring_omega;
          for (int i = 0; i < 4; ++i) { vf.face[i] = 0.0; vf.plate_deg[i] = 0.0; }
          if (pred.plate_hit_ok) vf.face[0] = pred.plate_face;
          est.viz_faces(vf.face, vf.plate_deg, vf.phase);
          vf.ring_R = cfg.aim.ring_radius;
          vf.plate_w_m = cfg.aim.plate_width_small;
          vf.plate_t_m = 0.025;
          vf.phase_deg = vf.phase;
          // ---- 四区块新增数据 ----
          vf.has_vehicle = tracking;
          vf.yaw_cmd = yaw_cmd;
          vf.yaw_fb = fb.valid ? fb.yaw_deg : yaw_cmd;
          vf.pitch_cmd = pitch_cmd;
          vf.pitch_fb = fb.valid ? fb.pitch_deg : pitch_cmd;
          vf.obs_age_ms = obs_age_ms;
          vf.detect_ms = det_ms_now;
          vf.infer_ms = detector.last_infer_ms();
          vf.e2e_ms = det_ms_now + obs_age_ms;   // 到达→决策的近似（检测+排队+控制）
          vf.sigma_deg = pred.model_sigma * 180.0 / kPi;
          vf.track_hits = est.hits();
          vf.batch_n = shared.batch_n;
          {
            static int duty_on = 0, duty_n = 0;
            static double duty_t = -1e9, duty_last = 0.0;
            ++duty_n;
            if (cmd.fire) ++duty_on;
            if (t_now - duty_t > 0.5) {
              duty_last = duty_n ? static_cast<double>(duty_on) / duty_n : 0.0;
              duty_t = t_now; duty_on = duty_n = 0;
            }
            vf.fire_duty = duty_last;
            static double shot_t = -1e9, rate_t = -1e9;
            static int rate_n = 0, rate_n0 = 0, rate_last = 0;
            if (cmd.fire && t_now - shot_t > 0.10) { shot_t = t_now; ++rate_n; }
            if (t_now - rate_t > 1.0) {
              if (rate_t > -1e8) rate_last = rate_n - rate_n0;
              rate_t = t_now; rate_n0 = rate_n;
            }
            vf.bullets_per_s = rate_last;
            vf.fps = g_fps.load();
            vf.det_pct = g_det_pct.load();
          }
          viz.push_trace(vf);
          // 每 100 ms 记一个"落点"（与仿真器 10 Hz 开火节拍对齐）
          static double last_shot_t = -1e9;
          if (cmd.fire && t_now - last_shot_t > 0.10) {
            last_shot_t = t_now;
            rmaim::VizImpact vi;
            if (pred.plate_hit_ok) {
              // 环工况：落点 = 瞄准点相对"命中时刻那块板"的偏移（板坐标）
              vi.dx = (pred.az - pred.plate_az_hit) * 180.0 / kPi / std::max(0.05, half_w);
              vi.dy = (pred.el - pred.plate_el_hit) * 180.0 / kPi / std::max(0.05, half_h);
            } else if (!last_obs.t) {
              vi.dx = vi.dy = 0;
            } else {
              // 常规工况：落点用"预测 vs 最新观测"的偏差（等价于跟踪误差）
              vi.dx = (pred.az - last_obs.az) * 180.0 / kPi / std::max(0.05, half_w);
              vi.dy = (pred.el - last_obs.el) * 180.0 / kPi / std::max(0.05, half_h);
              vi.dx = std::clamp(vi.dx, -2.5, 2.5);
              vi.dy = std::clamp(vi.dy, -2.5, 2.5);
            }
            vi.inside = std::fabs(vi.dx) < 1.0 && std::fabs(vi.dy) < 1.0;
            viz.impact(vi);
          }
          viz.update(vf);
          static double last_viz_t = -1e9;
          if (t_now - last_viz_t > 0.05) { last_viz_t = t_now; viz.render(); }
        }

        if (std::getenv("AIM_DEBUG")) {
          static int dbg_n = 0;
          if (++dbg_n % 12 == 0) {
            if (rmaim_verbose()) std::printf("[dbg] obs az=%.3f el=%.3f r=%.2f | pred az=%.3f el=%.3f model=%s "
                        "σ=%.3f° | d=%.2f h=%.2f θ=%.2f° | cmd yaw=%.2f pitch=%.2f | fb=(%.2f,%.2f)\n",
                        last_obs.az * 180 / kPi, last_obs.el * 180 / kPi, last_obs.range,
                        pred.az * 180 / kPi, pred.el * 180 / kPi, pred.model_name,
                        pred.model_sigma * 180 / kPi, d, h, theta_deg, yaw_cmd, pitch_cmd,
                        fb.yaw_deg, fb.pitch_deg);
            if (rmaim_verbose()) std::printf("[dbg2] %s | %s\n", est.diag(), est.phase_diag());
          }
        }
      }
    } else {
      // ---- 丢靶：保持当前姿态（不要发 target_state=0，否则仿真器会让云台停住）----
      const double lost_ms = (t_now - last_obs_time) * 1000.0;
      if (have_cmd && lost_ms < 1500.0) {
        rmaim::Command cmd;
        cmd.control = true;
        cmd.fire = false;
        cmd.yaw_deg = yaw_cmd;
        cmd.pitch_deg = pitch_cmd;
        publish_and_stat(io, cmd, shared, g_stage, fresh);
      }
    }

    // ---- 日志（1 Hz）----
    if (t_now - log_t > 1.0) {
      log_t = t_now;
      int frames = 0, detected = 0, cls = -1, batch_n = 0;
      double det_ms = 0, age_ms = 0, face = 0;
      {
        std::lock_guard<std::mutex> lk(shared.mu);
        frames = shared.frames;
        detected = shared.detected;
        det_ms = shared.det_ms;
        cls = shared.cls;
        face = shared.face;
        batch_n = shared.batch_n;
        age_ms = age_n ? sum_age / age_n : 0.0;
        shared.frames = 0;
        shared.detected = 0;
      }
      const double fps = static_cast<double>(frames);
      g_fps.store(fps);
      g_det_pct.store(frames ? 100.0 * detected / frames : 0.0);
      int trk_n = 0, trk_cnt = 0;
      double trk_span = 0.0;
      est.track_stats(trk_n, trk_span, trk_cnt);
if (rmaim_verbose()) std::printf(
          "[aim] fps=%2.0f det=%2.0f%% 到达→检完=%.1fms (预%.1f/推%.1f/后%.1f) 模型=%-6s σ=%.3f° "
          "cls=%d yaw=%.2f pitch=%.2f 开火=%d/%d trk=%d/%.1fs/%d nx=%d lb=%d/%.1fms/+%d\n",
          fps, frames ? 100.0 * detected / frames : 0.0, det_ms, detector.last_pre_ms(),
          detector.last_infer_ms(), detector.last_post_ms(), est.current_model(),
          est.current_sigma() * 57.2958, cls, yaw_cmd, pitch_cmd, fire_count, pub_count,
          trk_n, trk_span, trk_cnt, batch_n,
          lbar_plates.exchange(0) / std::max(1, lbar_frames.exchange(0)),
          lbar.last_ms(), lbar_added.exchange(0));
      g_gate.report();
      if (fps > 0 && fps < 35) {
        if (rmaim_verbose()) std::printf("[warn] 本进程帧率只有 %.0f Hz（仿真器渲染被降频？窗口被遮挡/桌面负载高）"
                    "—— 成绩会系统性偏低，请把仿真器窗口置于最前再测\n", fps);
      }
      {
        const long long wt = g_win_total.exchange(0), wo = g_win_open.exchange(0);
        if (wt > 0) if (rmaim_verbose()) std::printf("[win] 命中时刻门控 开=%lld/%lld (%.1f%%)\n", wo, wt,
                                100.0 * static_cast<double>(wo) / static_cast<double>(wt));
      }
      g_stage.report("");
      fire_count = 0;
      pub_count = 0;
      age_n = 0;
      sum_age = 0.0;
      (void)age_ms;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
   } catch (const std::exception& e) {
     static int ctl_err = 0;
     if (++ctl_err % 200 == 1) std::printf("[warn] 控制线程异常：%s（已忽略并继续）\n", e.what());
     std::this_thread::sleep_for(std::chrono::milliseconds(2));
   }
  }

  g_stop = true;
  if (det_thread.joinable()) det_thread.join();
  io.stop();
  if (rmaim_verbose()) std::printf("[aim] 退出：共接收 %llu 帧\n", static_cast<unsigned long long>(io.frames_received()));
  rclcpp::shutdown();
  return 0;
}
