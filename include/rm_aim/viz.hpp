// 可视化窗口：左=装甲板模型 + 实时落点，右=四板空间俯视图
//
// 只做"看"，不参与任何控制：所有量都由控制线程推进来，渲染也在控制线程里做（20 Hz）。
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <string>

namespace rmaim {

struct VizConfig {
  bool enable = true;
  std::string window = "aim_viz";
  std::string save_dir = "";   // 非空则周期性把画布存成 PNG（便于离线核对/写文档）
  int save_every = 40;         // 每 N 个渲染帧存一次
  int max_points = 160;        // 左侧保留的落点个数
};

// 一次开火节拍对应的"这一发的落点"（板坐标，归一化：±1 = 板边）
struct VizImpact {
  double dx = 0.0, dy = 0.0;
  bool inside = false;
};

// 每帧交给可视化的状态
struct VizFrame {
  bool valid = false;
  double range = 1.0;
  double half_w_deg = 1.0, half_h_deg = 0.8;   // 板半宽/半高（角度）
  double az_aim = 0.0, el_aim = 0.0;           // 当前瞄准角（度）
  bool tracking = false, fire = false, acquiring = false, ring = false;
  std::string model = "none";
  // ---- 四区块新增的数据 ----
  bool has_vehicle = false;                    // 只要"看到车"就把四板俯视图打开（不限于自转靶）
  double yaw_cmd = 0.0, yaw_fb = 0.0, pitch_cmd = 0.0, pitch_fb = 0.0;
  double obs_age_ms = 0.0, detect_ms = 0.0, infer_ms = 0.0, e2e_ms = 0.0;
  double fps = 0.0, det_pct = 0.0, fire_duty = 0.0;
  double sigma_deg = 0.0, range_med = 0.0;
  int track_hits = 0, batch_n = 0, bullets_per_s = 0;
  // 环参数（ring=true 时有意义）
  double az0 = 0.0, hw = 0.0, omega = 2.0;
  double ring_R = 0.212;                       // 环（车身外接圆）半径，m —— 机械常熟
  double plate_w_m = 0.135, plate_t_m = 0.025; // 板宽/板厚，m
  double phase_deg = 0.0;                      // "最正对那块板"的机械角 θ（度，θ=180 表示正对相机）
  double face[4] = {0, 0, 0, 0};               // 4 块板的正对度（来自逐板轨迹/环模型）
  double phase = 0.0;                          // 环相位（rad）
  double plate_deg[4] = {0, 0, 0, 0};          // 4 块板的方位（相对 az0，度）
};

class AimViz {
 public:
  explicit AimViz(const VizConfig& cfg) : cfg_(cfg) {}
  ~AimViz();
  void impact(const VizImpact& p) { impacts_.push_back(p); while (static_cast<int>(impacts_.size()) > cfg_.max_points) impacts_.pop_front(); }
  void update(const VizFrame& f) { frame_ = f; }
  void render();          // 画布 + 显示（20 Hz 由调用方控制）
  void push_trace(const VizFrame& f);   // 滚动轨迹（供左下"跟踪与时序"面板）
  const cv::Mat& canvas() const { return canvas_; }
  double hit_rate() const;

 private:
  VizConfig cfg_;
  VizFrame frame_;
  std::deque<VizImpact> impacts_;
  struct TraceS { double cmd, fb, err; };
  std::deque<TraceS> trace_;
  std::deque<double> rate_hist_;   // 最近若干秒的板内比例（右下滚动曲线）
  cv::Mat canvas_;
  int render_n_ = 0;
  bool inited_ = false;
};

}  // namespace rmaim
