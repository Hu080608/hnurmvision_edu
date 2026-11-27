#include "rm_aim/viz.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rmaim {
namespace {
const cv::Scalar kBg(34, 34, 40), kFg(232, 232, 232), kDim(150, 150, 162);
const cv::Scalar kGood(90, 235, 100), kBad(80, 80, 255), kPlate(200, 190, 120);
const cv::Scalar kAim(70, 230, 255), kRing(90, 160, 230);

void text(cv::Mat& img, const std::string& s, cv::Point p, cv::Scalar c = kFg, double sc = 0.42) {
  cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, sc, c, 1, cv::LINE_AA);
}
}  // namespace

AimViz::~AimViz() {
  if (cfg_.enable) {
    try { cv::destroyWindow(cfg_.window); } catch (...) {}
  }
}

double AimViz::hit_rate() const {
  if (impacts_.empty()) return 0.0;
  int n = 0;
  for (const auto& p : impacts_) if (p.inside) ++n;
  return 100.0 * static_cast<double>(n) / static_cast<double>(impacts_.size());
}


void AimViz::push_trace(const VizFrame& f) {
  TraceS t{f.yaw_cmd, f.yaw_fb, f.yaw_cmd - f.yaw_fb};
  trace_.push_back(t);
  while (trace_.size() > 260) trace_.pop_front();
  rate_hist_.push_back(f.fire_duty);
  while (rate_hist_.size() > 260) rate_hist_.pop_front();
}

namespace {
// 通用：面板标题 + 边框
cv::Rect panel_box(cv::Mat& img, int x, int y, int w, int h, const char* title) {
  cv::Rect r(x, y, w, h);
  cv::rectangle(img, r, kDim, 1);
  cv::rectangle(img, {x, y, w, 20}, cv::Scalar(50, 50, 58), -1);
  cv::putText(img, title, {x + 8, y + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.44, kFg, 1, cv::LINE_AA);
  return r;
}
}  // namespace

void AimViz::render() {
  const int W = 1020, H = 820;
  if (canvas_.empty()) canvas_ = cv::Mat(H, W, CV_8UC3, kBg);
  canvas_.setTo(kBg);
  ++render_n_;
  const VizFrame& f = frame_;
  char buf[256];

  // ======================= 区块 1：装甲板模型 + 落点 =======================
  {
    const int X = 8, Y = 8, PW = 496, PH = 382;
    panel_box(canvas_, X, Y, PW, PH, "1. ARMOR PLATE + IMPACTS  (x=az, y=el; +-1 = plate edge)");
    const int cx = X + PW / 2, cy = Y + 24 + (PH - 24) / 2 - 24;
    const int BW = 300, BH = 150;
    cv::Rect plate(cx - BW / 2, cy - BH / 2, BW, BH);
    cv::rectangle(canvas_, plate, cv::Scalar(52, 52, 58), -1);
    cv::rectangle(canvas_, plate, kPlate, 2);
    cv::rectangle(canvas_, cv::Rect(plate.x + 6, plate.y + 6, 16, plate.height - 12), kGood, -1);
    cv::rectangle(canvas_, cv::Rect(plate.x + plate.width - 22, plate.y + 6, 16, plate.height - 12), kGood, -1);
    cv::line(canvas_, {cx, plate.y}, {cx, plate.y + plate.height}, kDim, 1);
    cv::line(canvas_, {plate.x, cy}, {plate.x + plate.width, cy}, kDim, 1);
    text(canvas_, "+u", {plate.x + plate.width + 6, cy + 4}, kDim);
    text(canvas_, "+v", {cx + 4, plate.y - 6}, kDim);
    const int n = static_cast<int>(impacts_.size());
    int k = 0;
    for (const auto& p : impacts_) {
      const double a = 0.25 + 0.75 * (n > 1 ? static_cast<double>(k) / (n - 1) : 1.0);
      const cv::Scalar c = p.inside ? kGood : kBad;
      const int px = static_cast<int>(cx + p.dx * BW * 0.5);
      const int py = static_cast<int>(cy - p.dy * BH * 0.5);
      cv::circle(canvas_, {px, py}, 3, cv::Scalar(c[0] * a, c[1] * a, c[2] * a), -1, cv::LINE_AA);
      ++k;
    }
    double mx = 0, my = 0;
    for (const auto& p : impacts_) { mx += p.dx; my += p.dy; }
    if (n > 0) { mx /= n; my /= n; }
    std::snprintf(buf, sizeof(buf), "shots=%d  in-plate=%.1f%%  mean dx=%+.2f dy=%+.2f", n, hit_rate(), mx, my);
    text(canvas_, buf, {X + 8, Y + PH - 30}, kFg, 0.44);
    std::snprintf(buf, sizeof(buf), "half w/h = %.2f / %.2f deg   range=%.2f m", f.half_w_deg, f.half_h_deg, f.range);
    text(canvas_, buf, {X + 8, Y + PH - 12}, f.tracking ? kFg : kDim, 0.44);
  }

  // ======================= 区块 2：四板空间俯视图（常开） =======================
  {
    const int X = 516, Y = 8, PW = 496, PH = 382;
    panel_box(canvas_, X, Y, PW, PH, "2. TOP VIEW: body circle + 4 plates (always on)");
    const int rcx = X + PW / 2, rcy = Y + 24 + (PH - 24) / 2 + 8;
    const double ppm = 230.0;
    const double R = std::clamp(f.ring_R, 0.08, 0.60);
    const int Rp = static_cast<int>(std::clamp(R * ppm, 22.0, 140.0));
    const int t_px = std::max(3, static_cast<int>(f.plate_t_m * ppm));
    const int w_px = std::max(16, static_cast<int>(f.plate_w_m * ppm));
    cv::circle(canvas_, {rcx, rcy}, Rp, kRing, 2, cv::LINE_AA);
    cv::circle(canvas_, {rcx, rcy}, 3, kFg, -1);
    cv::line(canvas_, {X + 6, rcy}, {rcx - Rp, rcy}, kDim, 1, cv::LINE_AA);
    cv::drawMarker(canvas_, {rcx - Rp, rcy}, kAim, cv::MARKER_DIAMOND, 13, 2);
    text(canvas_, "cross", {rcx - Rp - 22, rcy - 12}, kAim, 0.36);
    for (int i = 0; i < 4; ++i) {
      const double th = (f.phase_deg + i * 90.0) * M_PI / 180.0;
      const double face = std::fabs(std::cos(th));
      const cv::Scalar c = face > 0.85 ? kGood : (face > 0.45 ? kAim : kDim);
      const double rr = R - 0.5 * f.plate_t_m;
      const int px = rcx + static_cast<int>(rr * ppm * std::cos(th));
      const int py = rcy - static_cast<int>(rr * ppm * std::sin(th));
      const double tx = -std::sin(th), ty = std::cos(th);
      const int hx = static_cast<int>(tx * w_px * 0.5), hy = static_cast<int>(-ty * w_px * 0.5);
      const int nx = static_cast<int>(std::cos(th) * t_px * 0.5);
      const int ny = static_cast<int>(-std::sin(th) * t_px * 0.5);
      cv::line(canvas_, {px - hx, py - hy}, {px + hx, py + hy}, c, 3, cv::LINE_AA);
      cv::line(canvas_, {px - hx + 2 * nx, py - hy + 2 * ny}, {px + hx + 2 * nx, py + hy + 2 * ny},
               c, 1, cv::LINE_AA);
      std::snprintf(buf, sizeof(buf), "%d", i);
      text(canvas_, buf, {px - 4 + nx, py - 6 + ny}, c, 0.40);
    }
    std::snprintf(buf, sizeof(buf), "%s R=%.2fm D=%.2fm hwG=%.1f hwO=%.1f",
                  f.ring ? "RING" : "STATIC/TRANS(assumed)",
                  R, f.range, std::atan2(R, std::max(0.5, f.range)) * 57.2958, f.hw * 57.2958);
    text(canvas_, buf, {X + 8, Y + PH - 30}, kFg, 0.42);
    std::snprintf(buf, sizeof(buf), "az0=%.2f  phase=%.0f deg  w=%.2f rad/s  aim%+.2f deg",
                  f.az0 * 57.2958, f.phase_deg, f.omega, (f.az_aim - f.az0) * 57.2958);
    text(canvas_, buf, {X + 8, Y + PH - 12}, kAim, 0.42);
  }

  // ======================= 区块 3：跟踪与时序（yaw 滚动曲线） =======================
  {
    const int X = 8, Y = 398, PW = 496, PH = 414;
    panel_box(canvas_, X, Y, PW, PH, "3. TRACKING & TIMING  (yaw cmd vs fb, ~13 s)");
    const cv::Rect plot(X + 56, Y + 30, PW - 76, 150);
    cv::rectangle(canvas_, plot, cv::Scalar(60, 60, 68), 1);
    double mean = 0.0;
    for (const auto& t : trace_) mean += t.cmd;
    if (!trace_.empty()) mean /= static_cast<double>(trace_.size());
    double lo = 1e9, hi = -1e9;
    for (const auto& t : trace_) { lo = std::min({lo, t.cmd, t.fb}); hi = std::max({hi, t.cmd, t.fb}); }
    if (trace_.empty()) { lo = -0.5; hi = 0.5; }
    const double half = std::max(0.15, std::max(std::fabs(hi - mean), std::fabs(lo - mean)) * 1.15);
    const double c = (hi + lo) * 0.5;
    auto ypx = [&](double v) {
      return plot.y + plot.height - static_cast<int>((v - (c - half)) / (2 * half) * plot.height);
    };
    cv::line(canvas_, {plot.x, ypx(c)}, {plot.x + plot.width, ypx(c)}, cv::Scalar(80, 80, 88), 1);
    const int n = static_cast<int>(trace_.size());
    for (int i = 1; i < n; ++i) {
      const int x0 = plot.x + static_cast<int>((i - 1) * 1.0 * plot.width / std::max(1, n - 1));
      const int x1 = plot.x + static_cast<int>(i * 1.0 * plot.width / std::max(1, n - 1));
      cv::line(canvas_, {x0, ypx(trace_[i - 1].cmd)}, {x1, ypx(trace_[i].cmd)}, kAim, 1, cv::LINE_AA);
      cv::line(canvas_, {x0, ypx(trace_[i - 1].fb)}, {x1, ypx(trace_[i].fb)}, kGood, 1, cv::LINE_AA);
    }
    std::snprintf(buf, sizeof(buf), "%+.2f", c + half);
    text(canvas_, buf, {X + 6, plot.y + 10}, kDim, 0.36);
    std::snprintf(buf, sizeof(buf), "%+.2f", c - half);
    text(canvas_, buf, {X + 6, plot.y + plot.height}, kDim, 0.36);
    std::snprintf(buf, sizeof(buf), "scale +-%.2f deg", half);
    text(canvas_, buf, {plot.x + 6, plot.y + 12}, kFg, 0.38);
    text(canvas_, "cmd(cyan) / fb(green)", {plot.x + 6, plot.y + plot.height - 8}, kDim, 0.38);
    const double err = trace_.empty() ? 0.0 : std::fabs(trace_.back().err);
    std::snprintf(buf, sizeof(buf), "|yaw_err|=%.2f deg   obs_age=%.0f ms", err, f.obs_age_ms);
    text(canvas_, buf, {X + 8, plot.y + plot.height + 26}, err < 0.6 ? kGood : kBad, 0.44);
    std::snprintf(buf, sizeof(buf), "fps=%.0f  det=%.0f%%  detect=%.1f ms (infer %.1f)  e2e=%.1f ms",
                  f.fps, f.det_pct, f.detect_ms, f.infer_ms, f.e2e_ms);
    text(canvas_, buf, {X + 8, plot.y + plot.height + 50}, kFg, 0.42);
    std::snprintf(buf, sizeof(buf), "model=%s  sigma=%.2f deg  trk_hits=%d  batch=%d",
                  f.model.c_str(), f.sigma_deg, f.track_hits, f.batch_n);
    text(canvas_, buf, {X + 8, plot.y + plot.height + 74}, kFg, 0.42);
    std::snprintf(buf, sizeof(buf), "fire_duty=%.0f%%  rate=%d shots/s  %s", f.fire_duty * 100.0,
                  f.bullets_per_s, f.acquiring ? "ACQUIRING" : "");
    text(canvas_, buf, {X + 8, plot.y + plot.height + 98}, f.fire ? kGood : kFg, 0.42);
    text(canvas_, "yaw jitter = settle shake after target switch; |yaw_err| gate = 0.6 deg",
         {X + 8, Y + PH - 14}, kDim, 0.40);
  }

  // ======================= 区块 4：落点统计（直方图 + 滚动板内比例） =======================
  {
    const int X = 516, Y = 398, PW = 496, PH = 414;
    panel_box(canvas_, X, Y, PW, PH, "4. IMPACT STATS  (dy<0 = shooting LOW)");
    // dy 直方图
    const cv::Rect hb(X + 30, Y + 32, PW - 60, 130);
    cv::rectangle(canvas_, hb, cv::Scalar(60, 60, 68), 1);
    const int BIN = 24;
    int hist[BIN] = {0};
    int cnt = 0;
    double sx = 0, sy = 0, sxx = 0, syy = 0;
    for (const auto& p : impacts_) {
      const double v = std::clamp(p.dy, -1.5, 1.5);
      const int b = std::clamp(static_cast<int>((v + 1.5) / 3.0 * BIN), 0, BIN - 1);
      ++hist[b]; ++cnt;
      sx += p.dx; sy += p.dy; sxx += p.dx * p.dx; syy += p.dy * p.dy;
    }
    int hmax = 1;
    for (int i = 0; i < BIN; ++i) hmax = std::max(hmax, hist[i]);
    const int bw = hb.width / BIN;
    for (int i = 0; i < BIN; ++i) {
      const int h = static_cast<int>(hist[i] * 1.0 * (hb.height - 6) / hmax);
      const double v = -1.5 + (i + 0.5) * 3.0 / BIN;
      const cv::Scalar c = std::fabs(v) < 1.0 ? kGood : kBad;
      cv::rectangle(canvas_, cv::Rect(hb.x + i * bw + 1, hb.y + hb.height - h - 1, bw - 2, h), c, -1);
    }
    auto vx = [&](double v) { return hb.x + static_cast<int>((v + 1.5) / 3.0 * hb.width); };
    cv::line(canvas_, {vx(0), hb.y}, {vx(0), hb.y + hb.height}, kFg, 1);
    cv::line(canvas_, {vx(-1), hb.y}, {vx(-1), hb.y + hb.height}, kDim, 1);
    cv::line(canvas_, {vx(1), hb.y}, {vx(1), hb.y + hb.height}, kDim, 1);
    text(canvas_, "-1", {vx(-1) - 8, hb.y + hb.height + 14}, kDim, 0.36);
    text(canvas_, "0", {vx(0) - 4, hb.y + hb.height + 14}, kFg, 0.36);
    text(canvas_, "+1", {vx(1) - 6, hb.y + hb.height + 14}, kDim, 0.36);
    if (cnt > 1) {
      const double mx = sx / cnt, my = sy / cnt;
      const double sdx = std::sqrt(std::max(0.0, sxx / cnt - mx * mx));
      const double sdy = std::sqrt(std::max(0.0, syy / cnt - my * my));
      cv::line(canvas_, {vx(my), hb.y + 4}, {vx(my), hb.y + hb.height - 4}, kAim, 2);
      std::snprintf(buf, sizeof(buf), "dy: mean=%+.2f sd=%.2f   dx: mean=%+.2f sd=%.2f   n=%d", my, sdy, mx, sdx, cnt);
      text(canvas_, buf, {X + 8, hb.y + hb.height + 34}, kFg, 0.44);
      std::snprintf(buf, sizeof(buf), "bias: %s (%.2f plate-height)", my < -0.15 ? "LOW" : (my > 0.15 ? "HIGH" : "centered"), std::fabs(my));
      text(canvas_, buf, {X + 8, hb.y + hb.height + 58}, std::fabs(my) < 0.15 ? kGood : kBad, 0.46);
    }
    // 滚动"板内比例"
    const cv::Rect rb(X + 30, hb.y + hb.height + 76, PW - 60, 88);
    cv::rectangle(canvas_, rb, cv::Scalar(60, 60, 68), 1);
    text(canvas_, "rolling in-plate ratio (per 100 ms shot slot)", {rb.x + 6, rb.y + 14}, kDim, 0.38);
    if (impacts_.size() > 2) {
      std::vector<double> curve;
      double acc = 0;
      int m = 0, k = 0;
      for (const auto& p : impacts_) {
        acc += p.inside ? 1 : 0; ++m;
        if (++k % 5 == 0) { curve.push_back(acc / m); }
      }
      const int cn = static_cast<int>(curve.size());
      for (int i = 1; i < cn; ++i) {
        const int x0 = rb.x + static_cast<int>((i - 1) * 1.0 * rb.width / std::max(1, cn - 1));
        const int x1 = rb.x + static_cast<int>(i * 1.0 * rb.width / std::max(1, cn - 1));
        const int y0 = rb.y + rb.height - 4 - static_cast<int>(curve[i - 1] * (rb.height - 20));
        const int y1 = rb.y + rb.height - 4 - static_cast<int>(curve[i] * (rb.height - 20));
        cv::line(canvas_, {x0, y0}, {x1, y1}, kAim, 1, cv::LINE_AA);
      }
      std::snprintf(buf, sizeof(buf), "now %.0f%%", curve.empty() ? 0.0 : curve.back() * 100.0);
      text(canvas_, buf, {rb.x + rb.width - 70, rb.y + 14}, kFg, 0.38);
    }
  }

  if (cfg_.enable) {
    if (!inited_) {
      cv::namedWindow(cfg_.window, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
      cv::resizeWindow(cfg_.window, W, H);
      inited_ = true;
    }
    cv::imshow(cfg_.window, canvas_);
    cv::waitKey(1);
  }
  if (!cfg_.save_dir.empty() && cfg_.save_every > 0 && render_n_ % cfg_.save_every == 0) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/viz_%05d.png", cfg_.save_dir.c_str(), render_n_);
    cv::imwrite(path, canvas_);
  }
}

}  // namespace rmaim
