#include "rm_aim/lightbar.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace rmaim {
namespace {
double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
struct Bar {
  int x = 0, y = 0, w = 0, h = 0;
  double cx = 0, cy = 0;
};
}  // namespace

std::vector<BarPlate> LightBarDetector::detect(const cv::Mat& bgr) {
  const double t0 = now_ms();
  last_bars_ = 0;
  std::vector<BarPlate> out;
  if (!cfg_.enable || bgr.empty() || bgr.channels() != 3) {
    last_ms_ = now_ms() - t0;
    return out;
  }

  // 1) 灰度阈值：灯条是自发光，远亮于背景
  cv::Mat gray, mask;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, mask, cfg_.gray_th, 255, cv::THRESH_BINARY);
  // 轻微闭运算，补上灯条内部的暗缝（不能开运算，否则细灯条会消失）
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  // 2) 连通域 + 形状筛选
  cv::Mat labels, stats, cents;
  const int n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);
  std::vector<Bar> bars;
  bars.reserve(32);
  for (int i = 1; i < n; ++i) {
    const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
    const int y = stats.at<int>(i, cv::CC_STAT_TOP);
    const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
    const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (h < cfg_.h_min || h > cfg_.h_max) continue;
    if (w < cfg_.w_min || w > cfg_.w_max) continue;
    const double ar = static_cast<double>(h) / std::max(1, w);
    if (ar < cfg_.ar_min || ar > cfg_.ar_max) continue;
    if (static_cast<double>(area) / static_cast<double>(std::max(1, w * h)) < cfg_.fill_min) continue;
    // 3) 颜色判据：只保留我方（蓝/青）灯条 —— 红队/环境灯柱在这一步被剔除
    const cv::Scalar m = cv::mean(bgr(cv::Rect(x, y, w, h)));
    if (std::min(m[0], m[1]) - m[2] < cfg_.color_dom) continue;   // (B,G,R)
    Bar b;
    b.x = x; b.y = y; b.w = w; b.h = h;
    b.cx = x + 0.5 * w;
    b.cy = y + 0.5 * h;
    bars.push_back(b);
  }
  last_bars_ = static_cast<int>(bars.size());
  if (bars.size() < 2) { last_ms_ = now_ms() - t0; return out; }

  // 4) 两两配对：同高、同水平线、间隙在灯条高的 0.6~3.4 倍之间
  struct Cand { int i, j; double score; };
  std::vector<Cand> cands;
  for (size_t i = 0; i < bars.size(); ++i) {
    for (size_t j = i + 1; j < bars.size(); ++j) {
      const Bar& a = bars[i];
      const Bar& b = bars[j];
      const double hh = 0.5 * (a.h + b.h);
      const double hr = std::fabs(static_cast<double>(a.h - b.h)) / std::max(1.0, hh);
      if (hr > cfg_.pair_h_tol) continue;
      const double dy = std::fabs(a.cy - b.cy);
      if (dy > cfg_.pair_dy_tol * hh) continue;
      const double gap = std::fabs(a.cx - b.cx) - 0.5 * (a.w + b.w);
      if (gap < cfg_.pair_gap_min * hh || gap > cfg_.pair_gap_max * hh) continue;
      const double gh = gap / std::max(1.0, hh);
      // 质量分：高度差 + 水平偏差 + 间隙偏离实测中值(≈2.1) 的惩罚
      const double score = hr + dy / std::max(1.0, hh) + 0.5 * std::fabs(gh - 2.1);
      cands.push_back({static_cast<int>(i), static_cast<int>(j), score});
    }
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score < b.score; });

  std::vector<char> used(bars.size(), 0);
  for (const auto& c : cands) {
    if (used[c.i] || used[c.j]) continue;
    used[c.i] = used[c.j] = 1;
    const Bar& a = bars[c.i];
    const Bar& b = bars[c.j];
    BarPlate p;
    p.cx = static_cast<float>(0.5 * (a.cx + b.cx));
    p.cy = static_cast<float>(0.5 * (a.cy + b.cy));
    p.h = static_cast<float>(0.5 * (a.h + b.h));
    p.w = static_cast<float>(std::fabs(a.cx - b.cx) + 0.5 * (a.w + b.w));
    p.score = static_cast<float>(c.score);
    p.n_bars = 2;
    out.push_back(p);
    if (out.size() >= 6) break;
  }

  last_ms_ = now_ms() - t0;
  return out;
}

}  // namespace rmaim
