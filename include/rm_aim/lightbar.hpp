// ① 灯条级观测：不经过神经网络，直接在图像上找「装甲板灯条」并配对成板。
//
// 为什么需要它？（见 docs/07 §23 的实测）
//   神经网络在小目标/半遮挡/侧视时置信度会掉到门槛以下（occ_trans 数据集中位数只有 0.44，11/40 帧低于 0.30），而灯条是**自发光的细长亮条**，几何特征极强：
//   灰度阈值 + 连通域 + 长宽比 就能稳定找到，且中心由两条灯条的几何中点给出，没有关键点抖动。于是它可以补上"模型置信度不足"的那部分帧。
//
// 本模块只输出「板级几何观测」，不改动任何模型/主选板逻辑，接入点是多板 batch。
#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace rmaim {

struct LightBarConfig {
  bool enable = false;   // ① 灯条级观测：实测中性，默认关（省 1.7 ms/帧）
  int gray_th = 120;        // 灰度阈值（灯条远亮于背景：实测背景均值 36，灯条均值 >200）
  int h_min = 18;           // 灯条高度范围（原图像素）
  int h_max = 220;
  int w_min = 2;            // 灯条宽度范围
  int w_max = 40;
  double ar_min = 2.5;      // 长宽比 h/w —— 灯条是细高条
  double ar_max = 25.0;
  double fill_min = 0.45;   // 连通域填充率（灯条是实心矩形，接近 1）
  double color_dom = 60.0;  // 颜色判据：min(B,G) − R > 它 → 我方（蓝/青）灯条
  double pair_h_tol = 0.35; // 配对：两灯条高度相对差上限
  double pair_dy_tol = 0.35;// 配对：两灯条中心 dy / 平均高度 上限
  double pair_gap_min = 0.6;// 配对：灯条间隙 / 平均高度 下限
  double pair_gap_max = 3.4;// 配对：灯条间隙 / 平均高度 上限
  double bar_h_ratio = 0.71;// 灯条高 / 板高（实测 28/39.5 ≈ 0.71，用于反推距离）
  double conf = 0.35;       // 合成置信度（低于正常模型检出，保证不抢主选板权重）
};

struct BarPlate {
  float cx = 0, cy = 0;   // 板中心（原图像素）
  float w = 0, h = 0;     // 板外缘宽、灯条高
  float score = 0;        // 配对质量（越小越好）
  int n_bars = 0;
};

class LightBarDetector {
 public:
  explicit LightBarDetector(const LightBarConfig& cfg) : cfg_(cfg) {}
  // 输入原图（BGR）。返回本帧配对成功的装甲板（我方颜色）。
  std::vector<BarPlate> detect(const cv::Mat& bgr);
  double last_ms() const { return last_ms_; }
  int last_bars() const { return last_bars_; }

 private:
  LightBarConfig cfg_;
  double last_ms_ = 0.0;
  int last_bars_ = 0;
};

}  // namespace rmaim
