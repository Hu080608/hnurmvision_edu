// 检测器：YOLO-pose 装甲板检测（ONNX Runtime，CUDA 优先）
// 输出布局（本模型实测确认）：[x1, y1, x2, y2, conf, cls, k1x,k1y, k2x,k2y, k3x,k3y, k4x,k4y]
#pragma once

#include "config.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace Ort {
class Session;
class Env;
}  // namespace Ort

namespace rmaim {

struct Detection {
  std::array<cv::Point2f, 4> corners{};  // 上左、上右、下右、下左
  cv::Rect2f box{};
  float conf = 0.f;
  int cls = -1;
  bool large = false;

  float width_px() const { return box.width; }
  cv::Point2f center() const {
    return {(corners[0].x + corners[1].x + corners[2].x + corners[3].x) * 0.25f,
            (corners[0].y + corners[1].y + corners[2].y + corners[3].y) * 0.25f};
  }
  bool valid() const { return conf > 0.f && box.width > 0.f && (cls >= 0 || !large); }
};

class Detector {
 public:
  explicit Detector(const DetectorConfig& cfg);
  ~Detector();

  std::vector<Detection> detect(const cv::Mat& bgr);
  double last_infer_ms() const { return last_ms_; }
  double last_pre_ms() const { return pre_ms_; }
  double last_post_ms() const { return post_ms_; }
  const std::string& provider() const { return provider_; }

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
  DetectorConfig cfg_;
  double last_ms_ = 0, pre_ms_ = 0, post_ms_ = 0;
  std::string provider_;
};

// 把 4 个任意顺序的关键点整理成「上左、上右、下右、下左」
std::array<cv::Point2f, 4> canonicalize(const std::array<cv::Point2f, 4>& in);
// 大板判定：宽高比 / 面积 / 类别
bool is_large_class(int cls);

}  // namespace rmaim
