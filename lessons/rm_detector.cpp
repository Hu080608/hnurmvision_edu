#include "rm_aim/detector.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace rmaim {
// 静默开关（默认静默，保持仓库原有控制台输出；RMAIM_VERBOSE=1 可打开本栈日志）
bool rmaim_verbose();
namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline float cross(const cv::Point2f& o, const cv::Point2f& a, const cv::Point2f& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

}  // namespace

std::array<cv::Point2f, 4> canonicalize(const std::array<cv::Point2f, 4>& in) {
  std::array<cv::Point2f, 4> p = in;
  cv::Point2f c(0, 0);
  for (const auto& q : p) c += q;
  c *= 0.25f;
  // 1) 按极角排序，得到确定的环绕顺序
  std::sort(p.begin(), p.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
    return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
  });
  // 2) 旋转到"最小 x+y"作为上左
  int start = 0;
  float best = 1e18f;
  for (int i = 0; i < 4; ++i) {
    const float s = p[i].x + p[i].y;
    if (s < best) { best = s; start = i; }
  }
  std::array<cv::Point2f, 4> out;
  for (int i = 0; i < 4; ++i) out[i] = p[(start + i) % 4];
  // 3) 保证顺序是「上左→上右→下右→下左」。
  //    图像坐标 y 向下时，这个顺序的叉积 cross(P0,P1,P2) 为正；为负说明反了。
  if (cross(out[0], out[1], out[2]) < 0.f) std::swap(out[1], out[3]);
  return out;
}

bool is_large_class(int cls) {
  // 与训练标签一致：0=1号英雄大板，6=基地大板，其余为小板（具体以模型标签为准）
  return cls == 0 || cls == 6;
}

struct Detector::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rm_aim"};
  std::unique_ptr<Ort::Session> session;
  std::string input_name;
  std::vector<std::string> output_names;
  std::vector<const char*> in_ptr, out_ptr;
  int input_h = 768, input_w = 768;
};

Detector::Detector(const DetectorConfig& cfg) : impl_(std::make_unique<Impl>()), cfg_(cfg) {
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetIntraOpNumThreads(1);
  bool cuda_ok = false;
  if (cfg.use_cuda) {
    try {
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = 0;
      opts.AppendExecutionProvider_CUDA(cuda_opts);
      cuda_ok = true;
    } catch (const std::exception& e) {
      if (rmaim_verbose()) std::printf("[det] CUDA provider 不可用（%s），回退 CPU\n", e.what());
    }
  }
  try {
    impl_->session = std::make_unique<Ort::Session>(impl_->env, cfg.model.c_str(), opts);
    provider_ = cuda_ok ? "CUDA" : "CPU";
  } catch (const std::exception& e) {
    if (!cuda_ok) throw;
    if (rmaim_verbose()) std::printf("[det] CUDA 会话创建失败（%s），回退 CPU\n", e.what());
    Ort::SessionOptions cpu;
    cpu.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, cfg.model.c_str(), cpu);
    provider_ = "CPU";
  }

  Ort::AllocatorWithDefaultOptions alloc;
  auto in_name = impl_->session->GetInputNameAllocated(0, alloc);
  impl_->input_name = in_name.get();
  const auto shape = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() == 4) {
    impl_->input_h = static_cast<int>(shape[2] > 0 ? shape[2] : cfg.input_size);
    impl_->input_w = static_cast<int>(shape[3] > 0 ? shape[3] : cfg.input_size);
  }
  const size_t n_out = impl_->session->GetOutputCount();
  for (size_t i = 0; i < n_out; ++i) {
    auto nm = impl_->session->GetOutputNameAllocated(i, alloc);
    impl_->output_names.emplace_back(nm.get());
  }
  for (auto& s : impl_->output_names) impl_->out_ptr.push_back(s.c_str());
  impl_->in_ptr.push_back(impl_->input_name.c_str());
  if (rmaim_verbose()) std::printf("[det] 模型=%s 输入=%dx%d 输出数=%zu 后端=%s\n", cfg.model.c_str(), impl_->input_w,
              impl_->input_h, n_out, provider_.c_str());
}

Detector::~Detector() = default;

std::vector<Detection> Detector::detect(const cv::Mat& bgr) {
  std::vector<Detection> out;
  if (bgr.empty()) return out;
  const double t0 = now_ms();

  // ---- letterbox（保持宽高比 + 灰度填充，与训练一致）----
  const int S = impl_->input_w;
  const double r = std::min(static_cast<double>(S) / bgr.rows, static_cast<double>(S) / bgr.cols);
  const int nh = static_cast<int>(std::lround(bgr.rows * r));
  const int nw = static_cast<int>(std::lround(bgr.cols * r));
  const int dx = (S - nw) / 2, dy = (S - nh) / 2;
  cv::Mat canvas(S, S, CV_8UC3, cv::Scalar(cfg_.pad_gray, cfg_.pad_gray, cfg_.pad_gray));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

  // ---- 转 float32 CHW + RGB ----
  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
  std::vector<float> blob(3 * S * S);
  std::vector<cv::Mat> chans(3);
  for (int c = 0; c < 3; ++c) chans[c] = cv::Mat(S, S, CV_32FC1, blob.data() + c * S * S);
  cv::split(rgb, chans);

  const double t1 = now_ms();
  const std::array<int64_t, 4> shape{1, 3, S, S};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(mem, blob.data(), blob.size(), shape.data(),
                                                     shape.size());
  auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, impl_->in_ptr.data(), &input, 1,
                                     impl_->out_ptr.data(), impl_->out_ptr.size());
  const double t2 = now_ms();
  if (outputs.empty()) return out;

  // ---- 解码 ----
  const auto info = outputs[0].GetTensorTypeAndShapeInfo();
  const auto oshape = info.GetShape();
  const float* data = outputs[0].GetTensorData<float>();
  int rows = 0, cols = 0;
  if (oshape.size() == 3) {
    rows = static_cast<int>(oshape[1]);
    cols = static_cast<int>(oshape[2]);
  } else if (oshape.size() == 2) {
    rows = static_cast<int>(oshape[0]);
    cols = static_cast<int>(oshape[1]);
  } else {
    return out;
  }
  if (cols < 14) return out;   // 不是本模型的输出布局

  const float inv = static_cast<float>(1.0 / r);
  for (int i = 0; i < rows; ++i) {
    const float* row = data + static_cast<size_t>(i) * cols;
    const float conf = row[4];
    if (conf < static_cast<float>(cfg_.conf)) continue;
    const int cls = static_cast<int>(std::lround(row[5]));
    Detection d;
    d.conf = conf;
    d.cls = cls;
    d.large = is_large_class(cls);
    if (conf < static_cast<float>(d.large ? cfg_.conf : cfg_.conf_small_digit)) {
      // 小板/大板各自门槛；这里只对"带数字的小板"再提一档
      if (!d.large && conf < static_cast<float>(cfg_.conf)) continue;
    }
    // 框：canvas → 原图
    const float x1 = (row[0] - dx) * inv;
    const float y1 = (row[1] - dy) * inv;
    const float x2 = (row[2] - dx) * inv;
    const float y2 = (row[3] - dy) * inv;
    d.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    if (d.box.width < static_cast<float>(cfg_.min_box_w_px)) continue;
    std::array<cv::Point2f, 4> kp;
    for (int k = 0; k < 4; ++k) {
      kp[k] = {(row[6 + 2 * k] - dx) * inv, (row[7 + 2 * k] - dy) * inv};
    }
    d.corners = canonicalize(kp);
    out.push_back(d);
  }

  // 去重：同一块板被重复检出时保留置信度高的
  std::sort(out.begin(), out.end(), [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
  std::vector<Detection> kept;
  for (const auto& d : out) {
    bool dup = false;
    for (const auto& k : kept) {
      const float dxc = d.center().x - k.center().x, dyc = d.center().y - k.center().y;
      if (dxc * dxc + dyc * dyc < 30.f * 30.f) { dup = true; break; }
    }
    if (!dup) kept.push_back(d);
    if (kept.size() >= 8) break;
  }

  const double t3 = now_ms();
  pre_ms_ = t1 - t0;
  last_ms_ = t2 - t1;
  post_ms_ = t3 - t2;
  return kept;
}

}  // namespace rmaim
