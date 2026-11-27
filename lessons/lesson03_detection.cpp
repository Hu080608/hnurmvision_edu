// ============================================================================
// 第三讲：装甲板检测 + 数字识别（YOLO26-Pose + ONNX Runtime）
// ============================================================================
#include "lessons.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <optional>

namespace training {

std::vector<Armor> detect_full(const cv::Mat& image);   // 第三讲正文实现（定义在本文件后部）

bool g_suppress_window = false;

// 第三讲：装甲板检测
// 输入：不可修改的 CV_8UC3 BGR 图像。
// 输出：Armor 列表；角点顺序为左上、右上、右下、左下，单位为像素。
// 图像坐标原点在左上；无法识别编号时 class_id=-1；没有装甲板时返回空列表。
namespace {

// 可调参数
constexpr float kConfThreshold = 0.55f;   // 全局阈值（逐帧判决，最终门槛）
// 防误检的最后一道物理防线
constexpr bool kEnablePerClassThreshold = true;    // 启用逐类阈值
constexpr bool kEnableLargeGeometryFilter = true;  // 间距比 + 面积校验（大板类）

// 逐类阈值（基地与前哨站分开设）：
constexpr float kConfBase    = 0.80f;   // 基地 BS(6)/RS(13)
constexpr float kConfOutpost = 0.70f;   // 前哨站 BO(5)/RO(12)

// “看到数字”的物理防线：有数字的类（1~5）如果灯条长度 < kMinDigitPlatePx，说明数字在图上只有几个像素，根本认不出 → 直接丢弃（不参与自瞄）。
constexpr float kMinDigitPlatePx = 12.0f;

// 小板
constexpr float kSmallPlatePx = 20.0f;     // 灯条长度低于此值算"看不见数字的小板"
constexpr float kSmallPlateConf = 0.90f;   // 小板需要的置信度（普通板 0.55）

// 裁剪放大后再识别一次（更严格，会误杀）（关）
constexpr bool  kEnableZoomVerify = false;
constexpr float kZoomVerifyPlatePx = 24.0f;
constexpr float kZoomVerifyMinConf = 0.35f;
constexpr int   kZoomVerifyMaxPerFrame = 3;
constexpr int   kZoomReclassMaxPerFrame = 2;   // 一帧最多做几次"放大重识别"（每次多跑一遍推理）


// 二级图案分类器（基地 S / 前哨站 O）：板面透视矫正到 96x48 再判图案
constexpr float kDedupIoU = 0.40f;
// 大板类（基地/前哨站）的最小面积，200（= plausible_large_geometry 里的面积下限，两者对齐）
constexpr float kMinLargePlateArea = 200.f;
// 两块物理上独立的装甲板不可能共用角点/边：若两个检出共享角点（距离<该值）或近似共边，则必有一个是假框 → 丢弃置信度低的那个。
constexpr float kSharedCornerPx = 8.f;

// ── 时域去闪：阈值迟滞（不动位置、不外推）──
constexpr bool  kEnableTrack = true;
constexpr int   kTrackMaxMissed = 0;      // 0=不保持（推荐）；1~2 可再少丢几帧，代价是快速甩镜头时会看到 1~2 帧"停在旧位置"的框
constexpr float kTrackIouMatch = 0.20f;   // 跨帧关联的 IoU 门槛
constexpr float kExitConfRatio = 0.60f;   // 在显示的目标：分数≥门槛×0.6 就继续显示
constexpr int   kClassHoldFrames = 2;     // 换标签需连续 2 帧一致，防抖

// ── 时间证据（借用前几帧判断当前这一帧）──
// 对边缘分数的目标要求："最近 kSupportFrames 帧里至少出现 kSupportHits 次"：
//   · 分数 ≥ kShowConf 的新目标 → 立刻显示，一帧都不等（不引入任何滞后）；
//   · 边缘分数的新目标 → 攒够时间证据再显示（最多晚 1 帧 = 33 ms）；
//   · 已经在显示的目标 → 只要本帧还关联得上就继续显示（分数掉一点也不闪，因为位置仍是本帧实测值，不是旧位置）。
constexpr float kShowConf = 0.75f;        // 高置信：直接显示
constexpr int   kSupportFrames = 3;       // 时间窗：最近 3 帧
constexpr int   kSupportHits = 2;         // 其中 ≥2 帧在同一位置被检出才认

// 但那道"一致性"只该用来审”边缘分数“的目标：真装甲板常常一出现就是 0.95+，不能因为它在头两帧还没攒够历史就把它藏起来（那就是"框跟不上"）。
constexpr float kTrustConf = 0.90f;       // ≥ 这个分数：不再审类别一致性，立刻显示

// ── 低照度归一化（CLAHE）───
// 仿真器夜战场景实测整帧平均灰度只有 35（真实照片约 110），模型在这种输入上
// 数字板漏检、还把车身上的暗块认成大板。在缩放后的图上做一次 CLAHE（限制对比度适应直方图均衡），把局部对比度拉回训练分布。
constexpr bool  kEnableLowLightNorm = true;
constexpr float kClaheClip = 5.0f;        // 对比度限制，越大越猛
constexpr int   kClaheTile = 8;           // 分块数（8×8）
// 兜底：一帧什么都没检出时，用原图跑一遍
constexpr bool  kEnableRawRetry = true;

constexpr int   kPatternW = 96;
constexpr int   kPatternH = 48;
constexpr float kPatternOverrideConf = 0.95f;
// 三分类图案模型（D=数字板 / O=前哨站 / S=基地）的"否决"阈值：
// 大板类（BO/BS/RO/RS）的候选，若图案被判成 D（数字板）且置信度≥该值 → 直接丢弃。
constexpr float kPatternVetoConf = 0.70f;
// "所有装甲板中间必有图案/数字，没有的一定不是装甲板"——把它量化成板面内部的高频纹理能量。
constexpr float kMinPatternEnergy = 3.5f;   // 板面内部 |Laplacian| 均值下限
constexpr bool  kEnablePatternEnergyGate = false;  // 仅仿真域建议 true
// 开关：只在"仿真器"开启。否则会误检
constexpr bool kEnablePatternOverride = true;
const char* kDefaultPatternPath = "assets/pattern_dos.onnx";

inline float class_conf_threshold(int cls) {
  if (!kEnablePerClassThreshold) return kConfThreshold;
  if (cls == 6 || cls == 13) return kConfBase;      // 基地 BS/RS
  if (cls == 5 || cls == 12) return kConfOutpost;   // 前哨站 BO/RO
  return kConfThreshold;                            // 1~5 号
}
constexpr int kMaxDet = 300;              // 端到端模型的固定槽位数（输出第一维）
constexpr int kNumClasses = 14;           // 14 类
constexpr int kNumKeypoints = 4;          // 4 个角点
constexpr int kChannels = 4 + 1 + 1 + kNumKeypoints * 2;  // 框4 + 分数1 + 类别1 + 角点8 = 14

// 模型路径：相对仓库根目录；可用环境变量 ARMOR_ONNX 覆盖
const char* kDefaultModelPath = "assets/armor_pose.onnx";

// 类别名（仅用于绘制）
const std::array<const char*, kNumClasses> kClassNames = {
    "B1", "B2", "B3", "B4", "B5", "BO", "BS",
    "R1", "R2", "R3", "R4", "R5", "RO", "RS"};

// ── letterbox 结果 ──
struct LetterboxResult {
  std::vector<float> blob;   // 长度 3*H*W 的连续内存（NCHW，RGB，0~1）
  float scale = 1.0f;        // 等比缩放系数
  float dx = 0.0f;           // 左侧填充像素
  float dy = 0.0f;           // 顶部填充像素
};

// 低照度归一化：只拉 L（亮度）通道的局部对比度，不改色相与饱和度，
// 所以蓝/红阵营的判色不受影响。放在缩放之后做：成本只与 768×768 有关（≈2ms），也不会被 letterbox 的 114 灰填充干扰直方图。
void lowlight_norm(cv::Mat& bgr, bool enable) {
  if (!enable || bgr.empty()) return;
  static cv::Ptr<cv::CLAHE> clahe =
      cv::createCLAHE(kClaheClip, cv::Size(kClaheTile, kClaheTile));
  cv::Mat lab;
  cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> ch;
  cv::split(lab, ch);
  clahe->apply(ch[0], ch[0]);
  cv::merge(ch, lab);
  cv::cvtColor(lab, bgr, cv::COLOR_Lab2BGR);
}

// letterbox：等比缩放到 size×size 内，余下用 114 灰填充。
// 为什么不用 cv::resize 直接拉伸？会改变长宽比，破坏大小板判定与角点几何。
LetterboxResult letterbox(const cv::Mat& bgr, int size, bool light_norm) {
  LetterboxResult out;
  const int w = bgr.cols, h = bgr.rows;
  out.scale = std::min(size / static_cast<float>(w), size / static_cast<float>(h));
  const int new_w = static_cast<int>(std::lround(w * out.scale));
  const int new_h = static_cast<int>(std::lround(h * out.scale));
  out.dx = (size - new_w) * 0.5f;
  out.dy = (size - new_h) * 0.5f;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  lowlight_norm(resized, light_norm);     // 低照度归一化（仿真夜战必须开）

  cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(static_cast<int>(out.dx), static_cast<int>(out.dy), new_w, new_h)));

  // BGR→RGB → float32(0~1) → split 成 3 个平面 → 拼成 NCHW
  cv::Mat rgb, f32;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> planes(3);
  cv::split(f32, planes);                       // 每个 (size, size) 单通道
  out.blob.assign(static_cast<size_t>(3) * size * size, 0.f);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(out.blob.data() + static_cast<size_t>(c) * size * size, planes[c].ptr<float>(0),
                sizeof(float) * size * size);
  }
  return out;
}

// 把 4 个角点规范化成仓库约定顺序：左上、右上、右下、左下
std::array<cv::Point2f, 4> canonicalize_corners(const std::array<cv::Point2f, 4>& in) {
  std::vector<cv::Point2f> pts(in.begin(), in.end());

  // ① 按绕质心的极角排序得到环序
  cv::Point2f c(0.f, 0.f);
  for (const auto& p : pts) c += p;
  c *= 0.25f;
  std::sort(pts.begin(), pts.end(), [&c](const cv::Point2f& a, const cv::Point2f& b) {
    return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
  });

  // ② 统一绕向：屏幕坐标（y 向下）下 TL→TR→BR→BL 的鞋带面积为正
  float area2 = 0.f;
  for (int i = 0; i < 4; ++i) {
    const auto& p = pts[i];
    const auto& q = pts[(i + 1) % 4];
    area2 += p.x * q.y - q.x * p.y;
  }
  if (area2 < 0.f) std::reverse(pts.begin(), pts.end());

  // ③ 旋转起点：第一个点取"最靠近左上角"（x+y 最小）
  int best = 0;
  float best_val = 1e18f;
  for (int i = 0; i < 4; ++i) {
    const float v = pts[i].x + pts[i].y;
    if (v < best_val) { best_val = v; best = i; }
  }
  std::rotate(pts.begin(), pts.begin() + best, pts.end());

  // ④ 保证第 2 个点在第 4 个点右侧（右上、左下的相对关系）
  if (pts[1].x < pts[3].x) std::swap(pts[1], pts[3]);

  return {pts[0], pts[1], pts[2], pts[3]};
}

// 大小装甲板：由类别直接决定
//   RM 规则：基地（BS/RS）与前哨站（BO/RO）用大装甲板 225×125mm；1~5 号用小装甲板 135×125mm。

inline bool is_large_class(int class_id) {
  // 类别顺序：B1..B5=0..4, BO=5, BS=6, R1..R5=7..11, RO=12, RS=13
  return class_id == 5 || class_id == 6 || class_id == 12 || class_id == 13;
}

// 两个四边形的外接矩形 IoU（用于类别无关去重）
inline float quad_iou(const std::array<cv::Point2f, 4>& a, const std::array<cv::Point2f, 4>& b) {
  auto rect = [](const std::array<cv::Point2f, 4>& q) {
    float x0 = q[0].x, x1 = q[0].x, y0 = q[0].y, y1 = q[0].y;
    for (int i = 1; i < 4; ++i) {
      x0 = std::min(x0, q[i].x); x1 = std::max(x1, q[i].x);
      y0 = std::min(y0, q[i].y); y1 = std::max(y1, q[i].y);
    }
    return cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
  };
  const cv::Rect2f ra = rect(a), rb = rect(b);
  const float inter = (ra & rb).area();
  const float uni = ra.area() + rb.area() - inter;
  return uni > 1e-6f ? inter / uni : 0.f;
}

// 间距比 = 两灯条水平间距 ÷ 灯条长度 = 板宽 ÷ 板高。
inline bool plausible_large_geometry(const std::array<cv::Point2f, 4>& q) {
  // 两条竖直边的长度 = 灯条长度；上下边 = 两灯条间距
  const float bar = 0.5f * (cv::norm(q[0] - q[3]) + cv::norm(q[2] - q[1]));
  const float gap = 0.5f * (cv::norm(q[1] - q[0]) + cv::norm(q[3] - q[2]));
  if (bar < 1e-3f || gap < 1e-3f) return false;
  const float spacing = std::max(gap, bar) / std::min(gap, bar);
  const float area = gap * bar;
  return spacing >= 1.45f && spacing <= 4.80f && area >= 200.f;
}

// ── ONNX Runtime 会话封装（懒加载单例）───
class Detector {
 public:
  static Detector& instance() {
    static Detector inst;   // C++11 局部静态初始化线程安全
    return inst;
  }

  bool ok() const { return ok_; }
  int input_h() const { return input_h_; }
  int input_w() const { return input_w_; }
  bool is_e2e() const { return is_e2e_; }   // true: [1,300,14] / false: [1,26,8400]

  std::vector<float> infer(const LetterboxResult& lb) {
    const std::array<int64_t, 4> shape{1, 3, input_h_, input_w_};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(lb.blob.data()), lb.blob.size(), shape.data(), shape.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> result(count);
    std::memcpy(result.data(), outputs[0].GetTensorData<float>(), count * sizeof(float));
    return result;
  }

 private:
  Detector() {
    try {
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "armor_detector");

      std::string path = kDefaultModelPath;
      if (const char* env_path = std::getenv("ARMOR_ONNX")) path = env_path;

      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(4);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      // 先试 CUDA，失败回退 CPU（CPU 版 onnxruntime 没有 CUDA provider）
      bool cuda_added = false;
      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        cuda_added = true;
      } catch (const std::exception&) {
        cuda_added = false;
      }

      try {
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);
        provider_ = cuda_added ? "CUDA" : "CPU";
      } catch (const std::exception& e) {
        if (!cuda_added) throw;
        Ort::SessionOptions cpu_opts;
        cpu_opts.SetIntraOpNumThreads(4);
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), cpu_opts);
        provider_ = "CPU(回退)";
        std::printf("[armor] CUDA 建会话失败，已回退 CPU：%s\n", e.what());
      }

      Ort::AllocatorWithDefaultOptions alloc;
      input_name_ = session_->GetInputNameAllocated(0, alloc).get();
      output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

      const auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      input_h_ = static_cast<int>(in_shape[2]);
      input_w_ = static_cast<int>(in_shape[3]);

      const auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      is_e2e_ = (out_shape.size() == 3 && out_shape[2] == kChannels);

      ok_ = true;
      std::printf("[armor] 模型=%s 输入=%dx%d 输出=[1,%lld,%lld] 端到端=%s 后端=%s\n", path.c_str(),
                  input_w_, input_h_, static_cast<long long>(out_shape[1]),
                  static_cast<long long>(out_shape[2]), is_e2e_ ? "是" : "否", provider_.c_str());
    } catch (const std::exception& e) {
      ok_ = false;
      std::printf("[armor] 模型加载失败：%s\n", e.what());
    }
  }

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_, output_name_, provider_;
  int input_h_ = 640, input_w_ = 640;
  bool is_e2e_ = true;
  bool ok_ = false;
};

// 调试绘制（OpenCV putText 不支持中文，标签用英文类别名）
void draw_debug(cv::Mat& canvas, const std::vector<Armor>& armors,
                const std::vector<Armor>& rejected = {}) {
  // 被阈值/几何过滤掉的大板候选：灰色虚线框 + "?置信度"，用来判断"漏检"是阈值问题还是真没检出
  for (const auto& a : rejected) {
    std::array<cv::Point, 4> pts{};
    for (int i = 0; i < 4; ++i)
      pts[i] = cv::Point(cvRound(a.corners[i].x), cvRound(a.corners[i].y));
    for (int i = 0; i < 4; ++i) cv::line(canvas, pts[i], pts[(i + 1) % 4], cv::Scalar(140, 140, 140), 1);
    char lb[32];
    std::snprintf(lb, sizeof(lb), "?%.2f", a.confidence);
    cv::putText(canvas, lb, cv::Point(pts[0].x, std::max(12, pts[0].y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(160, 160, 160), 1);
  }
  for (const auto& a : armors) {
    std::array<cv::Point, 4> pts{};
    for (int i = 0; i < 4; ++i) {
      pts[i] = cv::Point(cvRound(a.corners[i].x), cvRound(a.corners[i].y));
    }
    const cv::Scalar color = (a.class_id >= 7) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 128, 0);
    for (int i = 0; i < 4; ++i) cv::line(canvas, pts[i], pts[(i + 1) % 4], color, 2);
    for (const auto& p : pts) cv::circle(canvas, p, 3, cv::Scalar(0, 255, 255), -1);

    const cv::Rect box = cv::boundingRect(pts);
    const std::string name =
        (a.class_id >= 0 && a.class_id < kNumClasses) ? kClassNames[a.class_id] : "??";
    char label[64];
    std::snprintf(label, sizeof(label), "%s %.2f %s", name.c_str(), a.confidence,
                  a.large_armor ? "L" : "S");
    cv::putText(canvas, label, cv::Point(box.x, std::max(14, box.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
  }
}

// ── 一次完整推理：图像 → 原始检测（角点已还原到该图像坐标系）────────────
struct RawDet {
  int cls = -1;
  float conf = 0.f;
  std::array<cv::Point2f, 4> quad{};   // TL,BL,BR,TR（数据集顺序，未规范化）
};

std::vector<RawDet> detect_raw(const cv::Mat& image, Detector& det, bool light_norm) {
  std::vector<RawDet> dets;
  if (!det.ok() || image.empty()) return dets;
  const LetterboxResult lb = letterbox(image, det.input_h(), light_norm);
  std::vector<float> out;
  try {
    out = det.infer(lb);
  } catch (const std::exception&) {
    return dets;
  }
  if (!det.is_e2e() || out.size() < static_cast<size_t>(kChannels)) return dets;
  const int n = static_cast<int>(out.size() / kChannels);
  for (int i = 0; i < n; ++i) {
    const float* row = out.data() + static_cast<size_t>(i) * kChannels;
    const float score = row[4];
    if (score < 0.20f) continue;                     // 复核时要看低分候选，先放宽
    const int cls = static_cast<int>(std::lround(row[5]));
    if (cls < 0 || cls >= kNumClasses) continue;
    RawDet d;
    d.cls = cls;
    d.conf = score;
    for (int k = 0; k < kNumKeypoints; ++k) {
      d.quad[k] = cv::Point2f((row[6 + 2 * k] - lb.dx) / lb.scale,
                              (row[7 + 2 * k] - lb.dy) / lb.scale);
    }
    dets.push_back(d);
  }
  return dets;
}

// 灯条长度（左右两条边的均值），用来判断"数字在图上有多大"
inline float plate_bar_len(const std::array<cv::Point2f, 4>& q) {
  return 0.5f * (cv::norm(q[0] - q[1]) + cv::norm(q[2] - q[3]));
}

// 放大重识别：把候选区域裁出来放大，重新跑一次检测，取"靠近裁剪中心"的那个结果。
// 用途：当大板类候选被图案模型判成"数字板"时，不直接丢弃，而是靠放大重新认出正确类别。
bool zoom_reclassify(const cv::Mat& image, const RawDet& d, Detector& det, bool light_norm,
                     Armor* out) {
  cv::Rect br = cv::boundingRect(std::vector<cv::Point2f>(d.quad.begin(), d.quad.end()));
  const int pad = static_cast<int>(0.45f * std::max(br.width, br.height)) + 6;
  cv::Rect roi(br.x - pad, br.y - pad, br.width + 2 * pad, br.height + 2 * pad);
  roi &= cv::Rect(0, 0, image.cols, image.rows);
  if (roi.width < 16 || roi.height < 16) return false;

  const cv::Mat crop = image(roi).clone();
  const std::vector<RawDet> inner = detect_raw(crop, det, light_norm);
  const cv::Point2f c(roi.width * 0.5f, roi.height * 0.5f);
  const RawDet* best = nullptr;
  float best_score = 0.f;
  for (const auto& x : inner) {
    cv::Rect ir = cv::boundingRect(std::vector<cv::Point2f>(x.quad.begin(), x.quad.end()));
    const float dx = std::abs(ir.x + ir.width * 0.5f - c.x);
    const float dy = std::abs(ir.y + ir.height * 0.5f - c.y);
    if (dx > roi.width * 0.30f || dy > roi.height * 0.30f) continue;   // 必须在中心附近
    if (x.conf < 0.30f) continue;
    if (x.conf > best_score) { best_score = x.conf; best = &x; }
  }
  if (!best) return false;

  out->class_id = best->cls;
  out->confidence = std::min(0.99f, best->conf);
  out->large_armor = is_large_class(best->cls);
  for (int k = 0; k < 4; ++k)
    out->corners[k] = canonicalize_corners(best->quad)[k] + cv::Point2f(static_cast<float>(roi.x),
                                                                       static_cast<float>(roi.y));
  return true;
}

// 板面内部纹理能量：把四边形透视矫正到 96x48，取中间区域（避开两侧灯条）算 |Laplacian| 均值。
// 真装甲板中间必有图案/数字 → 能量高；空白结构/车体边角 → 能量低。
inline float pattern_energy(const cv::Mat& image, const std::array<cv::Point2f, 4>& quad) {
  const std::array<cv::Point2f, 4> dst = {
      cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 48.f), cv::Point2f(96.f, 48.f), cv::Point2f(96.f, 0.f)};
  cv::Mat mp = cv::getPerspectiveTransform(quad.data(), dst.data());
  cv::Mat plate;
  cv::warpPerspective(image, plate, mp, cv::Size(96, 48), cv::INTER_LINEAR);
  if (plate.empty()) return 0.f;
  cv::Mat gray;
  cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
  const cv::Rect inner(static_cast<int>(96 * 0.28f), static_cast<int>(48 * 0.18f),
                       static_cast<int>(96 * 0.44f), static_cast<int>(48 * 0.64f));
  cv::Mat lap;
  cv::Laplacian(gray(inner), lap, CV_32F);
  return static_cast<float>(cv::mean(cv::abs(lap))[0]);
}

// 数字复核：把小目标裁剪放大后再识别一次，能认出同一个数字才认为"看到了数字"
bool verify_digit_by_zoom(const cv::Mat& image, const RawDet& d, Detector& det, bool light_norm) {
  cv::Rect br = cv::boundingRect(std::vector<cv::Point2f>(d.quad.begin(), d.quad.end()));
  const int pad = static_cast<int>(0.35f * std::max(br.width, br.height)) + 4;
  cv::Rect roi(br.x - pad, br.y - pad, br.width + 2 * pad, br.height + 2 * pad);
  roi &= cv::Rect(0, 0, image.cols, image.rows);
  if (roi.width < 12 || roi.height < 12) return false;          // 太小，裁了也没用

  const cv::Mat crop = image(roi).clone();
  const std::vector<RawDet> inner = detect_raw(crop, det, light_norm);   // crop 会被 letterbox 放大
  const cv::Point2f c(roi.width * 0.5f, roi.height * 0.5f);
  for (const auto& x : inner) {
    if (x.cls != d.cls) continue;                                // 必须同一个数字
    if (x.conf < kZoomVerifyMinConf) continue;
    // 且要落在裁剪中心附近（防止认到画面边缘别的东西）
    cv::Rect ir = cv::boundingRect(std::vector<cv::Point2f>(x.quad.begin(), x.quad.end()));
    if (std::abs(ir.x + ir.width * 0.5f - c.x) > roi.width * 0.30f) continue;
    if (std::abs(ir.y + ir.height * 0.5f - c.y) > roi.height * 0.30f) continue;
    return true;
  }
  return false;
}

// ── 二级图案分类器：板面 → S(基地) / O(前哨站) ──
class PatternCls {
 public:
  static PatternCls& instance() {
    static PatternCls inst;
    return inst;
  }
  bool ok() const { return ok_; }

  int infer(const cv::Mat& image, const std::array<cv::Point2f, 4>& quad, float* conf) {
    if (!ok_ || image.empty()) return -1;
    const std::array<cv::Point2f, 4> dst = {
        cv::Point2f(0.f, 0.f), cv::Point2f(0.f, static_cast<float>(kPatternH)),
        cv::Point2f(static_cast<float>(kPatternW), static_cast<float>(kPatternH)),
        cv::Point2f(static_cast<float>(kPatternW), 0.f)};      // TL,BL,BR,TR
    cv::Mat mp = cv::getPerspectiveTransform(quad.data(), dst.data());
    cv::Mat plate;
    cv::warpPerspective(image, plate, mp, cv::Size(kPatternW, kPatternH), cv::INTER_LINEAR);
    if (plate.empty()) return -1;

    cv::Mat rgb, f32;
    cv::cvtColor(plate, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    const int area = kPatternW * kPatternH;
    std::vector<float> blob(static_cast<size_t>(3) * area);
    std::vector<cv::Mat> planes(3);
    cv::split(f32, planes);
    for (int c = 0; c < 3; ++c)
      std::memcpy(blob.data() + static_cast<size_t>(c) * area, planes[c].ptr<float>(0),
                  sizeof(float) * area);

    const std::array<int64_t, 4> shape{1, 3, kPatternH, kPatternW};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, blob.data(), blob.size(), shape.data(), shape.size());
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const float* logits = outs[0].GetTensorData<float>();
    const int nc = static_cast<int>(outs[0].GetTensorTypeAndShapeInfo().GetElementCount());
    if (nc < 2) return -1;
    float mx = logits[0];
    for (int i = 1; i < nc; ++i) mx = std::max(mx, logits[i]);
    float sum = 0.f;
    int best = 0;
    float bestv = -1e9f;
    for (int i = 0; i < nc; ++i) {
      const float e = std::exp(logits[i] - mx);
      sum += e;
      if (logits[i] > bestv) { bestv = logits[i]; best = i; }
    }
    const float p = std::exp(bestv - mx) / sum;
    if (conf) *conf = p;
    return best;                                   // 0=D(数字板) 1=O(前哨站) 2=S(基地)
  }

 private:
  PatternCls() {
    try {
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "armor_pattern");
      std::string path = kDefaultPatternPath;
      if (const char* e = std::getenv("ARMOR_PATTERN_ONNX")) path = e;
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(1);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);
      Ort::AllocatorWithDefaultOptions alloc;
      input_name_ = session_->GetInputNameAllocated(0, alloc).get();
      output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
      ok_ = true;
      std::printf("[armor] 图案分类器=%s（S=基地 / O=前哨站）\n", path.c_str());
    } catch (const std::exception& e) {
      ok_ = false;
      std::printf("[armor] 图案分类器加载失败（不影响主流程）：%s\n", e.what());
    }
  }
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_, output_name_;
  bool ok_ = false;
};

}  // namespace

// ── 对外接口 ──────────────────────────────────────────────────────────
std::vector<Armor> detect_full(const cv::Mat& image) {
  std::vector<Armor> armors;


  static const char* display_env = std::getenv("DISPLAY");
  static bool reported = false;
  if (!reported) {
    reported = true;
    std::printf("[armor] 调试窗口=lesson03 armor | DISPLAY=%s\n",
                display_env ? display_env : "(未设置，将不开窗)");
  }
  std::vector<Armor> rejected_dbg;      // 仅用于调试显示
  auto show = [&](const std::vector<Armor>& a) {
    if (!display_env || g_suppress_window) return;
    static cv::Mat canvas;
    if (image.empty()) return;
    image.copyTo(canvas);
    draw_debug(canvas, a, rejected_dbg);
    cv::imshow("lesson03 armor", canvas);
    cv::waitKey(1);          // 只弹出按键消息，不做任何存盘

  };

  if (image.empty()) return armors;

  Detector& det = Detector::instance();
  if (!det.ok()) {                       // 模型没加载成功 → 也要开窗（显示原图）
    show(armors);
    return armors;
  }

  // ③~④b 打包成一段：给一批原始检出，产出"通过所有门槛并去重"的装甲板列表。
  auto to_armors = [&](const std::vector<RawDet>& dets, bool light_norm) {
  std::vector<Armor> out_armors;
  int zoom_used = 0;          // verify_digit_by_zoom 的次数
  int zoom_relabel_used = 0;  // zoom_reclassify 的次数
  // 入口门槛压到"退出阈值"这一档（默认 0.6×门槛）：低于门槛但不算太低的候选也放进来，
  // 交给 ④c 的迟滞状态机决定"显示/不显示"。这正是迟滞能压住闪框的原因：分数在门槛上下来回抖时，已经在显示的目标不会因为一帧掉到 0.44 就消失。
  const float kAdmitConf = kConfThreshold * kExitConfRatio;
  for (const RawDet& d : dets) {
    if (d.conf < kAdmitConf) continue;                     // 采集门槛（很低，交由输出阶段裁决）

    Armor a;
    a.confidence = d.conf;
    a.class_id = d.cls;
    a.corners = canonicalize_corners(d.quad);
    a.large_armor = is_large_class(a.class_id);
    if (a.large_armor) {
      // 面积 = 灯条长度(左右两条短边) × 两灯条间距(上下两条长边)。
      const float bar0 = plate_bar_len(d.quad);
      const float gap0 = 0.5f * (cv::norm(d.quad[0] - d.quad[3]) + cv::norm(d.quad[1] - d.quad[2]));
      if (bar0 * gap0 < kMinLargePlateArea) {        // 太小 → 不可能是真基地/前哨站
        rejected_dbg.push_back(a);
        continue;
      }
    }
    if (kEnableLargeGeometryFilter && a.large_armor && !plausible_large_geometry(a.corners)) {
      rejected_dbg.push_back(a);
      continue;
    }

    // ③b 图案存在性校验：板面中间没有图案/数字 → 一定不是装甲板（默认关）
    if (kEnablePatternEnergyGate && pattern_energy(image, d.quad) < kMinPatternEnergy) {
      rejected_dbg.push_back(a);
      continue;
    }

    // 大板类（基地/前哨站）：用图案分类器判 S/O，修掉"前哨站被判成基地"的域差混淆。
    // 颜色沿用一级检测：BO=5,BS=6,RO=12,RS=13（blue = BO/BS）
    if (a.large_armor && PatternCls::instance().ok() && plate_bar_len(d.quad) >= 24.f) {
      float pc = 0.f;
      const int po = PatternCls::instance().infer(image, d.quad, &pc);   // 0=D 1=O 2=S
      // ① 图案是"数字板"→ 先尝试放大重新识别出正确类别；认不出才丢弃
      //    放大重识别要多跑一次推理，一帧最多做 kZoomReclassMaxPerFrame 次，防止掉帧。
      if (po == 0 && pc >= kPatternVetoConf) {
        if (zoom_relabel_used >= kZoomReclassMaxPerFrame) continue;
        ++zoom_relabel_used;
        Armor relab;
        if (!zoom_reclassify(image, d, det, light_norm, &relab)) continue;   // 放大认不出 → 丢弃
        a = relab;                                               // 用放大后的结果替换
        if (!((a.class_id <= 4) || (a.class_id >= 7 && a.class_id <= 11))) {
          if (a.large_armor) continue;                           // 仍是大板类 → 不可信，丢弃
        }
      }
      // ② 可选：高置信度时用图案纠正基地/前哨站（默认关，见 §8.8.7）
      if (kEnablePatternOverride && po > 0 && pc >= kPatternOverrideConf) {
        const bool blue = (a.class_id == 5 || a.class_id == 6);
        a.class_id = (po == 2) ? (blue ? 6 : 13) : (blue ? 5 : 12);     // S→BS/RS，O→BO/RO
      }
    }

    // 有数字的类（1~5）
    const bool digit_class = (a.class_id <= 4) || (a.class_id >= 7 && a.class_id <= 11);
    if (digit_class) {
      const float bar = plate_bar_len(d.quad);
      if (bar < kMinDigitPlatePx) continue;                  // 数字太小 → 丢弃
      // 看不见数字的小板：必须是"非常有把握"才报
      if (bar < kSmallPlatePx && a.confidence < kSmallPlateConf) {
        rejected_dbg.push_back(a);
        continue;
      }
      if (kEnableZoomVerify && bar < kZoomVerifyPlatePx) {    // 可选：放大复核，默认关
        if (zoom_used >= kZoomVerifyMaxPerFrame) continue;
        ++zoom_used;
        if (!verify_digit_by_zoom(image, d, det, light_norm)) continue;
      }
    }
    out_armors.push_back(a);
  }

  // ④ 按置信度降序（跟踪优先用高分目标）
  std::sort(out_armors.begin(), out_armors.end(), [](const Armor& x, const Armor& y) {
    return x.confidence > y.confidence;
  });

  // ④a 共享角点/边检查：两个检出若共享角点或近似共边 → 物理上不可能，只留一个。
  // 数字 > 大板
  if (out_armors.size() > 1) {
    auto is_digit_cls = [](int c) { return c <= 4 || (c >= 7 && c <= 11); };
    std::vector<Armor> kept;
    for (const auto& a : out_armors) {              // 已按置信度降序
      bool conflict = false;
      for (auto& b : kept) {
        int shared = 0;                             // 共享角点个数
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j)
            if (cv::norm(a.corners[i] - b.corners[j]) < kSharedCornerPx) { ++shared; break; }
        if (shared >= 1) {                          // 共用任一角点即判冲突
          if (is_digit_cls(a.class_id) && !is_digit_cls(b.class_id) &&
              a.confidence >= kExitConfRatio * class_conf_threshold(a.class_id)) {
            b = a;                                  // 数字类顶掉大板类
          }
          conflict = true;
          break;
        }
      }
      if (!conflict) kept.push_back(a);
    }
    out_armors.swap(kept);
  }

  // ④b 去重：同一块板被不同类别重复检出时，“有数字的类（1~5）优先”，因为大板类（基地/前哨站）在仿真域极易在数字板上误触发，且它的置信度常常更高。但**迟滞保留下来的弱框没有资格顶掉大板类
  if (out_armors.size() > 1) {
    auto is_digit = [](int c) { return c <= 4 || (c >= 7 && c <= 11); };
    std::vector<Armor> kept;
    for (const auto& a : out_armors) {             // 已按置信度降序
      bool dup = false;
      for (auto& b : kept) {
        if (quad_iou(a.corners, b.corners) <= kDedupIoU) continue;
        if (a.confidence >= kConfThreshold && is_digit(a.class_id) && !is_digit(b.class_id)) {
          b = a;                                   // 数字类顶掉大板类
        }
        dup = true;
        break;
      }
      if (!dup) kept.push_back(a);
    }
    out_armors.swap(kept);
  }
  return out_armors;
  };   // ← to_armors 结束

  // "有没有一个够门槛的候选"——注意不能只判空：CLAHE 那遍可能产出一堆**不够门槛**的弱候选，
  // 列表非空但一个都不能用，那时同样需要回退。
  auto has_confident = [](const std::vector<Armor>& v) {
    for (const auto& a : v)
      if (a.confidence >= class_conf_threshold(a.class_id)) return true;
    return false;
  };

  // ① + ② 预处理 + 推理（一次，输入做低照度归一化）
  armors = to_armors(detect_raw(image, det, true), true);

  // ③' 回退：这一帧”没有任何够门槛的候选“时，再用未做 CLAHE 的原图跑一遍。
  // 为什么需要它：CLAHE 是为"极暗夜战场景"调出来的，对"近距离大片基地板"反而有害——实测同一块基地板：原图最高分 0.92~0.96，CLAHE 后掉到 0.03~0.73（大面积金属面被拉花）。
  // 只有确实没东西可用时才补这一遍：正常有目标的帧一次推理就够，不白花 12 ms。
  if (kEnableRawRetry && !has_confident(armors)) {
    std::vector<Armor> alt = to_armors(detect_raw(image, det, false), false);
    if (has_confident(alt)) armors.swap(alt);
  }

  // ④c 时域去闪：阈值迟滞。位置一律用当前帧实测值（不平滑、不外推），只对"显不显示"做迟滞，所以既不会跟不上，也不会冲过头。
  static bool track_enabled = kEnableTrack;      // 默认开；ARMOR_TRACK=0 可关
  if (const char* e = std::getenv("ARMOR_TRACK"))
    track_enabled = (std::string(e) != "0");
  if (track_enabled) {
    struct Track {
      Armor a;                 // 最后一次实测的框（丢失帧保持不动，不外推）
      int bad = 0;             // 连续没关联上的帧数
      int support = 0;         // 最近几帧里"在同一位置被检出"的次数（时间证据）
      bool shown = false;      // 上一帧是否真的画出来了
      int pending_cls = -1;    // 待确认的新类别
      int pending_hits = 0;    // 待确认类别连续出现的帧数
    };
    static std::vector<Track> tracks;
    std::vector<bool> used(tracks.size(), false);

    for (const Armor& nw : armors) {              // 已按置信度降序
      int best = -1;
      float best_iou = kTrackIouMatch;
      for (size_t t = 0; t < tracks.size(); ++t) {
        if (used[t]) continue;
        const float v = quad_iou(tracks[t].a.corners, nw.corners);
        if (v > best_iou) { best_iou = v; best = static_cast<int>(t); }
      }
      // 小目标（十几个像素）位置抖一点，IoU 就可能掉到门槛以下，轨迹被切成两条 → “时间证据"清零、又"重新出生"。所以补一条：中心距 < 0.6 倍目标尺寸也算同一个。
      if (best < 0) {
        const auto ctr = [](const std::array<cv::Point2f, 4>& q) {
          cv::Point2f c(0.f, 0.f);
          for (const auto& p : q) c += p;
          return c * 0.25f;
        };
        const auto box_span = [](const std::array<cv::Point2f, 4>& q) {
          float x0 = q[0].x, x1 = q[0].x, y0 = q[0].y, y1 = q[0].y;
          for (int k = 1; k < 4; ++k) {
            x0 = std::min(x0, q[k].x); x1 = std::max(x1, q[k].x);
            y0 = std::min(y0, q[k].y); y1 = std::max(y1, q[k].y);
          }
          return std::max(x1 - x0, y1 - y0);
        };
        float best_d = 1e9f;
        for (size_t t = 0; t < tracks.size(); ++t) {
          if (used[t]) continue;
          const float lim = 0.6f * std::max(box_span(tracks[t].a.corners), box_span(nw.corners));
          const float d = cv::norm(ctr(tracks[t].a.corners) - ctr(nw.corners));
          if (d < lim && d < best_d) { best_d = d; best = static_cast<int>(t); }
        }
      }
      if (best < 0) {                             // 新出现的位置
        if (nw.confidence < class_conf_threshold(nw.class_id)) continue;
        Track nt;
        nt.a = nw;
        nt.support = 1;                           // 本帧算第 1 次证据
        tracks.push_back(nt);
        used.push_back(true);
        continue;
      }
      Track& tr = tracks[best];
      used[best] = true;
      Armor out = nw;                             // 位置/分数都用本帧实测值 → 不滞后不超前
      if (nw.class_id != tr.a.class_id) {         // 类别迟滞：连续 2 帧一致才换标签
        if (tr.pending_cls == nw.class_id) {
          ++tr.pending_hits;
        } else {
          tr.pending_cls = nw.class_id;
          tr.pending_hits = 1;
        }
        if (tr.pending_hits < kClassHoldFrames) { // 还没确认 → 沿用旧标签
          out.class_id = tr.a.class_id;
          out.large_armor = is_large_class(out.class_id);
        } else {
          tr.pending_cls = -1;
          tr.pending_hits = 0;
        }
      } else {
        tr.pending_cls = -1;
        tr.pending_hits = 0;
      }
      tr.bad = 0;
      tr.support = std::min(kSupportFrames, tr.support + 1);   // 同一位置又被看到一次
      tr.a = out;
    }

    for (size_t t = 0; t < tracks.size(); ++t) {  // 没关联上的：不挪位置，证据随时间衰减
      if (used[t]) continue;
      ++tracks[t].bad;
      tracks[t].support = std::max(0, tracks[t].support - 1);
    }

    // 显示规则：高分立刻显示（零延迟）；边缘分数要"时间证据"；
    // 已经显示过的目标，只要本帧还关联得上就继续显示（位置仍是本帧实测值，不是旧位置）。
    armors.clear();
    for (auto& tr : tracks) {
      bool visible = false;
      if (tr.bad == 0) {
        if (tr.a.confidence >= kTrustConf) visible = true;              // 高分 → 零延迟
        else if (tr.shown) visible = true;                              // 已显示 → 不闪
        else if (tr.a.confidence >= kShowConf) visible = true;
        else if (tr.support >= kSupportHits) visible = true;            // 边缘分数 → 攒够证据
      } else if (tr.bad <= kTrackMaxMissed) {
        visible = tr.shown;                                            // 保持（默认 0 → 不显示）
      }
      tr.shown = visible;
      if (visible) armors.push_back(tr.a);
    }
    // 不显示 ≠ 立刻忘掉。被藏起来的轨迹再多留 kSupportFrames 帧，这样它下次被检出时"时间证据"还能接着累计，而不是每断一帧就从头开始。
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [](const Track& t) {
                                  return !t.shown && t.bad > kSupportFrames;
                                }),
                 tracks.end());
  } else {
    // 关闭跟踪时退化为纯粹的逐帧阈值判断
    std::vector<Armor> kept;
    for (const auto& a : armors)
      if (a.confidence >= class_conf_threshold(a.class_id)) kept.push_back(a);
    armors.swap(kept);
  }

  // ⑤ 调试可视化（始终开窗，0 检出也显示原图）
  show(armors);
  return armors;
}

// ============================================================================
// 第三讲补充：ROI 二次检测（小窗口优先 → 全图回退）
// "以上一帧锁定的位置为中心裁一小块，再送进同一个网络"：
// · 等效放大：512×384 的裁片缩到 768×768 ⇒ 放大 1.5~2 倍，小板变"大板"；
// · 更省算力：网络输入固定 768×768，但只用一块 512×384 的图去算，前后处理都更轻。
//
// 代价与对策：
//   ① 裁错窗口 → 裁片里什么都没有 ⇒ 裁片无结果时必须立刻回退全图；
//   ② 目标瞬移/换靶 → 连续依赖先验会锁死在旧位置 ⇒ 每 kFullEvery 帧强制一次全图；
//   ③ 坐标必须回到原图：裁片里的 (px,py) 实际是原图的 (px+x0, py+y0)，否则 PnP/角度全错。
// ============================================================================

namespace {

// ── 裁窗来源：上一帧锁定的装甲板 ───
struct RoiGate {
  // 可调参数
  static constexpr int kWidth = 512;              // 裁片宽（原图像素）
  static constexpr int kHeight = 384;             // 裁片高
  static constexpr int kFullEvery = 4;            // 每 N 帧强制一次全图（防止跟丢）
  static constexpr float kMinLockWidthPx = 8.0f;  // 太窄的框不值得当锚点

  cv::Point2f lock{};        // 上次锁定的板心（原图像素）
  bool has_lock = false;     // 有没有锁定过
  int frame_no = 0;          // 帧计数

  // 本帧要不要裁？返回 std::nullopt 表示"这一帧走全图"。
  std::optional<cv::Rect> next(const cv::Size& full) {
    ++frame_no;
    if (!has_lock || full.width <= 0 || full.height <= 0) return std::nullopt;
    if (kFullEvery > 1 && frame_no % kFullEvery == 0) return std::nullopt;   // 定期回退
    const int w = std::min(full.width, std::max(128, kWidth));
    const int h = std::min(full.height, std::max(128, kHeight));
    int x0 = static_cast<int>(std::lround(lock.x - w * 0.5f));
    int y0 = static_cast<int>(std::lround(lock.y - h * 0.5f));
    x0 = std::clamp(x0, 0, full.width - w);
    y0 = std::clamp(y0, 0, full.height - h);
    return cv::Rect(x0, y0, w, h);
  }

  // 用本帧检测结果更新锚点：取"最宽、且不是基地/前哨站"的那块。
  // 为什么按宽度取？框宽 ∝ 1/距离，最宽的通常就是最近、最值得打的那辆。
  void update(const std::vector<Armor>& armors) {
    float best = kMinLockWidthPx;
    for (const auto& a : armors) {
      if (is_large_class(a.class_id)) continue;   // {5,6,12,13} 基地/前哨站不是自瞄目标
      const float w = static_cast<float>(cv::norm(a.corners[1] - a.corners[0]));
      if (w > best) {
        best = w;
        lock = (a.corners[0] + a.corners[1] + a.corners[2] + a.corners[3]) * 0.25f;
        has_lock = true;
      }
    }
  }
};

// 把裁片上的结果搬回原图坐标系（漏了这一步，云台会朝左上角飞）
inline void shift_armors(std::vector<Armor>& armors, const cv::Point& offset) {
  for (auto& a : armors)
    for (auto& p : a.corners) {
      p.x += static_cast<float>(offset.x);
      p.y += static_cast<float>(offset.y);
    }
}

}  // namespace

// 输入/输出约定与第三讲正文一致（角点顺序 左上→右上→右下→左下，单位像素）。
std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  static RoiGate gate;                       // 跨帧状态：单线程主循环使用

  if (image.empty()) return detect_full(image);

  const cv::Rect roi = gate.next(image.size()).value_or(cv::Rect{});
  if (roi.width <= 0) {                      // 本帧走全图
    std::vector<Armor> out = detect_full(image);
    gate.update(out);
    return out;
  }

  // ① 先看小窗口（等效放大）：这一趟**不画窗口**（否则窗口尺寸在裁片/全图之间跳，会闪）
  g_suppress_window = true;
  std::vector<Armor> out = detect_full(image(roi));
  g_suppress_window = false;
  bool drew_full = false;
  if (out.empty()) {                         // ② 小窗口没结果 → 立刻回退全图（逃生通道）
    out = detect_full(image);                //    这趟会正常画窗口（尺寸=原图，固定）
    drew_full = true;
  } else {
    shift_armors(out, roi.tl());             // ③ 坐标搬回原图
  }
  gate.update(out);

  // ④ ROI 成功那几帧不会走到全图绘制 → 这里补一次"固定尺寸全图"的显示，保证窗口不闪
  if (!drew_full && std::getenv("DISPLAY") && !image.empty()) {
    static cv::Mat canvas;
    image.copyTo(canvas);                    // 尺寸恒为原图 1440×1080
    draw_debug(canvas, out);
    cv::imshow("lesson03 armor", canvas);
    cv::waitKey(1);
  }
  return out;
}

}  // namespace training
