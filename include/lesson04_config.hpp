#pragma once
// ============================================================================
// lesson04_config.hpp —— 第四讲全部可调参数
// ----------------------------------------------------------------------------
// 调参的时候只需要改这一个文件
// ============================================================================

namespace training {
namespace l4 {

// ---------------------------------------------------------------- 模式开关
// kTrackerMode 是本讲唯一的"架构开关"，用来在四种跟踪器之间切换
//   0 = 板级 EKF，单模型"匀速"（CV）——最简、最稳，回退用
//   1 = 板级 EKF，单模型"常加速"（CA）——机动跟得上，静止时略抖
//   2 = 板级 EKF + IMM(CV/CA 双模型)——默认，兼顾两者（实测最优）
//   3 = 环模型 EKF（12 维，含车心/半径/自转角）——自转靶、跳板场景的进阶方案
constexpr int kTrackerMode = 2;

// 是否启用 IMM（多模型）。只有 kTrackerMode == 2 时才生效。
constexpr bool kEnableImm = (kTrackerMode == 2);

// ---------------------------------------------------------------- 装甲板尺寸
// ★ 关键参数：网络输出的 4 个角点围成的是"两条灯条的外接矩形"，
//  换真车记得测实际数据
constexpr double kArmorWidthSmall = 0.135;   // m，小装甲板（1~5 号）：灯条外缘间距 135 mm
constexpr double kArmorHeightSmall = 0.055;  // m，灯条长度 55 mm
constexpr double kArmorWidthLarge = 0.225;   // m，大装甲板（前哨站/基地）：225 mm
constexpr double kArmorHeightLarge = 0.055;  // m，大板灯条长度同为 55 mm

// ---------------------------------------------------------------- PnP 相关
constexpr double kReprojGatePx = 1.5;   // 重投影误差超过它 → 丢弃该观测
constexpr double kMaxDistanceM = 12.0;  // 超过它 → 视为误检（3 m 场景留足余量）
constexpr double kMinDistanceM = 0.5;   // 小于它 → 视为误检
constexpr bool kEnableRefineLm = true;  // IPPE 之后用 LM 再精化一次（§4.6）

// ---------------------------------------------------------------- 观测与滤波
// 观测噪声（标准差，1σ）。这三个数决定"滤波器有多信任单帧观测"。
// 讲义 §6.8 给出标定方法与实测推荐值。
constexpr double kSigmaPosXy = 0.020;   // m，板中心水平位置
constexpr double kSigmaPosZ = 0.030;    // m，板中心高度（更差：受角点垂直精度限制）
constexpr double kSigmaYaw = 0.12;      // rad ≈ 7°，板朝向（最差的一维）

// 过程噪声（谱密度，连续时间模型；离散化在 Q 里按 dt 幂次展开）
constexpr double kAccelNoise = 0.5;     // m/s²（CA 模型 / IMM 高动态模型用）
constexpr double kCvAccelNoise = 0.3;   // m/s²（环模型的 CV 分支：把加速度"拉回 0"的强度）
// IMM 里那两个子模型各自的过程噪声。为什么不直接用 kAccelNoise/kCvAccelNoise：IMM 的加权本身会带来额外平滑，所以低动态子模型取更小的 0.15 才与"单模型 CA"等效。
constexpr double kLowDynAccelNoise = 0.15;  // m/s²（IMM 的低动态/CV 子模型）
constexpr double kFallbackAccelNoise = 0.05; // m/s²（模式 0 的纯匀速回退：几乎不信加速度）
constexpr double kYawAccelNoise = 8.0;  // rad/s²（自转靶 2 rad/s 时的角加速度量级）
// 环半径 r：默认"冻结"成常数。
// 原因：当目标正对相机且不自转时，"车心 x" 与 "r" 在数学上完全不可分，把 r 当自由状态去估计会让 x 沿不可观方向漂走（实测会漂 3 m 以上）。
// 只有确认目标在小范围内自转（yaw 变化 > 30°）时，才把 kEstimateRadius 打开。
constexpr bool kEstimateRadius = false;
constexpr double kRingRadius = 0.22;
// 整车环模型预测里，车心平动项的置信速度尺度：自转切向速度远大于它时，板速主要是"绕车心转"，扣出的平动量不可信，按 1/(1+(tang/v_ref)^2) 衰减（§6.11.4）。
constexpr double kCenterVelRefSpeed = 0.15;   // m/s
// 评测用：预测队列里"已经过时这么多"的条目直接丢弃。
// 不丢的话，一旦中间有掉检帧，队列会一直积压，后面每次配对都是拿很旧的预测去比新观测，误差被人为放大（实测过：平移靶从 97.6 mm 虚高到 189.7 mm，见 §9.4.4 坑 1）。
constexpr double kEvalMaxStaleSec = 0.10;   // s，见 §9.4.4
// 预测用的角速度做 EMA 平滑（时间常数，秒）。换板瞬间 vyaw 会被 yaw 重置带出尖峰，直接拿它做 250 ms 圆弧外推会被放大成几十毫米误差
constexpr double kVyawEmaTau = 0.18;    // s，预测专用角速度 EMA 时间常数
constexpr double kRadiusNoise = kEstimateRadius ? 0.05 : 0.0;  // m/√s
constexpr double kRadiusPriorSigma = kEstimateRadius ? 0.08 : 0.002;  // m，初始 1σ
constexpr double kCvAccelTau = 0.05;    // s，CV 模型的加速度衰减时间常数

// 环半径的物理约束（防止发散；赛规下装甲板环半径约 0.12~0.40 m，程序放宽到 0.10~0.45）
constexpr double kRadiusMin = 0.10;
constexpr double kRadiusMax = 0.45;

// IMM 模型切换参数
constexpr double kImmSwitchDwell = 0.15;  // s，模型平均逗留时间
constexpr double kImmProbFloor = 1e-4;    // 概率下限，防止数值下溢把模型锁死

// ---------------------------------------------------------------- 关联与状态机
constexpr double kGateChi2 = 18.47;      // 马氏距离门限（4 自由度 99.9% 分位）
constexpr int kDetectFrames = 3;         // 连续命中几帧 → 进入 TRACKING
constexpr int kLostFrames = 15;          // 连续丢几帧 → 退回 LOST（20 FPS 下≈0.75 s，够跨过板切换空档）

// ---- 装甲板跳变（自转靶换板）----
// 同一台车的板每隔 360°/armor_num（4 板时 90°）就有一块新的转进视野。旧设计用"方位残差必须 < kYawGateRad" 的硬门限，而 90° > 69°，换板那一帧会被直接拒掉，于是状态长时间停在 TEMP_LOST、误差被尖峰拉高。正确做法：允许"整块板跳变"。
constexpr double kYawJumpTolRad = 0.45;  // 换板后残余方位误差容限（rad）≈26°
constexpr double kPlateJumpMargin = 0.35; // 换板时位置门限在 2r 之外再放宽的余量（m）
// 换板必须伴随"位置真的跳了一段"：相邻两板中心相距 2r·sin(45°)=√2 r ≈ 0.31 m。
// 若方位差刚好接近 90° 但位置几乎没动（连续自转、滤波器只是滞后），那是普通匹配，不是换板。
constexpr double kPlateJumpMinDist = 0.25;  // m
// 更硬的判据：换板只可能发生在"上一块板已经看不见了"之后 —— 即必须先连续丢过几帧。
// 有了这一条，正常跟随时绝不会误判换板（连续自转也安全）。
constexpr int kJumpMinMisses = 2;
constexpr double kJumpCovInflate = 16.0;  // 换板后位置/朝向方差放大倍数（表示"刚换了目标点"）

// ---- 机动自适应（Q 自适应）----
// 归一化新息持续偏大 = 模型跟不上（急起急停、随机运动）。此时临时放大过程噪声，让滤波器"更信观测"，而不是把观测当野值拒掉。
constexpr double kManeuverEmaAlpha = 0.2;   // 新息幅度的 EMA 系数
constexpr double kManeuverThreshold = 2.5;  // 归一化新息超过它开始放大 Q
constexpr double kManeuverQMax = 6.0;       // Q 最多放大到 6 倍
constexpr int kMaxTracks = 4;            // 同时最多维护几条目标轨迹（场上多车）
constexpr double kYawGateRad = 1.2;      // 关联时的朝向门限（rad）≈ 69°
constexpr double kNewTrackSuppressDist = 0.35;  // 与现有轨迹太近的观测不新建轨迹（m）
constexpr double kMaxTrackAgeSec = 0.5;  // 超过这么久没观测到就回收轨迹（s）
constexpr double kArcPredictMinVyaw = 0.5;  // |角速度| 超过它才用环形外推（rad/s）

// ---------------------------------------------------------------- 预测
// 作业要求的预测提前量。改这里即可改成任意 n（毫秒）。
constexpr double kPredictHorizonSec = 0.25;
// 误差配对窗口：预测时刻与当前时刻相差小于它才算"到点了"（半个帧间隔）
constexpr double kEvalMatchTolSec = 0.010;
constexpr int kEvalHistorySize = 64;     // 预测历史队列长度

// ---------------------------------------------------------------- 调试输出
constexpr bool kVerbosePnp = false;      // 每帧打印 PnP 细节（很吵，调试时开）
constexpr int kPrintEveryMs = 500;       // 状态行打印间隔（毫秒）；别设太小，终端 I/O 会拖慢主循环

}  // namespace l4
}  // namespace training

