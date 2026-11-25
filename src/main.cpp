#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <csignal>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
}

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  while (!interrupted) {
    const auto input = devices.read();
    if (!input) continue;    
    training::Command command{};
    if (input) {
      const auto armors = training::lesson03_detection(input->image);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target) {
        // 状态机信息，用来判断"这一帧的目标值不值得信"
        const auto* tr = tracking.manager.selected();
        const bool trustworthy =
            (tr != nullptr) && tr->tracking() &&                 // TRACKING / TEMP_LOST
            (tr->misses() <= 3) &&                               // 没有连续丢检
            (tr->bank().position_sigma() < 0.10);                // 位置不确定度 < 10 cm

        training::lesson05_set_attitude(input->imu_q, tracking.solver.R_world0());
        if (trustworthy)
          command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
        // 不 trusted 时 command 保持默认（control=false / shoot=false）→ IO 发 LOST_TARGET
      }
    }
    if (interrupted) break;
    devices.write(command);

    if (input && devices.config.display) {
      training::lesson04_display(input->image, tracking);
      cv::imshow("RoboMaster training", input->image);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }
  }

  devices.close();
  return 0;
}