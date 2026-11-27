cmake_minimum_required(VERSION 3.16)
project(hnurmvision_training LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(hnurm_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d freetype)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# ONNX Runtime：优先 -DONNXRUNTIME_ROOT，其次环境变量，再次 third_party/onnxruntime
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime 根目录（含 include/ 与 lib/）")
if(NOT ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  else()
    foreach(_cand "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" "$ENV{HOME}/train/opt/onnxruntime")
      if(EXISTS "${_cand}/include/onnxruntime_cxx_api.h" AND EXISTS "${_cand}/lib/libonnxruntime.so")
        set(ONNXRUNTIME_ROOT "${_cand}")
        break()
      endif()
    endforeach()
  endif()
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  message(FATAL_ERROR "未找到 ONNX Runtime：用 -DONNXRUNTIME_ROOT=<含 include/ 与 lib/ 的目录>，或放到 third_party/onnxruntime")
endif()

# CUDA 12 运行库（仅本机 cu13 环境需要）：探测到才写进 RPATH
set(ORT_CUDA_LIBS "" CACHE PATH "CUDA 12 运行库目录（可选）")
if(NOT ORT_CUDA_LIBS)
  foreach(_cand "$ENV{HOME}/train/ort-cu12/lib" "$ENV{HOME}/ort-cu12/lib")
    if(EXISTS "${_cand}/libcublas.so.12")
      set(ORT_CUDA_LIBS "${_cand}")
      break()
    endif()
  endforeach()
endif()

add_library(training_io io/config.cpp)
target_include_directories(training_io PUBLIC include io ${OpenCV_INCLUDE_DIRS})
target_link_libraries(training_io Eigen3::Eigen yaml-cpp ${OpenCV_LIBS})
ament_target_dependencies(training_io rclcpp sensor_msgs hnurm_interfaces)
target_compile_options(training_io PRIVATE -Wall -Wextra -Wpedantic)


add_library(training_lessons lessons/lesson02_io.cpp
 lessons/lesson03_detection.cpp lessons/lesson04_tracking.cpp
 lessons/lesson05_aiming.cpp lessons/lesson06_estimator.cpp
 lessons/rm_detector.cpp lessons/rm_io.cpp
 lessons/rm_lightbar.cpp lessons/rm_viz.cpp)
 
target_include_directories(training_lessons PUBLIC "${CMAKE_SOURCE_DIR}/include/rm_aim")
target_link_libraries(training_lessons PUBLIC training_io)
target_include_directories(training_lessons PUBLIC "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(training_lessons PUBLIC "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

set(_ort_rpath "${ONNXRUNTIME_ROOT}/lib")
if(ORT_CUDA_LIBS)
  string(APPEND _ort_rpath ";${ORT_CUDA_LIBS}")
endif()
set_target_properties(training_lessons PROPERTIES
    BUILD_RPATH   "${_ort_rpath}"
    INSTALL_RPATH "${_ort_rpath}")

add_executable(training src/main.cpp)
target_link_libraries(training PRIVATE training_lessons)
