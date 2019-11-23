cmake_minimum_required(VERSION 3.15)
project(caffe_model)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_executable(caffe_model src/main.cpp)

target_link_libraries(caffe_model ${OpenCV_LIBS})