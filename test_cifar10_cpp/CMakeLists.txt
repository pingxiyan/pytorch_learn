cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(TARGET_NAME testapp)
project(${TARGET_NAME})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
