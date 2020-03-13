cmake_minimum_required(VERSION 3.8)

project(facenet)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV 4 REQUIRED)
# you can also use https://github.com/cjweeks/tensorflow-cmake#step-3-configure-the-cmake-project
# if you have already installed tensorflow, but i had some problems with is
find_package(TensorflowCC REQUIRED) # https://github.com/FloopCZ/tensorflow_cc
find_package(Eigen3 REQUIRED)

add_library(facenet STATIC Classifier.hpp facenet_tf.cpp)

target_link_libraries(facenet ${OpenCV_LIBS})
target_link_libraries(facenet TensorflowCC::Shared)

find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "Cuda enabled")
    target_link_libraries(facenet ${CUDA_LIBRARIES})
endif ()

add_executable(run classify.cpp)
target_link_libraries(run facenet)
