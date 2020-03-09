cmake_minimum_required(VERSION 3.8)

project(facenet)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

find_package(OpenCV 4 REQUIRED)
find_package(TensorflowCC REQUIRED) # https://github.com/FloopCZ/tensorflow_cc
find_package(Eigen3 REQUIRED)

set(SOURCE_FILES Classifier.hpp classify.cpp facenet_tf.cpp)
add_executable(facenet ${SOURCE_FILES})

target_link_libraries(facenet ${OpenCV_LIBS})
target_link_libraries(facenet TensorflowCC::Shared)

# find_package(CUDA)
# if(CUDA_FOUND)
#   target_link_libraries(facenet ${CUDA_LIBRARIES})
# endif()
