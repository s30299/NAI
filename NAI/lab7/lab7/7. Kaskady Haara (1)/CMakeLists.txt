cmake_minimum_required(VERSION 3.23)
project(07_opencv)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 20)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(07_opencv main.cpp)
target_link_libraries( 07_opencv ${OpenCV_LIBS} )

