cmake_minimum_required(VERSION 3.20)
project(netam-libtorch VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

add_executable(netam-libtorch)
target_include_directories(netam-libtorch PUBLIC include)
target_link_libraries(netam-libtorch PUBLIC "${TORCH_LIBRARIES}")
target_sources(netam-libtorch PRIVATE src/main.cpp)
