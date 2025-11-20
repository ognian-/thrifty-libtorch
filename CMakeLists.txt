cmake_minimum_required(VERSION 3.20)
project(thrifty-libtorch VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

add_executable(thrifty-libtorch)
target_include_directories(thrifty-libtorch PUBLIC include)
target_link_libraries(thrifty-libtorch PUBLIC "${TORCH_LIBRARIES}")
target_sources(thrifty-libtorch PRIVATE src/main.cpp)
