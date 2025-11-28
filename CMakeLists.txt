cmake_minimum_required(VERSION 3.30)
project(netam-libtorch VERSION 0.0.1 LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG "yaml-cpp-0.7.0"
  GIT_SHALLOW true
  GIT_PROGRESS true
  UPDATE_DISCONNECTED true
  CMAKE_ARGS "-DYAML_BUILD_SHARED_LIBS=no"
)
FetchContent_MakeAvailable(yaml-cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

set(NETAM_STRICT_WARNINGS -Wall -Wextra -pedantic -Wconversion -Wold-style-cast
  -Wdouble-promotion -Wnull-dereference -Wnrvo -Wswitch-enum -Wuseless-cast
  -Wuninitialized -Wstrict-aliasing=3 -Wshadow -Wcast-align=strict -Wcast-qual
  -Wlogical-op -Wlogical-not-parentheses -Wredundant-decls -Wunreachable-code
)

add_executable(netam-libtorch)
target_include_directories(netam-libtorch PUBLIC include)
target_compile_options(netam-libtorch PUBLIC -Wno-unknown-warning-option)
target_compile_options(netam-libtorch PUBLIC ${NETAM_STRICT_WARNINGS})
target_link_libraries(netam-libtorch PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(netam-libtorch PUBLIC yaml-cpp::yaml-cpp)
target_sources(netam-libtorch PRIVATE src/main.cpp)
