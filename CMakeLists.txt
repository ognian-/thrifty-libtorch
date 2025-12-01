cmake_minimum_required(VERSION 3.30)
project(netam-libtorch VERSION 0.0.1 LANGUAGES CXX)

option(NETAM_USE_ASAN "Use address sanitizer" OFF)
option(NETAM_USE_TSAN "Use thread sanitizer" OFF)
option(NETAM_USE_MSAN "Use memory sanitizer" OFF)
option(NETAM_USE_MOLD "Use mold linker" OFF)
option(NETAM_USE_LTO "Use lto" OFF)

if(NETAM_USE_ASAN AND NETAM_USE_TSAN)
  message(FATAL_ERROR "Can't use address and thread sanitizer together")
endif()

if(NETAM_USE_ASAN AND NETAM_USE_MSAN)
  message(FATAL_ERROR "Can't use address and memory sanitizer together")
endif()

if(NETAM_USE_TSAN AND NETAM_USE_MSAN)
  message(FATAL_ERROR "Can't use thread and memory sanitizer together")
endif()

if(NETAM_USE_ASAN)
  add_compile_options("-fsanitize=address,undefined")
  add_link_options("-fsanitize=address,undefined")
endif()

if(NETAM_USE_TSAN)
  add_compile_options("-fsanitize=thread,undefined")
  add_link_options("-fsanitize=thread,undefined")
endif()

if(NETAM_USE_MSAN)
  add_compile_options("-fsanitize=memory,undefined")
  add_link_options("-fsanitize=memory,undefined")
endif()

if(NETAM_USE_MOLD)
  add_link_options("-fuse-ld=mold")
endif()

if(NETAM_USE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

include(FetchContent)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG "12.1.0"
  GIT_SHALLOW true
  GIT_PROGRESS true
  UPDATE_DISCONNECTED true
)
FetchContent_MakeAvailable(fmt)

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
find_package(ZLIB REQUIRED)

set(NETAM_STRICT_WARNINGS -Wall -Wextra -pedantic -Wconversion -Wold-style-cast
  -Wdouble-promotion -Wnull-dereference -Wnrvo -Wswitch-enum -Wuseless-cast
  -Wuninitialized -Wstrict-aliasing=3 -Wshadow -Wcast-align=strict -Wcast-qual
  -Wlogical-op -Wlogical-not-parentheses -Wredundant-decls -Wunreachable-code
)

add_executable(netam-libtorch)
target_include_directories(netam-libtorch PUBLIC include)
target_compile_options(netam-libtorch PRIVATE -Wno-unknown-warning-option)
target_compile_options(netam-libtorch PRIVATE ${NETAM_STRICT_WARNINGS})
target_link_libraries(netam-libtorch PRIVATE fmt::fmt)
target_link_libraries(netam-libtorch PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(netam-libtorch PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(netam-libtorch PRIVATE ZLIB::ZLIB)
target_sources(netam-libtorch PRIVATE src/main.cpp)
