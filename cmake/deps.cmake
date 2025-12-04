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

FetchContent_Declare(matplotplusplus
  GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
  GIT_TAG "v1.2.2"
  GIT_SHALLOW true
  GIT_PROGRESS true
  UPDATE_DISCONNECTED true
  CMAKE_ARGS "-DMATPLOTPP_BUILD_EXAMPLES=no -DMATPLOTPP_BUILD_TESTS=no"
)
FetchContent_MakeAvailable(matplotplusplus)
if(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  target_compile_definitions(matplot PUBLIC __linux)
endif()

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

