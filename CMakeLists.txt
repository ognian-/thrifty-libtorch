cmake_minimum_required(VERSION 3.28)
project(netam-libtorch VERSION 0.0.1 LANGUAGES CXX)

include(cmake/opts.cmake)
include(cmake/deps.cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NETAM_STRICT_WARNINGS -Wall -Wextra -pedantic -Wconversion -Wold-style-cast
  -Wdouble-promotion -Wnull-dereference -Wswitch-enum -Wuseless-cast
  -Wuninitialized -Wstrict-aliasing=3 -Wshadow -Wcast-align=strict -Wcast-qual
  -Wlogical-op -Wlogical-not-parentheses -Wredundant-decls -Wunreachable-code
)

add_library(netam-libtorch)
target_include_directories(netam-libtorch PUBLIC include)
target_compile_options(netam-libtorch PUBLIC -Wno-unknown-warning-option)
target_compile_options(netam-libtorch PUBLIC ${NETAM_STRICT_WARNINGS})
target_link_libraries(netam-libtorch PUBLIC fmt::fmt)
target_link_libraries(netam-libtorch PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(netam-libtorch PUBLIC yaml-cpp::yaml-cpp)
target_link_libraries(netam-libtorch PUBLIC ZLIB::ZLIB)
target_link_libraries(netam-libtorch PUBLIC Matplot++::matplot)
target_sources(netam-libtorch PRIVATE
  src/indep_rscnn_model.cpp
  src/kmer_sequence_encoder.cpp
  src/pcp_dataframe.cpp
  src/crepe.cpp
  src/likelihood.cpp
)

include(examples/examples.cmake)
include(test/tests.cmake)
