add_executable(main-example)
target_link_libraries(main-example PRIVATE netam-libtorch)
target_sources(main-example PRIVATE examples/main.cpp)

add_executable(thrifty-demo)
target_link_libraries(thrifty-demo PRIVATE netam-libtorch)
target_sources(thrifty-demo PRIVATE examples/thrifty_demo.cpp)
