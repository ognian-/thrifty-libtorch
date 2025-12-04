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

