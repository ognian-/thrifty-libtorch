include(CTest)

function(netam_test_executable PRODUCT)
  add_executable(${PRODUCT} ${ARGN})
  target_include_directories(${PRODUCT} PRIVATE
    test/include)
  target_link_libraries(${PRODUCT} PRIVATE netam-libtorch)
  add_test(${PRODUCT} ${PRODUCT} COMMAND ${PRODUCT})
endfunction()

netam_test_executable(test_basic test/basic.cpp)
netam_test_executable(test_kmer_sequence_encoder test/kmer_sequence_encoder.cpp)
