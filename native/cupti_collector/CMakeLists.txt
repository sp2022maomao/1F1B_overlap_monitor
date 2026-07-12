cmake_minimum_required(VERSION 3.20)
project(overlap_cupti_collector LANGUAGES CXX)

find_package(CUDAToolkit 12.0 REQUIRED)

add_library(overlap_cupti SHARED src/collector.cpp)
target_compile_features(overlap_cupti PRIVATE cxx_std_17)
target_include_directories(overlap_cupti PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(overlap_cupti PRIVATE CUDA::cupti CUDA::cuda_driver)
set_target_properties(overlap_cupti PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

install(TARGETS overlap_cupti LIBRARY DESTINATION lib)
install(FILES include/overlap_cupti.h DESTINATION include)
