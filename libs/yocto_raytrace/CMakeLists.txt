add_library(yocto_trace yocto_raytrace.h yocto_raytrace.cpp)

set_target_properties(yocto_trace PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(yocto_trace PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(yocto_trace yocto)
