cmake_minimum_required(VERSION 3.22)

project(ndnet VERSION 0.1 LANGUAGES C CXX)

# set debug mode
set(CMAKE_BUILD_TYPE Debug)

# find the GSL (GNU Scientific Library) package
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED) # for testing

# declare the library
add_library(ndnet
    SHARED
    src/normal_distributions.c
    src/kullback_leibler.c
    src/ndt.c
    src/pointclouds.c
    src/voxel.c
    src/matrix.c
)

# declare the tests executable
add_executable(tests
    tests/test_pointclouds.cpp
)

# test ndt downsample
add_executable(test_ndt_downsample
    tests/ndt_downsample.c
)

# set the include directory
include_directories(include ${GSL_INCLUDE_DIRS} ${OPENMP_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

# link the GSL library
target_link_libraries(ndnet GSL::gsl GSL::gslcblas ${OPENMP_LIBRARIES})

# link the tests executable
target_link_libraries(tests GTest::gtest GTest::gtest_main ${OPENMP_LIBRARIES} ndnet)

target_link_libraries(test_ndt_downsample ndnet)
