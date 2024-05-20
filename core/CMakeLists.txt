cmake_minimum_required(VERSION 3.22)

project(ndtnetpp VERSION 0.1 LANGUAGES C CXX)

# set debug mode
set(CMAKE_BUILD_TYPE Debug)

# find the GSL (GNU Scientific Library) package
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED) # for testing

# declare the library
add_library(ndtnetpp
    SHARED
    src/ndt.c
    src/pointclouds.c
    src/voxel.c
    src/matrix.c
)

# declare the tests executable
add_executable(tests
    tests/test_pointclouds.cpp
    tests/test_voxel.cpp
)

# set the include directory
include_directories(include ${GSL_INCLUDE_DIRS} ${OPENMP_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

# link the GSL library
target_link_libraries(ndtnetpp GSL::gsl GSL::gslcblas ${OPENMP_LIBRARIES})

# link the tests executable
target_link_libraries(tests GTest::gtest GTest::gtest_main ${OPENMP_LIBRARIES} ndtnetpp)
