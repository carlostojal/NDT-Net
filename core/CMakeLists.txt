cmake_minimum_required(VERSION 3.22)

project(ndtnetpp VERSION 0.1 LANGUAGES C)

# find the GSL (GNU Scientific Library) package
find_package(GSL REQUIRED)

# declare the library
add_library(ndtnetpp
    SHARED
    src/ndt.c
    src/pointclouds.c
    src/voxel.c
)

# set the include directory
include_directories(include)

# link the GSL library
target_link_libraries(ndtnetpp GSL::gsl GSL::gslcblas)
