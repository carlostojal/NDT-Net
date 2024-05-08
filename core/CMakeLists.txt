cmake_minimum_required(VERSION 3.22)

project(ndtnetpp VERSION 0.1 LANGUAGES C)

# declare the library
add_library(ndtnetpp
    SHARED
    src/ndt.c
)

# set the include directory
include_directories(include)
