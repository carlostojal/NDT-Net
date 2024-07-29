#include "gtest/gtest.h"
#include <ndnet_core/ndt.h>
#include <iostream>

TEST(NDTTests, TestDownsample) {
    // 16 points
    double pointcloud[48] = {
        -1.0, 1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, 1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0,
        -0.5, 0.5, -0.5,
        0.5, -0.5, -0.5,
        0.5, 0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, 0.5, 0.5,
        0.5, -0.5, 0.5,
        0.5, 0.5, 0.5,
        -0.5, -0.5, 0.5
    };

    double downsampled[24];
    unsigned long num_points;
    ndt_downsample(pointcloud, 3, 16, 8,
                   downsampled, &num_points);

    EXPECT_EQ(num_points, 8);
    
    print_matrix(downsampled, num_points, 3);
}

TEST(NDTTests, TestDownsample2) {
    // 16 points
    double pointcloud[48] = {
        -1.0, 1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, 1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0,
        -0.5, 0.5, -0.5,
        0.5, -0.5, -0.5,
        0.5, 0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, 0.5, 0.5,
        0.5, -0.5, 0.5,
        0.5, 0.5, 0.5,
        -0.5, -0.5, 0.5
    };

    double downsampled[12];
    unsigned long num_points;
    ndt_downsample(pointcloud, 3, 16, 4,
                   downsampled, &num_points);

    EXPECT_EQ(num_points, 4);
    
    print_matrix(downsampled, num_points, 3);
}

TEST(NDTTests, TestDownsample3) {
    // 16 points
    double pointcloud[48] = {
        -1.0, 1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        -1.01, -1.0, -1.0,
        -1.0, 1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0,
        -0.5, 0.5, -0.51,
        0.5, -0.5, -0.5,
        0.5, 0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, 0.48, 0.5,
        0.5, -0.5, 0.5,
        0.5, 0.52, 0.5,
        -0.5, -0.5, 0.5
    };

    double downsampled[12];
    unsigned long num_points;
    ndt_downsample(pointcloud, 3, 16, 4,
                   downsampled, &num_points);

    EXPECT_EQ(num_points, 4);
    
    print_matrix(downsampled, num_points, 3);
}

TEST(NDTTests, TestDownsample4) {
    // 16 points
    double pointcloud[48] = {
        -1.0, 1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, 1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0,
        -0.5, 0.5, -0.5,
        0.5, -0.5, -0.5,
        0.5, 0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, 0.5, 0.5,
        0.5, -0.5, 0.5,
        0.5, 0.5, 0.5,
        -0.5, -0.5, 0.5
    };

    double downsampled[24];
    unsigned long num_points;
    ndt_downsample(pointcloud, 3, 16, 3,
                   downsampled, &num_points);

    EXPECT_EQ(num_points, 3);
    
    print_matrix(downsampled, num_points, 3);
}
