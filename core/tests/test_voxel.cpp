#include "gtest/gtest.h"
#include <ndtnetpp_core/voxel.h>

TEST(VoxelTests, TestEstimateVoxelSize) {
    double max_x = 2.0;
    double max_y = 1.5;
    double max_z = 0.5;
    double min_x = -2.0;
    double min_y = -1.5;
    double min_z = -0.5;
    double voxel_size;
    int len_x, len_y, len_z;
    estimate_voxel_size(12, max_x, max_y, max_z, min_x, min_y, min_z, &voxel_size, &len_x, &len_y, &len_z);
    EXPECT_EQ(voxel_size, 1);
    EXPECT_EQ(len_x, 4);
    EXPECT_EQ(len_y, 3);
    EXPECT_EQ(len_z, 1);
}

TEST(VoxelTests, TestEstimateVoxelSize2) {
    double max_x = 2.0;
    double max_y = 2.0;
    double max_z = 1.0;
    double min_x = -2.0;
    double min_y = -2.0;
    double min_z = -1.0;
    double voxel_size;
    int len_x, len_y, len_z;
    estimate_voxel_size(32, max_x, max_y, max_z, min_x, min_y, min_z, &voxel_size, &len_x, &len_y, &len_z);
    EXPECT_EQ(voxel_size, 1);
    EXPECT_EQ(len_x, 4);
    EXPECT_EQ(len_y, 4);
    EXPECT_EQ(len_z, 2);
}

TEST(VoxelTests, TestEstimateVoxelSize3) {
    double max_x = 2.0;
    double max_y = 2.0;
    double max_z = 1.0;
    double min_x = -2.0;
    double min_y = -2.0;
    double min_z = -1.0;
    double voxel_size;
    int len_x, len_y, len_z;
    estimate_voxel_size(256 , max_x, max_y, max_z, min_x, min_y, min_z, &voxel_size, &len_x, &len_y, &len_z);
    EXPECT_DOUBLE_EQ(voxel_size, 0.5);
    EXPECT_EQ(len_x, 8);
    EXPECT_EQ(len_y, 8);
    EXPECT_EQ(len_z, 4);
}
