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

TEST(VoxelTests, MetricToVoxelSpace) {
    double point[3] = {0.0, 0.0, 0.0};
    double voxel_size = 1.0;
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    unsigned int voxel_x, voxel_y, voxel_z;
    metric_to_voxel_space(point, voxel_size, len_x, len_y, len_z, &voxel_x, &voxel_y, &voxel_z);
    EXPECT_EQ(voxel_x, 2);
    EXPECT_EQ(voxel_y, 1);
    EXPECT_EQ(voxel_z, 0);
}

TEST(VoxelTests, MetricToVoxelSpace2) {
    double point[3] = {0.0, 1.0, 0.0};
    double voxel_size = 1.0;
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    unsigned int voxel_x, voxel_y, voxel_z;
    metric_to_voxel_space(point, voxel_size, len_x, len_y, len_z, &voxel_x, &voxel_y, &voxel_z);
    EXPECT_EQ(voxel_x, 2);
    EXPECT_EQ(voxel_y, 2);
    EXPECT_EQ(voxel_z, 0);
}

TEST(VoxelTests, MetricToVoxelSpace3) {
    double point[3] = {0.0, 1.49999, 0.0};
    double voxel_size = 1.0;
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    unsigned int voxel_x, voxel_y, voxel_z;
    metric_to_voxel_space(point, voxel_size, len_x, len_y, len_z, &voxel_x, &voxel_y, &voxel_z);
    EXPECT_EQ(voxel_x, 2);
    EXPECT_EQ(voxel_y, 2);
    EXPECT_EQ(voxel_z, 0);
}

TEST(VoxelTests, VoxelToMetricSpace) {
    unsigned int voxel_x = 2;
    unsigned int voxel_y = 1;
    unsigned int voxel_z = 0;
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    double voxel_size = 1.0;
    double point[3];
    voxel_to_metric_space(voxel_x, voxel_y, voxel_z, len_x, len_y, len_z, voxel_size, point);
    EXPECT_DOUBLE_EQ(point[0], 0.0);
    EXPECT_DOUBLE_EQ(point[1], 0.0);
    EXPECT_DOUBLE_EQ(point[2], 0.0);
}

TEST(VoxelTests, VoxelToMetricSpace2) {
    unsigned int voxel_x = 2;
    unsigned int voxel_y = 2;
    unsigned int voxel_z = 0;
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    double voxel_size = 1.0;
    double point[3];
    voxel_to_metric_space(voxel_x, voxel_y, voxel_z, len_x, len_y, len_z, voxel_size, point);
    EXPECT_DOUBLE_EQ(point[0], 0.0);
    EXPECT_DOUBLE_EQ(point[1], 1.0);
    EXPECT_DOUBLE_EQ(point[2], 0.0);
}

TEST(VoxelTests, VoxelToMetricSpace3) {
    unsigned int voxel_x = 2;
    unsigned int voxel_y = 2;
    unsigned int voxel_z = 1;
    int len_x = 5;
    int len_y = 3;
    int len_z = 2;
    double voxel_size = 1.0;
    double point[3];
    voxel_to_metric_space(voxel_x, voxel_y, voxel_z, len_x, len_y, len_z, voxel_size, point);
    EXPECT_DOUBLE_EQ(point[0], 0.0);
    EXPECT_DOUBLE_EQ(point[1], 1.0);
    EXPECT_DOUBLE_EQ(point[2], 0.5);
}

TEST(VoxelTests, GetNeighborIndexZPos) {
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    unsigned long index = 7;
    unsigned long neighbor_index = get_neighbor_index(index, len_x, len_y, len_z, Z_POS);
    EXPECT_EQ(neighbor_index, 22);
}

TEST(VoxelTests, GetNeighborIndexYPos) {
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    unsigned long index = 7;
    unsigned long neighbor_index = get_neighbor_index(index, len_x, len_y, len_z, Y_POS);
    EXPECT_EQ(neighbor_index, 12);
}

TEST(VoxelTests, GetNeighborIndexXPos) {
    int len_x = 5;
    int len_y = 3;
    int len_z = 1;
    unsigned long index = 7;
    unsigned long neighbor_index = get_neighbor_index(index, len_x, len_y, len_z, X_POS);
    EXPECT_EQ(neighbor_index, 8);
}
