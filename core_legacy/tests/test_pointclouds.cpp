#include "gtest/gtest.h"
#include <ndnet_core/pointclouds.h>
#include <iostream>

TEST(PointCloudTests, TestPointCloudLimits)
{
    double point_cloud[18] = {
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -2.0
    };
    double max_x, max_y, max_z, min_x, min_y, min_z;
    get_pointcloud_limits(point_cloud, 3, 6, &max_x, &max_y, &max_z, &min_x, &min_y, &min_z);
    EXPECT_EQ(max_x, 1.0);
    EXPECT_EQ(max_y, 1.0);
    EXPECT_EQ(max_z, 1.0);
    EXPECT_EQ(min_x, -1.0);
    EXPECT_EQ(min_y, -1.0);
    EXPECT_EQ(min_z, -2.0);
}

TEST(PointCloudTests, TestPointCloudLimits2)
{
    double point_cloud[90] = {
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -2.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -2.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -2.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -2.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -2.0
    };
    double max_x, max_y, max_z, min_x, min_y, min_z;
    get_pointcloud_limits(point_cloud, 3, 6, &max_x, &max_y, &max_z, &min_x, &min_y, &min_z);
    EXPECT_EQ(max_x, 1.0);
    EXPECT_EQ(max_y, 1.0);
    EXPECT_EQ(max_z, 1.0);
    EXPECT_EQ(min_x, -1.0);
    EXPECT_EQ(min_y, -1.0);
    EXPECT_EQ(min_z, -2.0);
}
