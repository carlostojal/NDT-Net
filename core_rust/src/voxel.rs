/*
 MIT License

 Copyright (c) 2024 Carlos Cabaço Tojal

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 */

use nalgebra::Vector3;

/// Voxel
/// 
/// The voxel module contains methods for voxel estimation, mapping and manipulation.
/// 

/// Estimates the voxel size of a point cloud.
/// 
/// Arguments
/// 
/// * `num_desired_voxels` - The number of desired voxels.
/// * `max` - The maximum point of the point cloud.
/// * `min` - The minimum point of the point cloud.
/// * `offsets` - The offsets of the point cloud.
/// 
/// Returns
/// 
/// * `f64` - The voxel size.
/// * `Vector3<u32>` - The number of voxels in each dimension.
pub fn estimate_voxel_size(num_desired_voxels: u64,
                            max: Vector3<f64>,
                            min: Vector3<f64>) -> (f64, Vector3<u32>) {

    // calculate the decimal dimensions of the point cloud
    let dims = max - min;

    // calculate the voxel size
    // multiply the dimensions and divide by the number of desired voxels
    let voxel_size = (dims.x * dims.y * dims.z) / num_desired_voxels as f64;

    // calculate the number of voxels in each dimension
    let num_voxels = Vector3::new((dims.x / voxel_size).ceil() as u32,
                                   (dims.y / voxel_size).ceil() as u32,
                                   (dims.z / voxel_size).ceil() as u32);
    
    // return the voxel size and the number of voxels
    (voxel_size, num_voxels)
}
