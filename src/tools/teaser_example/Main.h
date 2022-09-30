//
// Created by praktikum on 21.09.22.
//

#ifndef LAS_VEGAS_MAIN_H
#define LAS_VEGAS_MAIN_H

#include "open3d/Open3D.h"
#include "teaser/registration.h"

using namespace open3d;
/**
    * @brief Read and process the Open3D pointcloud from ply
    *
    * @param path to ply to read the cloud from
    * @return the Open3D Pointcloud
    */
open3d::geometry::PointCloud readAndPreprocessPointCloud(std::string path);
/**
    * @brief Computes ISS for given Pointcloud (not necessary for teaser registration)
    *
    * @param cloud to compute the ISS Features on
    * @return ISS-reduced pointcloud
    */
open3d::geometry::PointCloud computeISSPointCloud(open3d::geometry::PointCloud cloud);
/**
    * @brief Computes FPFH features for given Pointcloud
    *
    * @param cloud
    * @return 33-dimensional FPFH Feature Vector
    */
pipelines::registration::Feature computeFPFHs(open3d::geometry::PointCloud cloud);
/**
    * @brief converts Open3D::geometry::PointCloud to teaser::PointCloud
    *
    * @param open3d cloud to be converted
    * @return teaser cloud
    */
teaser::PointCloud convertToTeaserCloud(geometry::PointCloud cloud);
/**
    * @brief converts teaser pointcloud to Matrix (not necessary for teaser registration)
    *
    * @param cloud
    * @return returns converted Matrix
    */
Eigen::Matrix<double, 3, Eigen::Dynamic> convertToEigen(teaser::PointCloud cloud);
/**
    * @brief Computes the Correspondences and writes them to a meta file inside the given project file
    *
    * @param src_cloud to get the transformation to
    * @param target_cloud to register
    * @param initial correspondences
    */
void solveTeaserWithCorrespondences(teaser::PointCloud src_cloud, teaser::PointCloud target_cloud, std::vector<std::pair<int, int>> correspondences);
/**
    * @brief solves registration without correspondences (only for small amount of points)
    *
    * @param src_cloud as a matrix
    * @param target_cloud as a matrix
    */
void solveTeaserWithoutCorrespondences(Eigen::Matrix<double, 3, Eigen::Dynamic> src_cloud, Eigen::Matrix<double, 3, Eigen::Dynamic> target_cloud);
/**
    * @brief for a two given scans of a scan project it automatically saves a registration matrix of the second as a meta file
    * @param src_path to get the transformation to
    * @param target_path cloud to register
    * @param voxel_size for the VoxelDownSample reduction, higher value -> stronger reduction (Values between 0.4
    * and 1.5 are recommended)
    */
void teaserRegistration(std::string src_path, std::string target_path, double voxel_size = 1.0);
#endif //LAS_VEGAS_MAIN_H
