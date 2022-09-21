//
// Created by praktikum on 21.09.22.
//

#ifndef LAS_VEGAS_MAIN_H
#define LAS_VEGAS_MAIN_H

#include "open3d/Open3D.h"
#include "teaser/registration.h"

using namespace open3d;

open3d::geometry::PointCloud readAndPreprocessPointCloud(std::string path);
open3d::geometry::PointCloud computeISSPointCloud(open3d::geometry::PointCloud cloud);
pipelines::registration::Feature computeFPFHs(open3d::geometry::PointCloud iss_cloud);
teaser::PointCloud convertToTeaserCloud(geometry::PointCloud cloud);
Eigen::Matrix<double, 3, Eigen::Dynamic> convertToEigen(teaser::PointCloud cloud);
void solveTeaserWithCorrespondences(teaser::PointCloud src_cloud, teaser::PointCloud target_cloud, std::vector<std::pair<int, int>> correspondences);
void solveTeaserWithoutCorrespondences(Eigen::Matrix<double, 3, Eigen::Dynamic> src_cloud, Eigen::Matrix<double, 3, Eigen::Dynamic> target_cloud);
void workflowCorrespondences(std::string src_path, std::string target_path);
void workflowDirectlyWithISSAndDownSampling(std::string src_path, std::string target_path);
#endif //LAS_VEGAS_MAIN_H
