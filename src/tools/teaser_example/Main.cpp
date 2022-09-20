//
// Created by praktikum on 13.09.22.
//

#include <chrono>
#include <iostream>
#include <random>
#include "util/pointtovertex.h"
#include <Eigen/Core>

//#include <teaser/ply_io.h>
//#include <teaser/registration.h>
//#include <teaser/matcher.h>
#include "open3d/Open3D.h"
#include "util/MyMatching.h"

#define NOISE_BOUND 0.05
int main() {
    // file paths
//    std::string source_path = "/home/praktikum/robopraktikum/scans_test/001/scans/220720_095525.ply";
//    std::string target_path = "/home/praktikum/robopraktikum/scans_test/001/scans/2207_vertex.ply";
//    std::string source_path_vertex = "/home/praktikum/robopraktikum/scans_test/001/scans/220720_095525_vertex.ply";
//    std::string target_path_vertex = "/home/praktikum/robopraktikum/scans_test/001/scans/2207_vertex.ply";
    std::string reduced_source_path = "/home/praktikum/robopraktikum/scans_test/001/scans/reduced.ply";
    std::string reduced_target_path = "/home/praktikum/robopraktikum/scans_test/002/scans/reduced.ply";
//    std::string reduced_source_path_vertex = "/home/praktikum/robopraktikum/scans_test/001/scans/220720_095525_vertex.ply";
//    std::string reduced_target_path_vertex = "/home/praktikum/robopraktikum/scans_test/001/scans/2207_vertex.ply";
//    std::cout << "Convert ply header for Open3D (point to vertex)" << std::endl;
//    int stat1 = pointtovertex(source_path, source_path_vertex);

    open3d::geometry::PointCloud src_cloud;
    open3d::geometry::PointCloud target_cloud;
//    teaser::PointCloud src_cloud;

    // Reading PointClouds
    std::cout << "Reading PointCloud... ";
    auto search_param = open3d::geometry::KDTreeSearchParamKNN(30);
    open3d::io::ReadPointCloud(reduced_source_path, src_cloud);
    open3d::io::ReadPointCloud(reduced_target_path, target_cloud);
    std::cout << "- done." << std::endl;

    // Estimating Normals
    std::cout << "Estimating Normals... ";
    src_cloud.EstimateNormals(search_param);
    target_cloud.EstimateNormals(search_param);
    std::cout << "- done." << std::endl;

    // output size
//    src_cloud.Get


    std::cout << "Source cloud Size: " << src_cloud.points_.size() << std::endl;

    // ISS
    std::cout << "Computing ISS keypoints... ";
    open3d::geometry::PointCloud src_iss_cloud = *open3d::geometry::keypoint::ComputeISSKeypoints(src_cloud);
    open3d::geometry::PointCloud target_iss_cloud = *open3d::geometry::keypoint::ComputeISSKeypoints(target_cloud);
    std::cout << "- done." << std::endl;

    std::cout << "Source cloud ISS Dimension: " << src_iss_cloud.points_.size() << std::endl;
//    open3d::pipelines::registration::FastGlobalRegistration::AdvancedMatching(src_cloud, target_cloud,)

    // Compute Features
    std::cout << "Computing FPFHs... ";
    auto src_feature =  *open3d::pipelines::registration::ComputeFPFHFeature(src_iss_cloud, search_param);
    auto target_feature =  *open3d::pipelines::registration::ComputeFPFHFeature(target_iss_cloud, search_param);
    std::cout << "- done." << std::endl;

    // print feature information
    std::cout << "Features: " << std::endl;
    std::cout <<  "Dimensions: " << src_feature.Dimension() << std::endl;
    std::cout <<  "Number: " << src_feature.Num() << std::endl;

    for (int i = 0; i < 40; ++i) {
        std::cout << src_feature.data_(0, i) << "; ";
    }
    std::cout << std::endl;

    MyMatching(target_feature, src_feature);

//    std::cout << "Computing FPFHs... ";
//    // test own matching
//    util::AdvancedMatching(src_iss_cloud,
//                     target_iss_cloud,
//                     util::InitialMatching(src_feature, target_feature),
//                     pipelines::registration::FastGlobalRegistrationOption(
//            /* decrease_mu =  */ 1.4, true,
//                                 true, 0.75,
//                                 64,
//            /* tuple_scale =  */ 0.95,
//                                 1000));
//    std::cout << "- done." << std::endl;


    // compute correspondences with teaser
//    teaser::Matcher matcher;
//    auto correspondences = matcher.calculateCorrespondences(
//            src_iss_cloud, target_iss_cloud, src_feature, target_feature);

    // Prepare solver parameters
//    teaser::RobustRegistrationSolver::Params params;
//    params.noise_bound = NOISE_BOUND;
//    params.cbar2 = 1;
//    params.estimate_scaling = false;
//    params.rotation_max_iterations = 100;
//    params.rotation_gnc_factor = 1.4;
//    params.rotation_estimation_algorithm =
//            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
//    params.rotation_cost_threshold = 0.005;
//
//    // Solve with TEASER++
//    teaser::RobustRegistrationSolver solver(params);
//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//    solver.solve(src_iss_cloud, target_iss_cloud, correspondences);
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//
//    auto solution = solver.getSolution();




//
//    auto solution = solver.getSolution();
//
//    // Compare results
//    std::cout << "=====================================" << std::endl;
//    std::cout << "          TEASER++ Results           " << std::endl;
//    std::cout << "=====================================" << std::endl;
//    std::cout << "Expected rotation: " << std::endl;
//    std::cout << T.topLeftCorner(3, 3) << std::endl;
//    std::cout << "Estimated rotation: " << std::endl;
//    std::cout << solution.rotation << std::endl;
//    std::cout << "Error (deg): " << getAngularError(T.topLeftCorner(3, 3), solution.rotation)
//              << std::endl;
//    std::cout << std::endl;
//    std::cout << "Expected translation: " << std::endl;
//    std::cout << T.topRightCorner(3, 1) << std::endl;
//    std::cout << "Estimated translation: " << std::endl;
//    std::cout << solution.translation << std::endl;
//    std::cout << "Error (m): " << (T.topRightCorner(3, 1) - solution.translation).norm() << std::endl;
//    std::cout << std::endl;
//    std::cout << "Number of correspondences: " << N << std::endl;
//    std::cout << "Number of outliers: " << N_OUTLIERS << std::endl;
//    std::cout << "Time taken (s): "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
//                 1000000.0
//              << std::endl;

}