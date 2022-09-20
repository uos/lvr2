//
// Created by praktikum on 13.09.22.
//

#include <chrono>
#include <iostream>
#include <random>
#include "util/pointtovertex.h"
#include <Eigen/Core>

//#include <teaser/ply_io.h>
#include <teaser/registration.h>
//#include <teaser/matcher.h>
#include "open3d/Open3D.h"
#include "util/MyMatching.h"

#define NOISE_BOUND 0.05
int main() {
    // file paths
//    std::string source_path = "/home/praktikum/robopraktikum/scans_test/001/scans/220720_095525.ply";
//    std::string target_path = "/home/praktikum/robopraktikum/scans_test/002/scans/220720_100202.ply";
    std::string source_path_vertex = "/home/praktikum/robopraktikum/scans_test/001/scans/220720_095525_vertex.ply";
//    std::string target_path_vertex = "/home/praktikum/robopraktikum/scans_test/001/scans/220720_100202_vertex.ply";
//    std::string source_path = "/home/praktikum/robopraktikum/scans_test/001/scans/reduced.ply";
//    std::string target_path = "/home/praktikum/robopraktikum/scans_test/002/scans/reduced.ply";
    std::string target_path = "/home/praktikum/robopraktikum/scans_test/003/scans/220720_100819.ply";
    std::string target_path_vertex = "/home/praktikum/robopraktikum/scans_test/003/scans/220720_100819_vertex.ply";

    std::cout << "Convert ply header for Open3D (point to vertex)" << std::endl;
//    int stat1 = pointtovertex(source_path, source_path_vertex);
    int stat2 = pointtovertex(target_path, target_path_vertex);

    open3d::geometry::PointCloud src_cloud;
    open3d::geometry::PointCloud target_cloud;

    // Reading PointClouds
    std::cout << "Reading PointCloud... ";
    auto search_param = open3d::geometry::KDTreeSearchParamKNN(30);
    open3d::io::ReadPointCloud(source_path_vertex, src_cloud);
    open3d::io::ReadPointCloud(target_path_vertex, target_cloud);
    std::cout << "- done." << std::endl;

    // Estimating Normals
    std::cout << "Estimating Normals... ";
    src_cloud.EstimateNormals(search_param);
    target_cloud.EstimateNormals(search_param);
    std::cout << "- done." << std::endl;

    std::cout << "Source cloud Size: " << src_cloud.points_.size() << std::endl;

    // ISS
    std::cout << "Computing ISS keypoints... " << std::endl;
    double SALIENT_RADIUS = 0.005;
    double NON_MAX_RADIUS = 0.005;
    double GAMMA_21 = 0.5;
    double GAMMA_32 = 0.5;
    int MIN_NEIGHBORS = 5;
    open3d::geometry::PointCloud src_iss_cloud = *open3d::geometry::keypoint::ComputeISSKeypoints(src_cloud,

                                            SALIENT_RADIUS, NON_MAX_RADIUS, GAMMA_21, GAMMA_32, MIN_NEIGHBORS);
    open3d::geometry::PointCloud target_iss_cloud = *open3d::geometry::keypoint::ComputeISSKeypoints(target_cloud,
                                                        SALIENT_RADIUS, NON_MAX_RADIUS, GAMMA_21, GAMMA_32, MIN_NEIGHBORS);
    std::cout << "- done." << std::endl;

//    std::cout << "Source cloud ISS Points [0,0]: " << src_iss_cloud.points_[0,0] << std::endl;
//    std::cout << "Source cloud ISS Dimension [0,1]: " << src_iss_cloud.points_[0, 1] << std::endl;
//    std::cout << "Source cloud ISS Dimension [1,0]: " << src_iss_cloud.points_[1, 0] << std::endl;
//    std::cout << "Source cloud ISS Dimension [1,1]: " << src_iss_cloud.points_[1, 1] << std::endl;
//    std::cout << "Source cloud ISS Dimension [1,2]: " << src_iss_cloud.points_[1, 2] << std::endl;
//    std::cout << "Source cloud ISS Dimension [2,1]: " << src_iss_cloud.points_[2, 1] << std::endl;
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


    std::cout << "Computing correspondo... ";
    std::vector<std::pair<int, int>> correspondences = MyMatching(target_feature, src_feature);
    std::cout << "- done." << std::endl;

    // convert Open3D cloud in teaser cloud
    teaser::PointCloud src_teaser_cloud;
    for(size_t i = 0; i < src_iss_cloud.points_.size(); ++i) {
        src_teaser_cloud.push_back({static_cast<float>(src_iss_cloud.points_[i].x()), static_cast<float>(src_iss_cloud.points_[i].y()),
                               static_cast<float>(src_iss_cloud.points_[i].z())});
    }

    teaser::PointCloud target_teaser_cloud;
    for(size_t i = 0; i < target_iss_cloud.points_.size(); ++i) {
        target_teaser_cloud.push_back({static_cast<float>(target_iss_cloud.points_[i].x()), static_cast<float>(target_iss_cloud.points_[i].y()),
                                    static_cast<float>(target_iss_cloud.points_[i].z())});
    }
    // Prepare solver parameters
    std::cout << "Teaser solving... ";
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = NOISE_BOUND;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    solver.solve(src_teaser_cloud, target_teaser_cloud, correspondences);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto solution = solver.getSolution();


//    // Compare results
    std::cout << "=====================================" << std::endl;
    std::cout << "          TEASER++ Results           " << std::endl;
    std::cout << "=====================================" << std::endl;
//    std::cout << "Expected rotation: " << std::endl;
//    std::cout << T.topLeftCorner(3, 3) << std::endl;
    std::cout << "Estimated rotation: " << std::endl;
    std::cout << solution.rotation << std::endl;
//    std::cout << "Error (deg): " << getAngularError(T.topLeftCorner(3, 3), solution.rotation)
//              << std::endl;
//    std::cout << std::endl;
//    std::cout << "Expected translation: " << std::endl;
//    std::cout << T.topRightCorner(3, 1) << std::endl;
    std::cout << "Estimated translation: " << std::endl;
    std::cout << solution.translation << std::endl;
//    std::cout << "Error (m): " << (T.topRightCorner(3, 1) - solution.translation).norm() << std::endl;
//    std::cout << std::endl;
//    std::cout << "Number of correspondences: " << N << std::endl;
//    std::cout << "Number of outliers: " << N_OUTLIERS << std::endl;
    std::cout << "Time taken (s): "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                 1000000.0
              << std::endl;

}