//
// Created by praktikum on 13.09.22.
//

#include <chrono>
#include <iostream>
#include <random>
#include "util/pointtovertex.h"
#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>

#define NOISE_BOUND 0.05
int main() {
    // Load the .ply file, read source and target PLY-files to teaser pointclouds

    //change from point to vertex header
    std::string source_path = "/home/praktikum/robopraktikum/Datapraktikum/data/001/scans/220720_095525.ply";
    std::string target_path = "/home/praktikum/robopraktikum/Datapraktikum/data/002/scans/220720_100202.ply";
    std::string source_vertex_path = "/home/praktikum/robopraktikum/Datapraktikum/data/001/scans/220720_095525_vertex.ply";
    std::string target_vertex_path = "/home/praktikum/robopraktikum/Datapraktikum/data/002/scans/220720_100202_vertex.ply";

    std::cout << "Convert ply header for tiniply.." << std::endl;
    int stat1 = pointtovertex(source_path, source_vertex_path);
    int stat2 = pointtovertex(target_path, target_vertex_path);

    teaser::PLYReader reader1;
    teaser::PointCloud src_cloud;
    std::cout << "Reading first File" << std::endl;
    auto status1 = reader1.read(source_vertex_path, src_cloud);
    int N_src = src_cloud.size();
    std::cout << "Reading second File" << std::endl;
    teaser::PLYReader reader2;
    teaser::PointCloud target_cloud;
    auto status2 = reader2.read(target_vertex_path, target_cloud);
    int N_target = target_cloud.size();


    std::cout << "Writing source cloud into eigen Matrix." << std::endl;
    // Convert the source point cloud to Eigen
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N_src);
    for (size_t i = 0; i < N_src; ++i) {
        src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
    }

    std::cout << "Writing target cloud into eigen Matrix." << std::endl;
    // Convert the target point cloud to Eigen
    Eigen::Matrix<double, 3, Eigen::Dynamic> target(3, N_target);
    for (size_t i = 0; i < N_target; ++i) {
        target.col(i) << target_cloud[i].x, target_cloud[i].y, target_cloud[i].z;
    }

    std::cout << "Source Matrix Head: " << src.col(1) << "||"<< src.col(2) << std::endl;
    std::cout << "Target Matrix Head: " << target.col(1) << "||" << target.col(2)<< std::endl;
    std::cout << "Source Matrix colum size : " << src.cols() << std::endl;
    std::cout << "Target Matrix colum size: " << target.cols() << std::endl;

    //solve the Registrations problem without solving for scale, because scans are assumed to have same scale
    teaser::RobustRegistrationSolver::Params params;
    teaser::RobustRegistrationSolver solver(params);
    params.noise_bound = NOISE_BOUND;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    std::cout << "Solve registration task." << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    solver.solve(src,target);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto solution = solver.getSolution();
    // Compare results
    std::cout << "=====================================" << std::endl;
    std::cout << "          TEASER++ Results           " << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Estimated rotation: " << std::endl;
    std::cout << solution.rotation << std::endl;
    std::cout << "Estimated translation: " << std::endl;
    std::cout << solution.translation << std::endl;

}