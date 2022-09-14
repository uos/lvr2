//
// Created by praktikum on 13.09.22.
//

#include <chrono>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>

int main() {
    // Load the .ply file
    teaser::PLYReader reader;
    teaser::PointCloud src_cloud;
    auto status1 = reader.read("/home/praktikum/Datapraktikum/data/sample_data/220720_095525_vertex.ply", src_cloud);
    int N_src = src_cloud.size();

    teaser::PLYReader reader;
    teaser::PointCloud target_cloud;
    auto status2 = reader.read("/home/praktikum/Datapraktikum/data/sample_data/220720_100202_vertex.ply", target_cloud);
    int N_target = target_cloud.size();

    // Convert the point cloud to Eigen
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N_src);
    for (size_t i = 0; i < N_src; ++i) {
        src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
    }

    // Convert the point cloud to Eigen
    Eigen::Matrix<double, 3, Eigen::Dynamic> target(3, N_target);
    for (size_t i = 0; i < N_target; ++i) {
        src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
    }

    teaser::RobustRegistrationSolver::Params params;

    teaser::RobustRegistrationSolver solver(params);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    solver.solve(src, target);
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