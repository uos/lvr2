//
// Created by Gruppe teaser and IO on 13.09.22.
//

#include <chrono>
#include <iostream>
#include <random>
#include "util/pointtovertex.h"
#include <Eigen/Core>
#include <filesystem>
#include <teaser/registration.h>
#include "open3d/Open3D.h"
#include "lvr2/util/TransformUtils.hpp"
#include "util/MyMatching.h"

#define NOISE_BOUND 0.05

using namespace open3d;
namespace fs = std::filesystem;

auto search_param = open3d::geometry::KDTreeSearchParamKNN(30);

class transformRegistration;
int count=1;
Eigen::Matrix<float_t ,4,4>  OldTransformation;




open3d::geometry::PointCloud readAndPreprocessPointCloud(std::string path) {
    // Reading PointClouds
    geometry::PointCloud cloud;
    std::cout << "Reading PointCloud... " << std::flush;
    open3d::io::ReadPointCloud(path, cloud);
    std::cout << "- done." << std::endl;

    std::cout << "Cloud Size: " << cloud.points_.size() << std::endl;

    return cloud;
}

open3d::geometry::PointCloud computeISSPointClouds(open3d::geometry::PointCloud cloud)
{
    // ISS
    std::cout << "Computing ISS keypoints... " << std::endl;
    double SALIENT_RADIUS = 0;
    double NON_MAX_RADIUS = 0;
    double GAMMA_21 = 0.975;
    double GAMMA_32 = 0.975;
    int MIN_NEIGHBORS = 5;
    open3d::geometry::PointCloud iss_cloud = *open3d::geometry::keypoint::ComputeISSKeypoints(cloud,

                                                                                              SALIENT_RADIUS, NON_MAX_RADIUS, GAMMA_21, GAMMA_32, MIN_NEIGHBORS);
    std::cout << "- done." << std::endl;
    std::cout << "ISS Keypoints size: " << iss_cloud.points_.size() << std::endl;
    return iss_cloud;
}

pipelines::registration::Feature computeFPFHs(open3d::geometry::PointCloud cloud) {
        // Compute Features
        std::cout << "Computing FPFHs... " << std::flush;
        auto feature =  *pipelines::registration::ComputeFPFHFeature(cloud, search_param);
        std::cout << "- done." << std::endl;

        // print feature information
        std::cout <<  "Number of features: " << feature.Num() << std::endl;

        return feature;
}

teaser::PointCloud convertToTeaserCloud(geometry::PointCloud cloud) {
    // convert Open3D cloud i
    teaser::PointCloud teaser_cloud;
    for (size_t i = 0; i < cloud.points_.size(); ++i) {
        teaser_cloud.push_back(
                {static_cast<float>(cloud.points_[i].x()), static_cast<float>(cloud.points_[i].y()),
                 static_cast<float>(cloud.points_[i].z())});
    }
    return teaser_cloud;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> convertToEigen(teaser::PointCloud cloud) {
    std::cout << "Converting to eigen3 Matrix... " << std::flush;
    int N = cloud.size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen(3, N);
    for (size_t i = 0; i < cloud.size(); ++i) {
        eigen.col(i) << cloud[i].x, cloud[i].y, cloud[i].z;
    }
    std::cout << "- done." << std::endl;
    return eigen;
}

void solveTeaserWithoutCorrespondences(Eigen::Matrix<double, 3, Eigen::Dynamic> src_cloud, Eigen::Matrix<double, 3, Eigen::Dynamic> target_cloud) {
    // Prepare solver parameters
    std::cout << "Teaser solving... " << std::flush;
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
    solver.solve(src_cloud, target_cloud);
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
    std::cout << "Time taken (s): "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                 1000000.0
                 << std::endl;
}

void solveTeaserWithCorrespondences(teaser::PointCloud src_cloud, teaser::PointCloud target_cloud, std::vector<std::pair<int, int>> correspondences) {
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
    solver.solve(target_cloud, src_cloud, correspondences);
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
    std::cout << "Time taken (s): "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                 1000000.0
              << std::endl;

    //Save the Results of the teaser Solver to Metadata
    Eigen::Matrix<float_t ,4,4> transformation;

    for (int i=0;i<=3;i++){
        for (int j=0;j<=3;j++){
            if(i<3){
                if(j<3){
                    transformation(i,j)=solution.rotation(i,j);
                }
                else{
                    transformation(i,j)=solution.translation(i);
                }
            }
            else{
                transformation(i,j)=0;
            }

        }
    }
    transformation(3,3) = 1;


    OldTransformation=OldTransformation*transformation;
    std::ofstream file;
    stringstream s;
    s <<"/home/praktikum/Desktop/teaserOpen3dply/" <<count<< ".meta";
    file.open(s.str());
    file <<"Globalen Transformation"<<std::endl;
    file <<OldTransformation<< std::endl;
    file <<"Lokale Translation"<<std::endl;
    file << solution.translation << std::endl;
    file <<"Lokale rotation"<<std::endl;
    file<< solution.rotation<< std::endl;
    file.close();
    std::cout << "Scan"<<count<< std::endl;
    count++;

}


void teaserRegistration(std::string src_path, std::string target_path, double voxel_size = 1.0) {

    geometry::PointCloud src_cloud = readAndPreprocessPointCloud(src_path);
    geometry::PointCloud target_cloud = readAndPreprocessPointCloud(target_path);

    geometry::PointCloud sampled_src_cloud = *src_cloud.VoxelDownSample(voxel_size);
    geometry::PointCloud sampled_target_cloud = *target_cloud.VoxelDownSample(voxel_size);
    std::cout << "Voxel source size: " << sampled_src_cloud.points_.size() << std::endl;
    std::cout << "Voxel target size: " << sampled_target_cloud.points_.size() << std::endl;

    sampled_src_cloud.EstimateNormals();
    sampled_target_cloud.EstimateNormals();

    auto src_feature = computeFPFHs(sampled_src_cloud);
    auto target_feature = computeFPFHs(sampled_target_cloud);

    std::vector<std::pair<int, int>> correspondences = MyMatching(target_feature, src_feature);

    // convert Open3D cloud i
    teaser::PointCloud src_teaser_cloud = convertToTeaserCloud(sampled_src_cloud);
    teaser::PointCloud target_teaser_cloud = convertToTeaserCloud(sampled_target_cloud);

    solveTeaserWithCorrespondences(src_teaser_cloud, target_teaser_cloud, correspondences);
}

int main() {

    //voxelsize for the voxelreduktion, higher value -> stronger reduction
    const double voxelsize = 0.5;

    //File paths for Scanproject
    std::string robo_dir = "/home/praktikum/Desktop/Schematest/raw";
    std::string matrix_dir = "/home/praktikum/Desktop/teaserOpen3d/ply";

    std::string end = ".ply";
    std::string vertex = "_vertex";

    //Magicnumber for the amount of scans in the scanproject
    int numberOfScans = 11; //TODO: Read the Path and count the number of scans

    for (int i = 1; i <= numberOfScans; ++i) {
        std::stringstream tmp_stream;
        tmp_stream << robo_dir << "/" << std::setfill('0') << std::setw(8) << i << "/lidar_00000000/00000000/points.ply";
        std::string source_path_vertex = tmp_stream.str();
        std::stringstream tmp_stream2;
        tmp_stream2 << "/home/praktikum/Desktop/teaserOpen3d/ply/" << std::setfill('0') << std::setw(3) << i << ".ply";
        std::string dest_path = tmp_stream2.str();
        pointtovertex(source_path_vertex, dest_path);
    }

    OldTransformation<< 1,0,0,0,
                        0,1,0,0,
                        0,0,1,0,
                        0,0,0,1;



    for (int i = 1; i <= numberOfScans; ++i) {
        if(i < numberOfScans)
        {
            std::stringstream tmp_stream;
            tmp_stream << matrix_dir << "/" << std::setfill('0') << std::setw(3) << i << ".ply";
            std::stringstream tmp_stream2;
            tmp_stream2 << matrix_dir << "/" << std::setfill('0') << std::setw(3) << i+1 << ".ply";
            std::string src_path = tmp_stream.str();
            std::string target_path = tmp_stream2.str();
            teaserRegistration(src_path, target_path, voxelsize);
        }
        else if(i == numberOfScans)
        {
            //Skips the last scan since it cant be matched any further for our purposes (may need to be changed to match
            //the last scan with the first one
            break;
        }
    }
}
