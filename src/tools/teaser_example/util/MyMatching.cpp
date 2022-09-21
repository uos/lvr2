//
// Created by praktikum on 20.09.22.
//

#include "MyMatching.h"
#include <vector>
#include "open3d/Open3D.h"
#include "open3d/pipelines/registration/Feature.h"
//#include "open3d/pipelines/registration/FastGlobalRegistration.h"
#include "open3d/geometry/KDTreeFlann.h"

using namespace open3d;

std::vector<std::pair<int, int>> MyMatching(
        pipelines::registration::Feature& src_features, const pipelines::registration::Feature& dst_features) {
    geometry::KDTreeFlann src_tree(src_features);
    geometry::KDTreeFlann dst_tree(dst_features);
    std::cout << "col length: " << dst_features.data_.col(0).size() << std::endl;

    std::cout << "row length: " << dst_features.data_.row(0).size() << std::endl;
    std::vector<std::pair<int, int>> correspondo;
    std::map<int, int> correspondo_map;

    std::vector<int> tmp_index(1);
    std::vector<double> tmp_distance2(1);

    std::cout << "forschleife: " << std::endl;
    for (int i = 0; i < dst_features.data_.rows(); i++) {
        src_tree.SearchKNN(Eigen::VectorXd(dst_features.data_.col(i)), 1, tmp_index, tmp_distance2);
        if(i < 10) {
           std::cout << "i: " << i << ", tmp_index: " << tmp_index[0] << std::endl;
        }

        //insert the nearest neighbor from the i-th destination keypoint to a vector and a map
        const std::pair<int, int> p(i, tmp_index[0]);
        correspondo.push_back(p);
        correspondo_map[tmp_index[0]] = i;
    }
    std::vector<std::pair<int, int>> final_correspondo;
    //find the nearest neighbor for the i-th source keypoint
    for (int j = 0; j < src_features.data_.rows(); j++) {
        dst_tree.SearchKNN(Eigen::VectorXd(src_features.data_.col(j)), 1, tmp_index, tmp_distance2);
        if(j < 10) {
            std::cout << "i: " << j << ", tmp_index: " << tmp_index[0] << std::endl;
        }
        //write the indeces to the final correspondences only if the correspondences are the same from src to dst
        //and visa versa
        const std::pair<int, int> p(tmp_index[0], j);
        if(correspondo_map[j] == tmp_index[0]){
            final_correspondo.push_back(p);
        }


    }

    std::cout << "Pair vector: " << std::endl;

    for (int i = 0; i < 10; i++) {
        std::cout << "Pair " << i << ": " << correspondo[i].first << ", " << correspondo[i].second << std::endl;
    }

    for (int i = 0; i < 10; i++) {
        std::cout << "Pair " << i << ": " << final_correspondo[i].first << ", " << correspondo[i].second << std::endl;
    }
//    std::cout << "Index: " << index[0] << std::endl;
//    std::cout << "distance2: " << distance2[0] << std::endl;

    return correspondo;
// 1536768432

}
