//
// Created by praktikum on 20.09.22.
//

#ifndef LAS_VEGAS_MYMATCHING_H
#define LAS_VEGAS_MYMATCHING_H

#include "open3d/Open3D.h"
#include "open3d/pipelines/registration/Feature.h"
/**
    * @brief Builds Correspondences for to sets of FPFH-features
    *
    * @param src_features, the indices of these features are first in the correspondences pair
    * @param dst_features, the indices of these features are second in the correspondences pair
    * @return
    */
std::vector<std::pair<int, int>> MyMatching(
        open3d::pipelines::registration::Feature& src_features, const open3d::pipelines::registration::Feature& dst_features);

#endif //LAS_VEGAS_MYMATCHING_H
