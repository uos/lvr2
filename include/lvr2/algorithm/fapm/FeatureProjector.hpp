/**
 * Copyright (c) 2019, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * FeatureProjector.hpp
 *
 *  Created on: 11.02.2018
 *      Author: Alexander Mock
 */

#ifndef LVR2_FAPM_FEATURE_PROJECTOR_HPP
#define LVR2_FAPM_FEATURE_PROJECTOR_HPP

#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/types/MatrixTypes.hpp"

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace lvr2 {

class FeatureProjector {
public:
    FeatureProjector(
        lvr2::RaycasterBasePtr raycaster,
        cv::Ptr<cv::Feature2D> feature
    );

    FeatureProjector(
        lvr2::RaycasterBasePtr raycaster,
        cv::Ptr<cv::Feature2D> feature_detector,
        cv::Ptr<cv::Feature2D> feature_descriptor
    );

    void setDetector(cv::Ptr<cv::Feature2D> feature_detector);
    void setDescriptor(cv::Ptr<cv::Feature2D> feature_descriptor);

    void projectPixels(
        const std::vector<cv::Point2f>& pixels,
        const Extrinsicsd& T,
        const Intrinsicsd& M,
        std::vector<uint8_t>& hits,
        std::vector<lvr2::Vector3f>& intersections
    );

private:

    lvr2::RaycasterBasePtr m_raycaster;


    cv::Ptr<cv::Feature2D> m_feature_detector;
    cv::Ptr<cv::Feature2D> m_feature_descriptor;
};

using FeatureProjectorPtr = std::shared_ptr<FeatureProjector>;

} // namespace lvr2

#endif // LVR2_FAPM_FEATURE_PROJECTOR_HPP