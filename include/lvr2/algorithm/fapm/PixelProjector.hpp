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

#ifndef LVR2_FAPM_PIXEL_PROJECTOR_HPP
#define LVR2_FAPM_PIXEL_PROJECTOR_HPP

#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/ScanTypes.hpp"

// OpenCV includes
#include <opencv2/core.hpp>

namespace lvr2 {

class PixelProjector {
public:
    PixelProjector(
        RaycasterBasePtr raycaster,
        ScanCameraPtr scan_cam
    );

    /**
     * project single pixel
     */
    void project(
        const ScanImagePtr& scan_image,
        const cv::Point2f& pixel,
        uint8_t& hit, // output hit flag
        Vector3f& intersection // output intersections
    ) const;

    /**
     * project multiple pixels
     */
    void project(
        const ScanImagePtr& scan_image,
        const std::vector<cv::Point2f>& pixels,
        std::vector<uint8_t>& hits, // output hit flag
        std::vector<Vector3f>& intersections // output intersections
    ) const;

    /**
     * project all pixels
     */
    void project(
        const ScanImagePtr& scan_image,
        std::vector<std::vector<uint8_t> >& hits, // output hit flag
        std::vector<std::vector<Vector3f> >& intersections // output intersections
    ) const;

private:

    RaycasterBasePtr m_raycaster;
    ScanCameraPtr m_scan_cam;

};

using PixelProjectorPtr = std::shared_ptr<PixelProjector>;

} // namespace lvr2

#endif // LVR2_FAPM_PIXEL_PROJECTOR_HPP