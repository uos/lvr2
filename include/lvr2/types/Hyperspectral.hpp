/**
 * Copyright (c) 2018, University Osnabrück
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

#ifndef __HYPERSPECTRAL_HPP__
#define __HYPERSPECTRAL_HPP__

#include <vector>

#include <opencv2/core.hpp>

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

/**
 * @brief   Struct to hold hyperspectral a hyperspectral panorama
 *          cube and corresponding model parameters to align it 
 *          with a laser scan
 */
struct HyperspectralPanorama
{
    /// Distortion
    Vector3d distortion;

    /// Origin
    Vector3d origin;

    /// Principal point
    Vector2d principal;

    /// Rotation
    Vector3d rotation;

    /// Horizontal field of view
    float   fovh;

    /// Vertical field of view
    float   fovv;

    /// Min wavelength in nm, i.e., wavelength of the image 
    /// in the first channel
    float   wmin;

    /// Maximum wavelength, i.e., wavelangth of the image in 
    /// the last channel
    float   wmax;

    /// Vector of intensity (greyscale) images, one for each
    /// channel
    std::vector<cv::Mat> channels;

    HyperspectralPanorama() : fovv(0.0f), wmin(0.0f), wmax(0.0f) {}
};

} // namespace lvr2

#endif