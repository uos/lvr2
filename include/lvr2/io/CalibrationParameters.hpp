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

#ifndef CALIBRATIONPARAMETERS_HPP
#define CALIBRATIONPARAMETERS_HPP

namespace lvr2
{

typedef struct HyperspectralCalibration_
{
    HyperspectralCalibration_() :
        a0(0.0f), a1(0.0f), a2(0.0f),
        angle_x(0.0f), angle_y(0.0f), angle_z(0.0f),
        origin_x(0.0f), origin_y(0.0f), origin_z(0.0f),
        principal_x(0.0f), principal_y(0.0f) {}

    // 1st degree (linear) vertical distortion (aka. aspect ratio correction)
    float a0;

    // 2nd degree vertical distortion
    float a1;

    // 4th degree vertical distortion
    float a2;

    // Rotation around x axis
    float angle_x;

    // Rotation around y axis
    float angle_y;

    // Rotation around z axis
    float angle_z;

    // Translation from camera origin in x direction
    float origin_x;

    // Translation from camera origin in y direction
    float origin_y;

    // Translation from camera origin in z direction
    float origin_z;

    // Vertical offset of the camera image center
    float principal_y;

    // Horizontal offset of the camera image senter
    float principal_x;
} HyperspectralCalibration;

} // namespace lvr2

#endif // CALIBRATIONPARAMETERS_HPP
