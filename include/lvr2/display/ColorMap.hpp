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

 /**
 * ColorMap.h
 *
 *  @date 30.08.2011
 *  @author Thomas Wiemann
 */

#ifndef COLORMAP_H_
#define COLORMAP_H_

#include <iostream>

/***
 * @brief Class to generate and handle color gradients
 *
 * @TODO: Integrated from show.
 */

namespace lvr2
{

/// Identifies a color gradient
enum GradientType
{
    SOLID = 0,
    GREY = 1,
    HSV = 2,
    JET = 3,
    HOT = 4,
    SHSV = 5,
    SIMPSONS = 6
};

class ColorMap
{
public:

    /**
     * @brief Ctor. Constructs a color gradient with the given number
     *       of buckets
     *
     * @param buckets   Number of colors in current gradient
     */
    ColorMap(size_t buckets) : m_numBuckets(buckets) {}

    /**
     * @brief Dtor.
     */
    virtual ~ColorMap() {};

    /**
     * @brief Returns three float values for the color of the given bucket
     *
     * @param color     The three color components
     * @param bucket    The bucket index
     * @param gradient  The type of gradient (default grey)
     */
    void getColor(float* color, size_t bucket, GradientType gradient = GREY);

private:

    /**
     * @brief Converts the given color in HSV space into RGB space
     *
     * @param hue       Hue component of input color
     * @param s         Saturation component of input color
     * @param v         Value component of input color
     * @param r         Converted red component
     * @param g         Converted green component
     * @param b         Converted blue component
     */
    void convertHSVToRGB(float hue, float s, float v, float &r, float &g, float &b);

    /**
     * @brief Returns a color from a gray gradient
     *
     * @param color     The three color components
     * @param bucket    The bucket index
     */
    void calcColorGrey(float* color, size_t bucket);

    /**
     * @brief Returns a color from a HSV gradient
     *
     * @param color     The three color components
     * @param bucket    The bucket index
     */
    void calcColorHSV(float* color, size_t bucket);

    /**
     * @brief Returns a color from a Jet gradient
     *
     * @param color     The three color components
     * @param bucket    The bucket index
     */
    void calcColorJet(float* color, size_t bucket);

    /**
     * @brief Returns a color from a hot gradient
     *
     * @param color     The three color components
     * @param bucket    The bucket index
     */
    void calcColorHot(float* color, size_t bucket);

    /**
     * @brief Returns a color from a SHSV gradient
     *
     * @param color     The three color components
     * @param bucket    The bucket index
     */
    void calcColorSHSV(float* color, size_t bucket);

    /**
         * @brief Returns a color from a Simpsons gradient
         *
         * @param color     The three color components
         * @param bucket    The bucket index
         */
    void calcColorSimpsons(float* color, size_t bucket);

    /// Number of colors in the color gradient
    size_t      m_numBuckets;
};

} // namespace lvr2

#endif /* COLORMAP_H_ */
