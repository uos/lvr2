/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
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

namespace lssr
{

/// Identifies a color gradient
enum GradientType
{
    SOLID = 0,
    GREY = 1,
    HSV = 2,
    JET = 3,
    HOT = 4,
    SHSV = 5
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


    /// Number of colors in the color gradient
    size_t      m_numBuckets;
};

} // namespace lssr

#endif /* COLORMAP_H_ */
