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
 * ColorMap.cpp
 *
 *  @date 30.08.2011
 *  @author Thomas Wiemann
 */

#include "lvr2/display/ColorMap.hpp"

#include <cassert>
#include <cmath>

namespace lvr2
{

void ColorMap::getColor(float* color, size_t bucket, GradientType gradient )
{
    assert(color);

    switch(gradient)
    {
    case GREY:
        calcColorGrey(color, bucket);
        break;
    case JET:
        calcColorJet(color, bucket);
        break;
    case HOT:
        calcColorHot(color, bucket);
        break;
    case HSV:
        calcColorHSV(color, bucket);
        break;
    case SHSV:
        calcColorSHSV(color, bucket);
        break;
    case SIMPSONS:
    	calcColorSimpsons(color, bucket);
    	break;
    default:
        color[0] = 1.0;
        color[1] = 1.0;
        color[2] = 1.0;
        break;

    }
}

void ColorMap::calcColorSimpsons(float* color, size_t bucket)
{
	color[0] = fabs( cos( bucket )  );
	color[1] = fabs( sin( bucket * 30 ) );
	color[2] = fabs( sin( bucket * 2 ) );
}

void ColorMap::calcColorGrey(float *d, size_t i)
{
    d[0] = (float)i/(float)m_numBuckets;
    d[1] = (float)i/(float)m_numBuckets;
    d[2] = (float)i/(float)m_numBuckets;
}

void ColorMap::calcColorHSV(float *d, size_t i)
{
    float t = (float)i/(float)m_numBuckets;
    convertHSVToRGB(360.0*t, 1.0, 1.0,  d[0], d[1], d[2]);

}

void ColorMap::calcColorSHSV(float *d, size_t i)
{
    float t = (float)i/(float)m_numBuckets;
    convertHSVToRGB(360.0*t, 1.0, 1.0,  d[0], d[1], d[2]);
}

void  ColorMap::calcColorHot(float *d, size_t i)
{
    float t = (float)i/(float)m_numBuckets;
    if (t <= 1.0/3.0) {
        d[1] = d[2] = 0.0; d[0] = t/(1.0/3.0);
    } else if (t <= 2.0/3.0) {
        d[0] = 1.0; d[2] = 0.0; d[1] = (t-(1.0/3.0))/(1.0/3.0);
    } else {
        d[0] = 1.0; d[1] = 1.0; d[2] = (t-(2.0/3.0))/(1.0/3.0);
    }
}

void ColorMap::calcColorJet(float *d, size_t i)
{
    float t = (float)i/(float)m_numBuckets;
    if (t <= 0.125) {
      d[0] = d[1] = 0.0; d[2] = 0.5 + 0.5*(t/0.125);
    } else if (t < 0.375) {
      d[0] = 0.0; d[2] = 1.0; d[1] = (t-0.125)/0.25;
    } else if (t < 0.625) {
      d[1] = 1.0; d[0] = (t-0.375)/0.25;; d[2] = 1.0 - d[0];
    } else if (t < 0.875) {
      d[0] = 1.0; d[2] = 0.0; d[1] = 1.0 - (t-0.625)/0.25;
    } else {
      d[1] = d[2] = 0.0; d[2] = 1.0 - 0.5*(t/0.125);
    }
}

void ColorMap::convertHSVToRGB(float hue, float s, float v, float &r, float &g, float &b)
{
    float p1, p2, p3, i, f;
    float xh;

    if (hue == 360.0)
        hue = 0.0;

    xh = hue / 60.;                  // convert hue to be in 0,6
    i = (float)floor((double)xh);    // i is greatest integer smaller than h
    f = xh - i;                      // f is the fractional part of h
    p1 = v * (1 - s);
    p2 = v * (1 - (s * f));
    p3 = v * (1 - (s * (1 - f)));

    switch ((int) i)
    {
    case 0:
        r = v;
        g = p3;
        b = p1;
        break;
    case 1:
        r = p2;
        g = v;
        b = p1;
        break;
    case 2:
        r = p1;
        g = v;
        b = p3;
        break;
    case 3:
        r = p1;
        g = p2;
        b = v;
        break;
    case 4:
        r = p3;
        g = p1;
        b = v;
        break;
    case 5:
        r = v;
        g = p1;
        b = p2;
        break;
    }
}

} // namespace lvr2
