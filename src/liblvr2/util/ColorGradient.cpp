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

#include "lvr2/util/ColorGradient.hpp"
#include "lvr2/util/Timestamp.hpp"

#include <cassert>
#include <cmath>

namespace lvr2
{

void ColorGradient::getColor(RGBFColor& color, size_t bucket, ColorGradient::GradientType gradient ) const
{
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
    case BLACK:
        color[0] = 0.0;
        color[1] = 0.0;
        color[2] = 0.0;
        break;
    case WHITE:
    default:
        color[0] = 1.0;
        color[1] = 1.0;
        color[2] = 1.0;
        break;

    }
}

void ColorGradient::getColor(RGB8Color& color, size_t bucket, ColorGradient::GradientType gradient) const
{
    RGBFColor fc;
    getColor(fc, bucket, gradient);
    color[0] = static_cast<uint8_t>(fc[0] * 255);
    color[1] = static_cast<uint8_t>(fc[1] * 255); 
    color[2] = static_cast<uint8_t>(fc[2] * 255);
}

void ColorGradient::calcColorSimpsons(RGBFColor& color, size_t bucket) const
{
	color[0] = fabs( cos( bucket )  );
	color[1] = fabs( sin( bucket * 30 ) );
	color[2] = fabs( sin( bucket * 2 ) );
    //std::cout << bucket << " " << color[0] << " " << color[1] << " " << color[2] << std::endl;
}

void ColorGradient::calcColorGrey(RGBFColor& d, size_t i) const
{
    int s = i % m_numBuckets;
    d[0] = (float)s/(float)m_numBuckets;
    d[1] = (float)s/(float)m_numBuckets;
    d[2] = (float)s/(float)m_numBuckets;
}

void ColorGradient::calcColorHSV(RGBFColor& d, size_t i) const
{
    int s = i % m_numBuckets;
    float t = (float)s/(float)m_numBuckets;
    convertHSVToRGB(360.0*t, 1.0, 1.0,  d[0], d[1], d[2]);

}

void ColorGradient::calcColorSHSV(RGBFColor& d, size_t i) const
{
    int s = i % m_numBuckets;
    float t = (float)s/(float)m_numBuckets;
    convertHSVToRGB(360.0*t, 1.0, 1.0,  d[0], d[1], d[2]);
}

void  ColorGradient::calcColorHot(RGBFColor& d, size_t i) const
{
    int s = i % m_numBuckets;
    float t = (float)s/(float)m_numBuckets;
    if (t <= 1.0/3.0) {
        d[1] = d[2] = 0.0; d[0] = t/(1.0/3.0);
    } else if (t <= 2.0/3.0) {
        d[0] = 1.0; d[2] = 0.0; d[1] = (t-(1.0/3.0))/(1.0/3.0);
    } else {
        d[0] = 1.0; d[1] = 1.0; d[2] = (t-(2.0/3.0))/(1.0/3.0);
    }
}

void ColorGradient::calcColorJet(RGBFColor& d, size_t i) const
{
    int s = i % m_numBuckets;
    float t = (float)s/(float)m_numBuckets;
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

void ColorGradient::convertHSVToRGB(float hue, float s, float v, float &r, float &g, float &b) const
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

ColorGradient::GradientType ColorGradient::gradientFromString(const std::string& s)
{
    if(s == "Simpsons" || s == "simpsons" || s == "SIMPSONS")
    {
        return SIMPSONS;
    }
    else if(s == "GREY" || s == "grey" || s == "Grey")
    {
        return GREY;
    } 
    else if(s == "HSV" || s == "hsv" || s == "Hsv")
    {
        return HSV;
    }
    else if(s == "JET" || s == "jet" || s =="Jet")
    {
        return JET;
    }
    else if(s == "HOT" || s == "hot" || s =="Hot")
    {
        return HOT;
    }
    else if(s == "SHSV" || s == "shsv" || s =="Shsv")
    {
        return SHSV;
    }
    else if(s == "BLACK" || s == "black" || s =="Black")
    {
        return BLACK;
    }
    else if(s == "White" || s == "WHITE" || s =="white")
    {
        return WHITE;
    }
    else
    {
        std::cout << timestamp << "Warning: Unknown color graditent type: '" << s << "'" << std::endl;
        return GREY;
    }
}

} // namespace lvr2
