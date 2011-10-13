/**
 * ColorMap.cpp
 *
 *  @date 30.08.2011
 *  @author Thomas Wiemann
 */

#include "ColorMap.hpp"

#include <cassert>
#include <cmath>

namespace lssr
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
    default:
        color[0] = 1.0;
        color[1] = 1.0;
        color[2] = 1.0;
        break;

    }
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

} // namespace lssr


