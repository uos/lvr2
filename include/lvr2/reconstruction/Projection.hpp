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

#ifndef PROJECTION_H
#define PROJECTION_H
#include <cmath>

#include "lvr2/reconstruction/ModelToImage.hpp"

namespace lvr2
{

class Projection
{
public:

    Projection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

    virtual void project(int&i , int&j, float& r, float x, float y, float z) = 0;

    int w() { return m_width;}
    int h() { return m_height;}

protected:

    inline void toPolar(const float point[], float polar[]);

    float       m_xSize;
    float       m_ySize;
    float       m_xFactor;
    float       m_yFactor;
    int         m_width;
    int         m_height;
    float       m_minH;
    float       m_maxH;
    float       m_minV;
    float       m_maxV;

    bool        m_optimize;
    ModelToImage::CoordinateSystem        m_system;

    void setImageRatio();

    static constexpr float m_ph = 1.570796327;
};

class EquirectangularProjection: public Projection
{
public:
    EquirectangularProjection(int width, int height,
                              int minH, int maxH,
                              int minV, int maxV,
                              bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

    virtual void project(int&i , int&j, float& r, float x, float y, float z) override;

protected:
    float       m_xFactor;
    float       m_yFactor;
    int         m_maxWidth;
    int         m_maxHeight;
    float       m_lowShift;
};

class MercatorProjection: public Projection
{
public:
    MercatorProjection(int width, int height,
                       int minH, int maxH,
                       int minV, int maxV,
                       bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);
protected:
    float       m_heightLow;
    int         m_maxWidth;
    int         m_maxHeight;
};

class CylindricalProjection: public Projection
{
public:
    CylindricalProjection(int width, int height,
                          int minH, int maxH,
                          int minV, int maxV,
                          bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

protected:
    float       m_heightLow;
    int         m_maxWidth;
    int         m_maxHeight;
};


class ConicProjection : public Projection
{
public:
    ConicProjection(int width, int height,
                    int minH, int maxH,
                    int minV, int maxV,
                    bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

protected:
    float       m_lat0;
    float       m_long0;
    float       m_lat1;
    float       m_phi1;
    float       m_phi2;
    float       m_n;
    float       m_c;
    float       m_rho0;
    float       m_maxX;
    float       m_minX;
    float       m_minY;
    float       m_maxY;
    int         m_maxWidth;
    int         m_maxHeight;

};

class RectilinearProjection : public Projection
{
public:
    RectilinearProjection(int width, int height,
                          int minH, int maxH,
                          int minV, int maxV,
                          bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);
protected:
    float       m_interval;
    float       m_iMinY;
    float       m_iMaxY;
    float       m_iMinX;
    float       m_iMaxX;
    float       m_coscRectilinear;
    float       m_l0;
    float       m_coscRectlinear;
    float       m_max;
    float       m_min;
    float       m_p1;

};

class PanniniProjection: public Projection
{
public:
    PanniniProjection(int width, int height,
                      int minH, int maxH,
                      int minV, int maxV,
                      bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

protected:
    float       m_interval;
    float       m_iMinY;
    float       m_iMaxY;
    float       m_iMinX;
    float       m_iMaxX;
    float       m_max;
    float       m_min;
    float       m_l0;
    float       m_sPannini;
    float       m_p1;
};

class StereographicProjection: public Projection
{
public:
    StereographicProjection(int width, int height,
                            int minH, int maxH,
                            int minV, int maxV,
                            bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

protected:
    float       m_interval;
    float       m_iMinY;
    float       m_iMaxY;
    float       m_iMinX;
    float       m_iMaxX;
    float       m_max;
    float       m_min;
    float       m_l0;
    float       m_p1;
    float       m_k;
};

class AzimuthalProjection: public Projection
{
public:
    AzimuthalProjection(int width, int height,
                        int minH, int maxH,
                        int minV, int maxV,
                        bool optimize, ModelToImage::CoordinateSystem system = ModelToImage::NATIVE);

protected:
    float       m_kPrime;
    float       m_long0;
    float       m_phi1;
    float       m_maxX;
    float       m_minX;
    float       m_minY;
    float       m_maxY;
    int         m_maxWidth;
    int         m_maxHeight;
};


} // namespace lvr2

#endif // PROJECTION_H
