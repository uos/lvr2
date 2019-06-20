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

#include "lvr2/reconstruction/Projection.hpp"


#include <iostream>
using namespace std;

namespace lvr2
{

Projection::Projection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : m_width(width), m_height(height), m_optimize(optimize), m_system(system)
{
    // Convert degrees to radians
    m_maxH = maxH / 180.0 * m_ph;
    m_minH = minH / 180.0 * m_ph;
    m_maxV = maxV / 180.0 * m_ph;
    m_minV = minV / 180.0 * m_ph;
}



void Projection::setImageRatio()
{
    if(((double)m_xSize / m_ySize) != ((double)m_width / m_height))
    {
        if(m_optimize)
        {
            float tWidth;
            float tHeight;

            tWidth =  m_height * m_xSize / m_ySize;
            tHeight = m_width * m_ySize / m_xSize;

            if((double)(m_width/m_height) >= 1)
            {
                if((double)(m_xSize / m_ySize) >= 1)
                {
                    //m_height stays the same
                    if((double)(tWidth / m_height) >= 1)
                    {
                        m_width = tWidth;
                    }
                    //m_width stays the same
                    else if((double)(m_width / tHeight) >= 1)
                    {
                        m_height = tHeight;
                    }
                }
            }
            else
            {
                if((double)(m_xSize / m_ySize) < 1)
                {
                    //m_width stays the same
                    if((double)(m_width / tHeight) <= 1)
                    {
                        m_height = tHeight;
                    }
                    //m_height stays the same
                    else if((double)(tWidth / m_height) <= 1)
                    {
                        m_width = tWidth;
                    }
                }
            }
        }
    }
}



EquirectangularProjection::EquirectangularProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    //adding the longitude to x axis and latitude to y axis
    m_xSize = m_maxH - m_minH;
    m_ySize = m_maxV - m_minV;

    setImageRatio();

    m_xFactor = (float)m_width / m_xSize;
    m_yFactor = (float)m_height / m_ySize;

    m_maxWidth = m_width - 1;
    m_maxHeight = m_height - 1;

    //shift all the valuse to positive points on image
    m_lowShift = m_minV;

}

void EquirectangularProjection::project(int& i, int& j, float& range, float x, float y, float z)
{
    float kart[3];

    switch(m_system)
    {
        case ModelToImage::SLAM6D:
            kart[0] = z;
            kart[1] = -x;
            kart[2] = y;
            break;
        case ModelToImage::UOS:
            kart[0] = x;
            kart[1] = -z;
            kart[2] = y;
            break;
        case ModelToImage::NATIVE:
        default:
            kart[0] = x;
            kart[1] = y;
            kart[2] = z;

    }


//    if(m_system)
//    {
//        kart[0] = z;
//        kart[1] = -x;
//        kart[2] = y;
//    }
//    else
//    {
////        kart[0] = x;
////        kart[1] = -z;
////        kart[2] = y;

//        kart[0] = x;
//        kart[1] = y;
//        kart[2] = z;

//    }


    float polar[3] = {0, 0, 0};

    // Convert to polar coordinates
    if(kart[0] != 0 && kart[1] != 0 && kart[2] != 0)
    {
        toPolar(kart, polar);
    }
    else
    {
        i = 0;
        j = 0;
        return;
    }

    float theta = polar[0];
    float phi = polar[1];
    range = polar[2];

    // Horizantal angle of view of [0:360][minHorzAngle_:maxHorzAngle_] and vertical of [-40:60][minVertAngle_:maxVertAngle]
    // phi == longitude == horizantal angle of view of [0:360]
    // shift it to clockwise instead of counter clockwise
    phi = (2 * M_PI) - phi;

    // Theta == latitude == vertical angle of view of [-40:60]
    // shift the vertical angle instead of -90:90 to 0:180 from north to south pole
    theta -= M_PI/2.0;
    theta *= -1;

    i = (int) ( m_xFactor * phi);
    if (i < 0)
    {
        i = 0;
    }

    if (i > m_maxWidth)
    {
        i = m_maxWidth;
    }

    j = (int) ( m_yFactor * (theta - m_lowShift) );
    j = m_maxHeight - j;

    if (j < 0)
    {
        j = 0;
    }

    if (j > m_maxHeight)
    {
        j = m_maxHeight;
    }
    //cout << i << " " << j <<  " " << range << " / " << m_maxWidth << " " << m_maxHeight << endl;
}

ConicProjection::ConicProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    // Set up initial parameters according to MathWorld:
    // http://mathworld.wolfram.com/AlbersEqual-AreaConicProjection.html
    m_lat0 = (m_minV + m_maxV) / 2;
    m_long0 = (m_minH + m_maxH) / 2;
    m_phi1 = m_minV;
    m_phi2 = m_maxV;
    m_n = (sin(m_phi1) + sin(m_phi2)) / 2.;
    m_c = pow((cos(m_phi1)) + 2 * m_n * sin(m_phi1), 2);
    m_rho0 = sqrt(m_c - 2 * m_n * sin(m_lat0)) / m_n;

    // Set up max values for x and y and add the longitude to x axis and latitude to y axis
    m_maxX = (1./m_n * sqrt(m_c - 2*m_n * sin( m_minV)) ) * sin(m_n * (m_maxH - m_long0));
    m_minX = (1./m_n * sqrt(m_c - 2*m_n * sin( m_minV)) ) * sin(m_n * (m_minH - m_long0));
    m_xSize =  ( m_maxX - m_minX );

    m_maxY = m_rho0 - (1./m_n * sqrt(m_c - 2*m_n * sin(m_maxV)) ) * cos(m_n * (m_maxH - m_long0 ));
    m_minY = m_rho0 - (1./m_n * sqrt(m_c - 2*m_n * sin(m_minV)) ) * cos(m_n * ((m_minH + m_maxH)/2 - m_long0 ));
    m_ySize =  ( m_maxY - m_minY );

    setImageRatio();

    m_xFactor = (float) m_width / m_xSize;
    m_yFactor = (float) m_height / m_ySize;
    m_maxWidth = m_width- 1;


    // Shift all values to positive points on image
    m_maxHeight = m_height - 1;
}

CylindricalProjection::CylindricalProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    //adding the longitude to x and tan(latitude) to y
    m_xSize = m_maxH - m_minH;
    m_ySize = tan(m_maxV) - tan(m_minV);

    setImageRatio();

    //find the x and y range
    m_xFactor = (double) m_width / m_xSize;
    m_maxWidth = m_width - 1;
    m_yFactor = (double) m_height / m_ySize;
    m_heightLow = m_minV;
    m_maxHeight = m_height - 1;
}

MercatorProjection::MercatorProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    //find the x and y range
    m_xSize = m_maxH - m_minH;
    m_ySize =  ( log( tan( m_maxV) + ( 1 / cos( m_maxV) ) ) - log ( tan( m_minV) + (1 / cos(m_minV) ) ) );

    setImageRatio();

    m_xFactor = (double) m_width / m_xSize;
    m_maxWidth = m_width - 1;
    m_yFactor = (double) m_height / m_ySize;
    m_heightLow = log(tan(m_minV) + (1/cos(m_minV)));
    m_maxHeight = m_height - 1;
}

RectilinearProjection::RectilinearProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    int numberOfImages = 3;
    m_interval = (m_maxH - m_minH) / numberOfImages;
    m_iMinY = m_minV;
    m_iMaxY = m_maxV;

    // Latitude of projection center
    m_p1 = 0;

    m_iMinX = m_minH;
    m_iMaxX = m_minH + m_interval;

    // Longitude of projection center
    m_l0 = m_iMinX + m_interval / 2;

    // Finding min and max of the x direction
    m_coscRectlinear = sin(m_p1) * sin(m_iMaxY) + cos(m_p1) * cos(m_iMaxY) * cos(m_iMaxX - m_l0);
    m_max = (cos(m_iMaxY) * sin(m_iMaxX - m_l0) / m_coscRectlinear);
    m_coscRectlinear = sin(m_p1) * sin(m_iMinY) + cos(m_p1) * cos(m_iMinY) * cos(m_iMinX - m_l0);
    m_min = (cos(m_iMinY) * sin(m_iMinX - m_l0) / m_coscRectlinear);
    m_xSize =  (m_max - m_min);

    // Finding the min and max of y direction
    m_coscRectlinear = sin(m_p1) * sin(m_iMaxY) + cos(m_p1) * cos(m_iMaxY) * cos(m_iMaxX - m_l0);
    m_max = ( (cos(m_p1) * sin(m_iMaxY) - sin(m_p1) * cos(m_iMaxY) * cos(m_iMaxX - m_l0) )/ m_coscRectlinear);
    m_coscRectlinear = sin(m_p1) * sin(m_iMinY) + cos(m_p1) * cos(m_iMinY) * cos(m_iMinX - m_l0);
    m_min = ( (cos(m_p1) * sin(m_iMinY) - sin(m_p1) * cos(m_iMinY) * cos(m_iMinX - m_l0) )/ m_coscRectlinear);
    m_ySize = (m_max - m_min);

    setImageRatio();
}

PanniniProjection::PanniniProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    //default values for numberOfIMages and dPannini==param
    int param = 1;
    int numberOfIMages = 2;

    m_interval = (m_maxH - m_minH) / numberOfIMages;
    m_iMinY = m_minV;
    m_iMaxY = m_maxV;
    //latitude of projection center
    m_p1 = 0;

    m_iMinX = m_minH + (0 * m_interval);
    m_iMaxX = m_minH + ((0 + 1) * m_interval);
    //the longitude of projection center
    m_l0 = m_iMinX + m_interval / 2;

    //use the S variable of pannini projection mentioned in the thesis
    //finding the min and max of the x direction
    m_sPannini = (param + 1) / (param + sin(m_p1) * tan(m_iMaxY) + cos(m_p1) * cos(m_iMaxX - m_l0));
    m_max = m_sPannini * (sin(m_iMaxX - m_l0));
    m_sPannini = (param + 1) / (param + sin(m_p1) * tan(m_iMinY) + cos(m_p1) * cos(m_iMinX - m_l0));
    m_min = m_sPannini * (sin(m_iMinX - m_l0));
    m_xSize = m_max - m_min;
    //finding the min and max of y direction
    m_sPannini = (param + 1) / (param + sin(m_p1) * tan(m_iMaxY) + cos(m_p1) * cos(m_iMaxX - m_l0));
    m_max = m_sPannini * (tan(m_iMaxY) * (cos(m_p1) - sin(m_p1) * 1/tan(m_iMaxY) * cos(m_iMaxX - m_l0)));
    m_sPannini = (param + 1) / (param + sin(m_p1) * tan(m_iMinY) + cos(m_p1) * cos(m_iMinX - m_l0));
    m_min = m_sPannini * (tan(m_iMinY) * (cos(m_p1) - sin(m_p1) * 1/tan(m_iMinY) * cos(m_iMinX - m_l0)));
    m_ySize = m_max - m_min;

    setImageRatio();
}

StereographicProjection::StereographicProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
    : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    // Default values for numberOfIMages and rStereographic==param
    int param = 2;
    int numberOfIMages = 2;

    // m_l0 and m_p1 are the center of projection iminx, imaxx, iminy, imaxy are the bounderis of m_intervals
    m_interval = (m_maxH - m_minH) / numberOfIMages;
    m_iMinY = m_minV;
    m_iMaxY = m_maxV;

    // Latitude of projection center
    m_p1 = 0;

    m_iMinX = m_minH + (0 * m_interval);
    m_iMaxX = m_minH + ((0 + 1) * m_interval);

    // Longitude of projection center
    m_l0 = m_iMinX + m_interval / 2;

    // Use the R variable of stereographic projection mentioned in the thesis
    // finding the min and max of x direction
    m_k = (2 * param) / (1 + sin(m_p1) * sin(m_p1) + cos(m_p1) * cos(m_p1) * cos(m_iMaxX - m_l0));
    m_max = m_k * cos(m_p1) * sin (m_iMaxX - m_l0);
    m_k = (2 * param) / (1 + sin (m_p1) * sin(m_p1) + cos(m_p1) * cos(m_p1) * cos(m_iMinX -m_l0));
    m_min = m_k * cos(m_p1) * sin (m_iMinX -m_l0);
    m_xSize =  (m_max - m_min);

    // Finding the min and max of y direction
    m_k = (2 * param) / (1 + sin(m_p1) * sin(m_iMaxY) + cos(m_p1) * cos(m_iMaxY) * cos(m_iMaxX - m_l0));
    m_max = m_k * (cos(m_p1) * sin(m_iMaxY) - sin(m_p1) * cos(m_iMaxY) * cos(m_iMaxX - m_l0));
    m_k = (2 * param) / (1 + sin(m_p1) * sin(m_iMinY) + cos(m_p1) * cos(m_iMinY) * cos(m_iMinX - m_l0));
    m_min = m_k * (cos(m_p1) * sin(m_iMinY) - sin(m_p1) * cos(m_iMinY) * cos(m_iMinX - m_l0));
    m_ySize = (m_max - m_min);

    setImageRatio();
}

AzimuthalProjection::AzimuthalProjection(int width, int height, int minH, int maxH, int minV, int maxV, bool optimize, ModelToImage::CoordinateSystem system)
   : Projection(width, height, minH, maxH, minV, maxV, optimize, system)
{
    // set up initial parameters according to MathWorld: http://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html
    m_long0 = (m_minH + m_maxH) / 2;
    m_phi1 = (m_minV + m_maxV) / 2;

    // set up max values for x and y and add the longitude to x axis and latitude to y axis
    // sqrt(2/(1+sin(m_phi1)*sin(theta)+cos(m_phi1)*cos(theta)*cos(phi-m_long0)));
    m_maxX =  sqrt(2/(1+sin(m_phi1)*sin(m_maxH)+cos(m_phi1)*cos(m_maxH)*cos((m_minV/2+m_maxV/2)-m_long0)))*cos(m_maxH)*sin((m_minV/2+m_maxV/2)-m_long0);
    m_minX = sqrt(2/(1+sin(m_phi1)*sin(m_minH)+cos(m_phi1)*cos(m_minH)*cos((m_minV/2+m_maxV/2)-m_long0)))*cos(m_minH)*sin((m_minV/2+m_maxV/2)-m_long0);
    m_xSize =  ( m_maxX - m_minX );

    m_minY = m_minX;     // the thing is supposed to be circular, isn't it?
    m_maxY = m_maxX;
    m_ySize =  ( m_maxY - m_minY );

    setImageRatio();

    m_xSize=1/2.;
    m_ySize=1/2.;

    m_xFactor = (double) m_width / 2*m_xSize;
    m_maxWidth = m_width - 1;
    m_yFactor = (double) m_height / 2*m_ySize;
    //shift all the values to positive points on image
    m_maxHeight = m_height - 1;
}

void Projection::toPolar(const float _cartesian[], float polar[])
{
    float phi, theta, rho;

    float x_sqr, y_sqr, z_sqr;
    x_sqr = _cartesian[0] * _cartesian[0];
    y_sqr = _cartesian[1] * _cartesian[1];
    z_sqr = _cartesian[2] * _cartesian[2];

    rho = std::sqrt(x_sqr + y_sqr + z_sqr);

    float cartesian[3];
    cartesian[0] = _cartesian[0] / rho;
    cartesian[1] = _cartesian[1] / rho;
    cartesian[2] = _cartesian[2] / rho;

    phi = std::acos(cartesian[2]);

    float theta0;

    if(std::abs(phi) < 0.0001)
    {
        theta = 0.0;
    }
    else if(std::abs(M_PI - phi) < 0.0001)
    {
        theta = 0.0;
    }
    else
    {
        if(std::abs(cartesian[0]/std::sin(phi)) > 1.0)
        {
            if(cartesian[0]/std::sin(phi) < 0)
            {
                theta0 = M_PI;
            }
            else
            {
                theta0 = 0.0;
            }
        }
        else
        {
            theta0 = acos(cartesian[0]/std::sin(phi));

        }

        float sintheta = cartesian[1]/std::sin(phi);
        float EPS = 0.0001;

        if(std::abs(std::sin(theta0) - sintheta) < EPS)
        {
            theta = theta0;
        }
        else if(std::abs( std::sin( 2*M_PI - theta0 ) - sintheta ) < EPS)
        {
            theta = 2*M_PI - theta0;
        }
        else
        {
            theta = 0;
        }
    }
    polar[0] = phi;
    polar[1] = theta;
    polar[2] = rho;
}

} // namespace lvr2
