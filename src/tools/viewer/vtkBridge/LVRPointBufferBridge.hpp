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
 * LVRPointBufferBridge.hpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPOINTBUFFERBRIDGE_HPP_
#define LVRPOINTBUFFERBRIDGE_HPP_

#include <lvr/display/ColorMap.hpp>
#include <lvr/io/PointBuffer.hpp>
//#include <LVRPointBufferBridge.hpp>

#include <vtkSmartPointer.h>
#include <vtkActor.h>

#include <boost/shared_ptr.hpp>

namespace lvr
{

struct Pose;

class LVRPointBufferBridge
{
public:
    LVRPointBufferBridge(PointBufferPtr pointcloud);
    LVRPointBufferBridge(const LVRPointBufferBridge& b);
    virtual ~LVRPointBufferBridge();

    vtkSmartPointer<vtkActor>   getPointCloudActor();
    size_t                      getNumPoints();
    bool                        hasNormals();
    bool                        hasColors();

    void setBaseColor(float r, float g, float b);
    void setPointSize(int pointSize);
    void setOpacity(float opacityValue);
    void setVisibility(bool visible);
    /// set the spectral channel for (r, g, b) and set if it is used
    void setSpectralChannels(color<size_t> channels, color<bool> use_channel);
    /// get spectral channel mappings
    void getSpectralChannels(color<size_t> &channels, color<bool> &use_channel) const;
    /// set the gradienttype, desired channel, if the outputcolor should be normalized and if the NDVI should be used instead of the channel
    void setSpectralColorGradient(GradientType gradient, size_t channel, bool normalized = false, bool ndvi = false);
    /// get the gradienttype, channel, normalizend and ndvi flags
    void getSpectralColorGradient(GradientType &gradient, size_t &channel, bool &normalized, bool &useNDVI) const;
    /// switch between spectral mapping and gradient
    void useGradient(bool useGradient);
    /// get the point buffer
    PointBufferPtr getPointBuffer();

private:
    /// update the view with gradient information
    void refreshSpectralGradient();
    /// update the view with channel mappings
    void refreshSpectralChannel();

protected:

    void computePointCloudActor(PointBufferPtr pc);

    vtkSmartPointer<vtkActor>       m_pointCloudActor;
    size_t                          m_numPoints;
    bool                            m_hasNormals;
    bool                            m_hasColors;
    PointBufferPtr                  m_pointBuffer;
    bool                            m_useGradient;
    bool                            m_useNormalizedGradient;
    color<size_t>                   m_spectralChannels;
    color<bool>                     m_useSpectralChannel;
    GradientType                    m_spectralGradient;
    size_t                          m_spectralGradientChannel;
    bool                            m_useNDVI;
};

typedef boost::shared_ptr<LVRPointBufferBridge> PointBufferBridgePtr;

} /* namespace lvr */

#endif /* LVRPOINTBUFFERBRIDGE_HPP_ */
