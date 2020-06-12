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
 * LVRPointBufferBridge.hpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPOINTBUFFERBRIDGE_HPP_
#define LVRPOINTBUFFERBRIDGE_HPP_

#include "lvr2/display/ColorMap.hpp"
#include "lvr2/io/PointBuffer.hpp"

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkDoubleArray.h>

#include <boost/shared_ptr.hpp>

namespace lvr2
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
    void setNormalsVisibility(bool visible);
    void setColorsVisibility(bool visible);
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
    vtkSmartPointer<vtkPolyData> getPolyData();

    vtkSmartPointer<vtkPolyData> getPolyIDData();

private:
    /// update the view with gradient information
    void refreshSpectralGradient();
    /// update the view with channel mappings
    void refreshSpectralChannel();

protected:

    void computePointCloudActor(PointBufferPtr pc);
    
    vtkSmartPointer<vtkPolyData> m_vtk_polyData;

    //Maybe this is not neaded but 
    vtkSmartPointer<vtkPolyData> m_id_polyData;

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
    vtkSmartPointer<vtkDoubleArray> m_vtk_normals;
};

typedef boost::shared_ptr<LVRPointBufferBridge> PointBufferBridgePtr;

} /* namespace lvr2 */

#endif /* LVRPOINTBUFFERBRIDGE_HPP_ */
