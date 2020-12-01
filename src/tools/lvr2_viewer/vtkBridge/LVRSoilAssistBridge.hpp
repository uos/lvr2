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
 * LVRSoilAssistBridge.hpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPOLYGONBRIDGE_HPP_
#define LVRPOLYGONBRIDGE_HPP_

#include "lvr2/display/ColorMap.hpp"
#include "lvr2/io/Polygon.hpp"
#include "lvr2/io/SoilAssistField.hpp"
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkDoubleArray.h>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace lvr2
{

struct Pose;

class LVRSoilAssistBridge
{
public:
    LVRSoilAssistBridge(SoilAssistFieldPtr field);
    LVRSoilAssistBridge(const LVRSoilAssistBridge& b);
    virtual ~LVRSoilAssistBridge();

    std::vector<vtkSmartPointer<vtkActor>>   getPolygonActors();
    size_t                      getNumPoints();

    void setBaseColor(float r, float g, float b);
    void setPointSize(int pointSize);
    void setOpacity(float opacityValue);
    void setVisibility(bool visible);

    PolygonPtr getPolygon();


protected:

    void computePolygonActor(SoilAssistFieldPtr poly);
    vtkSmartPointer<vtkActor> makeArrow(float * start, float * end);
    vtkSmartPointer<vtkActor> computePolygonActor(PolygonPtr poly, bool polygon=true);
    std::vector<vtkSmartPointer<vtkActor> >       m_actors;
    PolygonPtr                      m_polygon;
    float               m_offset[3];
    bool                m_offset_set;

};

typedef boost::shared_ptr<LVRSoilAssistBridge> SoilAssistBridgePtr;

} /* namespace lvr2 */

#endif /* LVRPOLYGONBRIDGE_HPP_ */
