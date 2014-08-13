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
 * LVRMeshBufferBridge.h
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRMESHBUFFERBRIDGE_H_
#define LVRMESHBUFFERBRIDGE_H_

#include "io/MeshBuffer.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>

#include <boost/shared_ptr.hpp>

namespace lvr
{

class LVRMeshBufferBridge
{
public:
    LVRMeshBufferBridge(MeshBufferPtr meshbuffer);
    LVRMeshBufferBridge(const LVRMeshBufferBridge& b);

    virtual ~LVRMeshBufferBridge();

    vtkSmartPointer<vtkActor>   getMeshActor();
    vtkSmartPointer<vtkActor>   getWireframeActor();
    size_t                      getNumTriangles();
    size_t                      getNumVertices();
    bool                        hasTextures();

    void setBaseColor(float r, float g, float b);
    void setOpacity(float opacityValue);
    MeshBufferPtr getMeshBuffer();
    void setVisibility(bool visible);
    void setShading(int shader);

protected:
    void computeMeshActor(MeshBufferPtr meshbuffer);
    size_t                          m_numVertices;
    size_t                          m_numFaces;
    float*                          m_color;
    vtkSmartPointer<vtkActor>       m_meshActor;
    vtkSmartPointer<vtkActor>       m_wireframeActor;
    MeshBufferPtr                   m_meshBuffer;
};

typedef boost::shared_ptr<LVRMeshBufferBridge> MeshBufferBridgePtr;

} /* namespace lvr */

#endif /* LVRMESHBUFFERBRIDGE_H_ */
