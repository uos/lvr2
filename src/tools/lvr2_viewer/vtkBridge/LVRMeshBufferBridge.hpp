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

#include <lvr2/io/MeshBuffer2.hpp>
#include <lvr2/display/TexturedMesh.hpp>
#include <lvr2/display/GlTexture.hpp>
#include <lvr/geometry/Vertex.hpp>

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>

#include <boost/shared_ptr.hpp>

namespace lvr2
{

class LVRMeshBufferBridge
{

    using Vec      = BaseVector<float>;

public:
    LVRMeshBufferBridge(MeshBuffer2Ptr meshbuffer);
    LVRMeshBufferBridge(const LVRMeshBufferBridge& b);

    virtual ~LVRMeshBufferBridge();

    vtkSmartPointer<vtkActor>   			getMeshActor();
    vtkSmartPointer<vtkActor>   			getWireframeActor();
    vtkSmartPointer<vtkActorCollection>		getTexturedActors();
    size_t                      			getNumTriangles();
    size_t                      			getNumVertices();
    bool                        			hasTextures();

    void setBaseColor(float r, float g, float b);
    void setOpacity(float opacityValue);
    MeshBuffer2Ptr getMeshBuffer();
    void setVisibility(bool visible);
    void setShading(int shader);

    size_t									getNumColoredFaces();
    size_t									getNumTexturedFaces();
    size_t									getNumTextures();

protected:
    void computeMeshActor(MeshBuffer2Ptr meshbuffer);
    size_t                          m_numVertices;
    size_t                          m_numFaces;
    vtkSmartPointer<vtkActor>       m_meshActor;
    vtkSmartPointer<vtkActor>       m_wireframeActor;
    MeshBuffer2Ptr                  m_meshBuffer;

    size_t							m_numColoredFaces;
    size_t							m_numTexturedFaces;
    size_t							m_numTextures;

    void computeMaterialGroups(vector<MaterialGroup*>& matGroups, vector<MaterialGroup*>& colorMatGroups);
    void remapTexturedIndices(MaterialGroup* g, vector<Vector<Vec> >& vertices, vector<Vector<Vec> >& texCoords, vector<int>& indices);
    void remapIndices(vector<MaterialGroup*> g, vector<Vector<Vec> >& vertices, vector<lvr::Vertex<unsigned char> >& colors, vector<int>& indices);


    vtkSmartPointer<vtkActor>		getTexturedActor(MaterialGroup* g);
    vtkSmartPointer<vtkActor>		getColorMeshActor(vector<MaterialGroup*> groups);

    vtkSmartPointer<vtkTexture>		getTexture(int index);

private:

};

typedef boost::shared_ptr<LVRMeshBufferBridge> MeshBufferBridgePtr;

} /* namespace lvr2 */

#endif /* LVRMESHBUFFERBRIDGE_H_ */
