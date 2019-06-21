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
 * LVRMeshBufferBridge.h
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRMESHBUFFERBRIDGE_H_
#define LVRMESHBUFFERBRIDGE_H_

#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/display/TexturedMesh.hpp"
#include "lvr2/display/GlTexture.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>

#include <boost/shared_ptr.hpp>

namespace lvr2
{

class LVRMeshBufferBridge
{

    using Vec      = BaseVector<float>;
    using VecUChar = BaseVector<unsigned char>;

public:
    LVRMeshBufferBridge(MeshBufferPtr meshbuffer);
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
    MeshBufferPtr getMeshBuffer();
    void setVisibility(bool visible);
    void setShading(int shader);

    size_t									getNumColoredFaces();
    size_t									getNumTexturedFaces();
    size_t									getNumTextures();

protected:
    void computeMeshActor(MeshBufferPtr meshbuffer);
    size_t                          m_numVertices;
    size_t                          m_numFaces;
    vtkSmartPointer<vtkActor>       m_meshActor;
    vtkSmartPointer<vtkActor>       m_wireframeActor;
    vtkSmartPointer<vtkActorCollection> m_texturedActors;
    MeshBufferPtr                  m_meshBuffer;

    size_t							m_numColoredFaces;
    size_t							m_numTexturedFaces;
    size_t							m_numTextures;

    void computeMaterialGroups(vector<MaterialGroup*>& matGroups, vector<MaterialGroup*>& colorMatGroups);
    void remapTexturedIndices(MaterialGroup* g, vector<Vec >& vertices, vector<Vec >& texCoords, vector<int>& indices);
    void remapIndices(vector<MaterialGroup*> g, vector<Vec >& vertices, vector<VecUChar >& colors, vector<int>& indices);


    vtkSmartPointer<vtkActor>		getTexturedActor(MaterialGroup* g);
    vtkSmartPointer<vtkActor>		getColorMeshActor(vector<MaterialGroup*> groups);

    vtkSmartPointer<vtkTexture>		getTexture(int index);

private:

};

typedef boost::shared_ptr<LVRMeshBufferBridge> MeshBufferBridgePtr;

} /* namespace lvr2 */

#endif /* LVRMESHBUFFERBRIDGE_H_ */
