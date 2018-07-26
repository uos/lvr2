/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /**
 *
 * @file      MeshLoader.hpp
 * @brief     Interface for all mesh loading classes.
 * @details   The MeshLoader class specifies the storage and access to all
 *            available mesh data by implementing the get and set methods for
 *            these data.
 *
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @author    Thomas Wiemann, twiemann@uos.de, Universität Osnabrück
 * @author    Jan Philipp Vogtherr (jvogtherr@uni-osnabrueck.de)
 *
 **/

#ifndef MESHBUFFER2_HPP_
#define MESHBUFFER2_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>

#include <vector>

#include <lvr2/io/DataStruct.hpp>
#include <lvr2/texture/Material.hpp>
#include <lvr2/texture/Texture.hpp>

namespace lvr2
{

using std::vector;


// Forward declaration
template<typename BaseVecT>
class MaterializerResult;


template<typename BaseVecT>
class MeshBuffer
{

    public:
        MeshBuffer() {};

  //      MeshBuffer(lvr::MeshBuffer& oldBuffer);

        vector<float> getVertices();
        vector<unsigned char> getVertexColors();
        vector<float> getVertexConfidences();
        vector<float> getVertexIntensities();
        vector<float> getVertexNormals();
        vector<float> getVertexTextureCoordinates();
        vector<unsigned int> getFaceIndices();
        vector<unsigned int> getFaceMaterialIndices();
        vector<unsigned int> getClusterMaterialIndices();
        vector<Material> getMaterials();
        vector<Texture<BaseVecT>> getTextures();
        vector<vector<unsigned int>> getClusterFaceIndices();

        void setVertices(vector<float> v);
        void setVertexColors(vector<unsigned char> v);
        void setVertexConfidences(vector<float> v);
        void setVertexIntensities(vector<float> v);
        void setVertexNormals(vector<float> v);
        void setVertexTextureCoordinates(vector<float> v);
        void setFaceIndices(vector<unsigned int> v);
        void setFaceMaterialIndices(vector<unsigned int> v);
        void setClusterMaterialIndices(vector<unsigned int> v);
        void setMaterials(vector<Material> v);
        void setTextures(vector<Texture<BaseVecT>> v);
        void setClusterFaceIndices(vector<vector<unsigned int>> v);

//        // Convert to lvr1 MeshBuffer
//        boost::shared_ptr<lvr::MeshBuffer> toOldBuffer();
//        // Overloaded method for conversion with textures
//        boost::shared_ptr<lvr::MeshBuffer> toOldBuffer(MaterializerResult<BaseVecT> materializerResult);

    protected:

        vector<float> m_vertices;
        vector<unsigned char> m_vertexColors;
        vector<float> m_vertexConfidences;
        vector<float> m_vertexIntensities;
        vector<float> m_vertexNormals;
        vector<float> m_vertexTextureCoordinates;
        vector<unsigned int> m_faceIndices;
        vector<unsigned int> m_faceMaterialIndices;
        vector<unsigned int> m_clusterMaterialIndices;
        vector<Material> m_materials;
        vector<Texture<BaseVecT>> m_textures;
        vector<vector<unsigned int>> m_clusterFaceIndices;

};


template <typename BaseVecT>
using MeshBufferPtr = std::shared_ptr<MeshBuffer<BaseVecT>>;

} // namespace

#include <lvr2/io/MeshBuffer.tcc>

#endif /* MESHBUFFER2_HPP_ */
