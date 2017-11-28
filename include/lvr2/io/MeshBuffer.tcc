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

namespace lvr2
{

template <typename BaseVecT>
MeshBuffer<BaseVecT>::MeshBuffer(lvr::MeshBuffer& oldBuffer)
{
    size_t len;
    auto vertex_buf = oldBuffer.getVertexArray(len);
    if(len>0)
    {
        m_vertices.reserve(len * 3);
        std::copy(  vertex_buf.get(),
                    vertex_buf.get() + len*3,
                    m_vertices.begin());
    }

    auto vertexC_buf = oldBuffer.getVertexColorArray(len);
    if(len > 0)
    {
        m_vertexColors.reserve(len * 3);
        std::copy(  vertexC_buf.get(),
                    vertexC_buf.get() + len*3,
                    m_vertexColors.begin());
    }

    auto vertexConf_buf = oldBuffer.getVertexConfidenceArray(len);
    if(len > 0)
    {
        m_vertexConfidences.reserve(len);
        std::copy(  vertexConf_buf.get(),
                    vertexConf_buf.get() + len,
                    m_vertexConfidences.begin());
    }

    auto vertexInt_buf = oldBuffer.getVertexIntensityArray(len);
    if(len > 0)
    {
        m_vertexIntensities.reserve(len);
        std::copy(  vertexInt_buf.get(),
                    vertexInt_buf.get() + len,
                    m_vertexIntensities.begin());
    }

    auto vertexNormal_buf = oldBuffer.getVertexNormalArray(len);
    if(len > 0)
    {
        m_vertexNormals.reserve(len * 3);
        std::copy(  vertexNormal_buf.get(),
                    vertexNormal_buf.get() + len*3,
                    m_vertexNormals.begin());
    }

    auto vertexTex_buf = oldBuffer.getVertexTextureCoordinateArray(len);
    if(len > 0)
    {
        m_vertexTextureCoordinates.reserve(len * 3);
        std::copy(  vertexTex_buf.get(),
                    vertexTex_buf.get() + len*3,
                    m_vertexTextureCoordinates.begin());
    }

    auto faceInd_buf = oldBuffer.getFaceArray(len);
    if(len > 0)
    {
        m_faceIndices.reserve(len * 3);
        std::copy(  faceInd_buf.get(),
                    faceInd_buf.get() + len*3,
                    m_faceIndices.begin());
    }

    auto faceMaterialInd_buf = oldBuffer.getFaceMaterialIndexArray(len);
    if(len > 0)
    {
        m_faceMaterialIndices.reserve(len);
        std::copy(  faceMaterialInd_buf.get(),
                    faceMaterialInd_buf.get() + len,
                    m_faceMaterialIndices.begin());
    }

    auto material_buf = oldBuffer.getMaterialArray(len);
    if(len > 0)
    {
        m_materials.reserve(len);
        // TODO old material to new material

        for(size_t i=0; i<len; i++)
        {

        }

    }

    auto texture_buf = oldBuffer.getTextureArray(len);
    if(len > 0)
    {
        m_textures.reserve(len);
        // TODO old texture array to new
        for(size_t i=0; i<len; i++)
        {
            m_textures[i] = Texture<BaseVecT>(i,texture_buf[i]);
        }
    }

    // TODO: this vectors???
    // m_clusterMaterialIndices
    // m_clusterFaceIndices

}

template<typename BaseVecT>
vector<float> MeshBuffer<BaseVecT>::getVertices()
{
    return m_vertices;
}

template<typename BaseVecT>
vector<unsigned char> MeshBuffer<BaseVecT>::getVertexColors()
{
    return m_vertexColors;
}

template<typename BaseVecT>
vector<float> MeshBuffer<BaseVecT>::getVertexConfidences()
{
    return m_vertexConfidences;
}

template<typename BaseVecT>
vector<float> MeshBuffer<BaseVecT>::getVertexIntensities()
{
    return m_vertexIntensities;
}

template<typename BaseVecT>
vector<float> MeshBuffer<BaseVecT>::getVertexNormals()
{
    return m_vertexNormals;
}

template<typename BaseVecT>
vector<float> MeshBuffer<BaseVecT>::getVertexTextureCoordinates()
{
    return m_vertexTextureCoordinates;
}

template<typename BaseVecT>
vector<unsigned int> MeshBuffer<BaseVecT>::getFaceIndices()
{
    return m_faceIndices;
}

template<typename BaseVecT>
vector<unsigned int> MeshBuffer<BaseVecT>::getFaceMaterialIndices()
{
    return m_faceMaterialIndices;
}


template<typename BaseVecT>
vector<unsigned int> MeshBuffer<BaseVecT>::getClusterMaterialIndices()
{
    return m_clusterMaterialIndices;
}

template<typename BaseVecT>
vector<Material> MeshBuffer<BaseVecT>::getMaterials()
{
    return m_materials;
}

template<typename BaseVecT>
vector<Texture<BaseVecT>> MeshBuffer<BaseVecT>::getTextures()
{
    return m_textures;
}

template<typename BaseVecT>
vector<vector<unsigned int>> MeshBuffer<BaseVecT>::getClusterFaceIndices()
{
    return m_clusterFaceIndices;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setVertices(vector<float> v)
{
    m_vertices = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setVertexColors(vector<unsigned char> v)
{
    m_vertexColors = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setVertexConfidences(vector<float> v)
{
    m_vertexConfidences = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setVertexIntensities(vector<float> v)
{
    m_vertexIntensities = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setVertexNormals(vector<float> v)
{
    m_vertexNormals = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setVertexTextureCoordinates(vector<float> v)
{
    m_vertexTextureCoordinates = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setFaceIndices(vector<unsigned int> v)
{
    m_faceIndices = v;
}


template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setFaceMaterialIndices(vector<unsigned int> v)
{
    m_faceMaterialIndices = v;
}


template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setClusterMaterialIndices(vector<unsigned int> v)
{
    m_clusterMaterialIndices = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setMaterials(vector<Material> v)
{
    m_materials = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setTextures(vector<Texture<BaseVecT>> v)
{
    m_textures = v;
}

template<typename BaseVecT>
void MeshBuffer<BaseVecT>::setClusterFaceIndices(vector<vector<unsigned int>> v)
{
    m_clusterFaceIndices = v;
}

template<typename BaseVecT>
boost::shared_ptr<lvr::MeshBuffer> MeshBuffer<BaseVecT>::toOldBuffer()
{
    auto buffer = boost::make_shared<lvr::MeshBuffer>();
    buffer->setVertexArray(m_vertices);
    if (m_vertexConfidences.size() > 0)
    {
        buffer->setVertexConfidenceArray(m_vertexConfidences);
    }
    if (m_vertexIntensities.size() > 0)
    {
        buffer->setVertexIntensityArray(m_vertexIntensities);
    }
    buffer->setVertexNormalArray(m_vertexNormals);
    buffer->setVertexColorArray(m_vertexColors);
    buffer->setFaceArray(m_faceIndices);
    buffer->setVertexTextureCoordinateArray(m_vertexTextureCoordinates);
    buffer->setFaceMaterialIndexArray(m_faceMaterialIndices);
    // TODO: für materials und texturen muss die untere methode benutzt werden. bessere lösung finden!
    return buffer;
}

template<typename BaseVecT>
boost::shared_ptr<lvr::MeshBuffer> MeshBuffer<BaseVecT>::toOldBuffer(MaterializerResult<BaseVecT> materializerResult)
{
    auto buffer = toOldBuffer();
    vector<lvr::Material*> materials;
    for (auto material : m_materials)
    {
        lvr::Material* m = new lvr::Material; // TODO memory leak
        if (material.m_color)
        {
            m->r = material.m_color.get()[0];
            m->g = material.m_color.get()[1];
            m->b = material.m_color.get()[2];
        }
        else
        {
            m->r = 255;
            m->g = 255;
            m->b = 255;
        }
        if (material.m_texture && materializerResult.m_textures)
        {
            int textureIndex = materializerResult.m_textures.get().get(material.m_texture.get()).get().m_index;
            m->texture_index = textureIndex;
        }
        else
        {
            m->texture_index = -1;
        }
        materials.push_back(m);
    }
    buffer->setMaterialArray(materials);
    // TODO: set textures (lvr2::Texture muss zu GlTexture* umgebaut werden)
    return buffer;
}


} /* namespace lvr2 */
