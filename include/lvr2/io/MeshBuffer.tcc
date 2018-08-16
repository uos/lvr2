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

//template <typename BaseVecT>
//MeshBuffer::MeshBuffer(lvr::MeshBuffer& oldBuffer)
//{

//    // TODO: Obj files

//    size_t len = 0;
//    size_t vertex_len = 0;
//    auto vertex_buf = oldBuffer.getVertexArray(vertex_len);
//    if(vertex_len>0)
//    {
//        m_vertices.resize(vertex_len * 3);
//        std::copy(  vertex_buf.get(),
//                    vertex_buf.get() + vertex_len*3,
//                    m_vertices.begin() );
//    }


//    len = 0;
//    auto vertexC_buf = oldBuffer.getVertexColorArray(len);
//    if(len > 0)
//    {
//        m_vertexColors.resize(len * 3);
//        std::copy(  vertexC_buf.get(),
//                    vertexC_buf.get() + len*3,
//                    m_vertexColors.begin());
//    }


//    len = 0;
//    auto vertexConf_buf = oldBuffer.getVertexConfidenceArray(len);
//    if(len > 0)
//    {
//        m_vertexConfidences.resize(len);
//        std::copy(  vertexConf_buf.get(),
//                    vertexConf_buf.get() + len,
//                    m_vertexConfidences.begin());
//    }


//    len = 0;
//    auto vertexInt_buf = oldBuffer.getVertexIntensityArray(len);
//    if(len > 0)
//    {
//        m_vertexIntensities.resize(len);
//        std::copy(  vertexInt_buf.get(),
//                    vertexInt_buf.get() + len,
//                    m_vertexIntensities.begin());
//    }


//    len = 0;
//    auto vertexNormal_buf = oldBuffer.getVertexNormalArray(len);
//    if(len > 0)
//    {
//        m_vertexNormals.resize(len * 3);
//        std::copy(  vertexNormal_buf.get(),
//                    vertexNormal_buf.get() + len*3,
//                    m_vertexNormals.begin());
//    }


//    len = 0;
//    auto vertexTex_buf = oldBuffer.getVertexTextureCoordinateArray(len);
//    if(len > 0 && len <= 2 * vertex_len )
//    {
//        this->m_vertexTextureCoordinates.resize(len * 3);

//        std::copy(  vertexTex_buf.get(),
//                    vertexTex_buf.get() + len*3,
//                    this->m_vertexTextureCoordinates.begin());
//    }


//    len = 0;
//    auto faceInd_buf = oldBuffer.getFaceArray(len);
//    if(len > 0)
//    {

//        m_faceIndices.resize(len * 3);
//        std::copy(  faceInd_buf.get(),
//                    faceInd_buf.get() + len*3,
//                    m_faceIndices.begin());
//    }


//    len = 0;
//    auto faceMaterialInd_buf = oldBuffer.getFaceMaterialIndexArray(len);
//    if(len > 0)
//    {
//        m_faceMaterialIndices.resize(len);
//        std::copy(  faceMaterialInd_buf.get(),
//                    faceMaterialInd_buf.get() + len,
//                    m_faceMaterialIndices.begin());
//    }


//    len = 0;
//    auto material_buf = oldBuffer.getMaterialArray(len);
//    if(len > 0)
//    {
//        m_materials.resize(len);

//        for(size_t i=0; i<len; i++)
//        {
//            Material m;
//            if(material_buf[i]->texture_index >= 0)
//            {
//                m.m_texture = TextureHandle(material_buf[i]->texture_index);
//            }else{
//                m.m_texture = boost::none;
//            }

//            m.m_color = Rgb8Color();
//            m.m_color.get()[0] = static_cast<uint8_t>(material_buf[i]->r);
//            m.m_color.get()[1] = static_cast<uint8_t>(material_buf[i]->g);
//            m.m_color.get()[2] = static_cast<uint8_t>(material_buf[i]->b);

//            m_materials[i] = m;
//        }

//    }

//    // TODO: test this
//    len = 0;
//    auto texture_buf = oldBuffer.getTextureArray(len);
//    if(len > 0)
//    {
//        m_textures.resize(len);
//        // TODO old texture array to new
//        for(size_t i=0; i<len; i++)
//        {
//            m_textures[i] = Texture<BaseVecT>(i, texture_buf[i]);
//        }
//    }

//    // TODO: this vectors???
//    // m_clusterMaterialIndices
//    // m_clusterFaceIndices

//}


vector<float> MeshBuffer::getVertices()
{
    return m_vertices;
}


vector<unsigned char> MeshBuffer::getVertexColors()
{
    return m_vertexColors;
}


vector<float> MeshBuffer::getVertexConfidences()
{
    return m_vertexConfidences;
}


vector<float> MeshBuffer::getVertexIntensities()
{
    return m_vertexIntensities;
}


vector<float> MeshBuffer::getVertexNormals()
{
    return m_vertexNormals;
}


vector<float> MeshBuffer::getVertexTextureCoordinates()
{
    return m_vertexTextureCoordinates;
}


vector<unsigned int> MeshBuffer::getFaceIndices()
{
    return m_faceIndices;
}


vector<unsigned int> MeshBuffer::getFaceMaterialIndices()
{
    return m_faceMaterialIndices;
}



vector<unsigned int> MeshBuffer::getClusterMaterialIndices()
{
    return m_clusterMaterialIndices;
}


vector<Material> MeshBuffer::getMaterials()
{
    return m_materials;
}


vector<Texture> MeshBuffer::getTextures()
{
    return m_textures;
}


vector<vector<unsigned int>> MeshBuffer::getClusterFaceIndices()
{
    return m_clusterFaceIndices;
}


void MeshBuffer::setVertices(vector<float> v)
{
    m_vertices = v;
}


void MeshBuffer::setVertexColors(vector<unsigned char> v)
{
    m_vertexColors = v;
}


void MeshBuffer::setVertexConfidences(vector<float> v)
{
    m_vertexConfidences = v;
}


void MeshBuffer::setVertexIntensities(vector<float> v)
{
    m_vertexIntensities = v;
}


void MeshBuffer::setVertexNormals(vector<float> v)
{
    m_vertexNormals = v;
}


void MeshBuffer::setVertexTextureCoordinates(vector<float> v)
{
    m_vertexTextureCoordinates = v;
}


void MeshBuffer::setFaceIndices(vector<unsigned int> v)
{
    m_faceIndices = v;
}



void MeshBuffer::setFaceMaterialIndices(vector<unsigned int> v)
{
    m_faceMaterialIndices = v;
}



void MeshBuffer::setClusterMaterialIndices(vector<unsigned int> v)
{
    m_clusterMaterialIndices = v;
}


void MeshBuffer::setMaterials(vector<Material> v)
{
    m_materials = v;
}


void MeshBuffer::setTextures(vector<Texture> v)
{
    m_textures = v;
}


void MeshBuffer::setClusterFaceIndices(vector<vector<unsigned int>> v)
{
    m_clusterFaceIndices = v;
}

//
//boost::shared_ptr<lvr::MeshBuffer> MeshBuffer::toOldBuffer()
//{
//    auto buffer = boost::make_shared<lvr::MeshBuffer>();
//    buffer->setVertexArray(m_vertices);
//    if (m_vertexConfidences.size() > 0)
//    {
//        buffer->setVertexConfidenceArray(m_vertexConfidences);
//    }
//    if (m_vertexIntensities.size() > 0)
//    {
//        buffer->setVertexIntensityArray(m_vertexIntensities);
//    }
//    buffer->setVertexNormalArray(m_vertexNormals);
//    buffer->setVertexColorArray(m_vertexColors);
//    buffer->setFaceArray(m_faceIndices);
//    buffer->setVertexTextureCoordinateArray(m_vertexTextureCoordinates);
//    buffer->setFaceMaterialIndexArray(m_faceMaterialIndices);
//    // TODO: für materials und texturen muss die untere methode benutzt werden. bessere lösung finden!
//    boost::shared_array<lvr::Material*> materials( new lvr::Material*[ m_materials.size() ] );

//    for (size_t i=0; i<m_materials.size(); i++)
//    {

//        lvr::Material* m = new lvr::Material;
//        if (m_materials[i].m_color)
//        {
//            m->r = static_cast<unsigned char>(m_materials[i].m_color.get()[0]);
//            m->g = static_cast<unsigned char>(m_materials[i].m_color.get()[1]);
//            m->b = static_cast<unsigned char>(m_materials[i].m_color.get()[2]);
//        }
//        else
//        {
//            m->r = 255;
//            m->g = 255;
//            m->b = 255;
//        }
//        if (m_materials[i].m_texture)
//        {
//            int textureIndex = m_materials[i].m_texture.get().idx();
//            m->texture_index = textureIndex;
//        }
//        else
//        {
//            m->texture_index = -1;
//        }
//        materials[i] = m;
//    }
//    buffer->setMaterialArray(materials, m_materials.size());
//    return buffer;
//}

//
//boost::shared_ptr<lvr::MeshBuffer> MeshBuffer::toOldBuffer(MaterializerResult<BaseVecT> materializerResult)
//{
//    auto buffer = toOldBuffer();
//    vector<lvr::Material*> materials;
//    for (auto material : m_materials)
//    {
//        lvr::Material* m = new lvr::Material; // TODO memory leak
//        if (material.m_color)
//        {
//            m->r = material.m_color.get()[0];
//            m->g = material.m_color.get()[1];
//            m->b = material.m_color.get()[2];
//        }
//        else
//        {
//            m->r = 255;
//            m->g = 255;
//            m->b = 255;
//        }
//        if (material.m_texture && materializerResult.m_textures)
//        {
//            int textureIndex = materializerResult.m_textures.get().get(material.m_texture.get()).get().m_index;
//            m->texture_index = textureIndex;
//        }
//        else
//        {
//            m->texture_index = -1;
//        }
//        materials.push_back(m);
//    }
//    buffer->setMaterialArray(materials);

//    vector<GlTexture*> textures;
//    for (auto texture : m_textures)
//    {
//        GlTexture* t = new GlTexture();
//        t->m_texIndex = texture.m_index;
//        t->m_height = texture.m_height;
//        t->m_width = texture.m_width;
//        t->m_pixels = texture.m_data;

//        textures.push_back(t);
//    }
//    buffer->setTextureArray(textures);

//    return buffer;
//}


} /* namespace lvr2 */
