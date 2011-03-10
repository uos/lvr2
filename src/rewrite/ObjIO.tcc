/*
 * ObjIO.cpp
 *
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType, typename IndexType>
ObjIO<CoordType, IndexType>::ObjIO()
{
    m_normals       = 0;
    m_vertices      = 0;
    m_indices       = 0;
    m_vertexCount   = 0;
    m_faceCount     = 0;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setVertexArray(CoordType* vertices, size_t c)
{
    m_vertices = vertices;
    m_vertexCount = c;
}


template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setNormalArray(CoordType* normals, size_t c)
{
    m_normals = normals;
    m_vertexCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setIndexArray(IndexType* indices, size_t c)
{
    m_indices = indices;
    m_faceCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::write(string filename)
{
    ofstream out(filename.c_str());

    if(out.good())
    {

        for(size_t i = 0; i < m_vertexCount; i++)
        {
            IndexType index = i * 3;
            out << "v " << m_vertices[index] << " "
                        << m_vertices[index + 1] << " "
                        << m_vertices[index + 2] << " " << endl;
        }


        for(size_t i = 0; i < m_vertexCount; i++)
        {
            IndexType index = i * 3;
            out << "vn " << m_normals[index] << " "
                         << m_normals[index + 1] << " "
                         << m_normals[index + 2] << " " << endl;
        }

        for(size_t i = 0; i < m_faceCount; i++)
        {
            IndexType index = 3 * i;
            // Calculate the buffer positions for the three
            // triangle vertices (each vertex has 3 coordinates)
            IndexType v1 = m_indices[index];
            IndexType v2 = m_indices[index + 1];
            IndexType v3 = m_indices[index + 2];

            out << "f " << v1 + 1 << "//" << v1 + 1 << " "
                        << v2 + 1 << "//" << v2 + 1 << " "
                        << v3 + 1 << "//" << v3 + 1 << endl;
        }

    }
    else
    {

    }

}

}
