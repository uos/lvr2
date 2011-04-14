/*
 * StlIO.cpp
 *
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType, typename IndexType>
StlIO<CoordType, IndexType>::StlIO()
{
    m_normals       = 0;
    m_vertices      = 0;
    m_indices       = 0;
    m_vertexCount   = 0;
    m_faceCount     = 0;
}

template<typename CoordType, typename IndexType>
void StlIO<CoordType, IndexType>::setVertexArray(CoordType* vertices, size_t c)
{
    m_vertices = vertices;
    m_vertexCount = c;
}


template<typename CoordType, typename IndexType>
void StlIO<CoordType, IndexType>::setNormalArray(CoordType* normals, size_t c)
{
    m_normals = normals;
    m_vertexCount = c;
}

template<typename CoordType, typename IndexType>
void StlIO<CoordType, IndexType>::setIndexArray(IndexType* indices, size_t c)
{
    m_indices = indices;
    m_faceCount = c;
}

template<typename CoordType, typename IndexType>
void StlIO<CoordType, IndexType>::write(string filename)
{
    ofstream out(filename.c_str());
    if(out.good())
    {
        out << "solid mesh" << endl;
        for(size_t i = 0; i < m_faceCount; i++)
        {
            // Calculate base index of current face
            IndexType index = 3 * i;

            // Calculate the buffer positions for the three
            // triangle vertices (each vertex has 3 coordinates)
            IndexType v1 = m_indices[index]     * 3;
            IndexType v2 = m_indices[index + 1] * 3;
            IndexType v3 = m_indices[index + 2] * 3;

            // Interpolate normal for the current face
            Vertex<CoordType>    v(m_normals[v1], m_normals[v1 + 1], m_normals[v1 + 2]);
            v += Vertex<CoordType>(m_normals[v2], m_normals[v2 + 1], m_normals[v2 + 2]);
            v += Vertex<CoordType>(m_normals[v3], m_normals[v3 + 1], m_normals[v3 + 2]);

            Normal<float> n(v);

            // Write triangle definitions
            out << " facet normal " << n[0] << " " << n[1] << " " << n[2] << endl;
            out << "  outer loop" << endl;
            out << "   vertex " << m_vertices[v1] << " "
                                << m_vertices[v1 + 1] << " "
                                << m_vertices[v1 + 2] << endl;
            out << "   vertex " << m_vertices[v2] << " "
                                << m_vertices[v2 + 1] << " "
                                << m_vertices[v2 + 2] << endl;
            out << "   vertex " << m_vertices[v3] << " "
                                << m_vertices[v3 + 1] << " "
                                << m_vertices[v3 + 2] << endl;
            out << "  endloop" << endl;
            out << " endfacet" << endl;

        }
        out << "endsolid mesh" << endl;
    }
    else
    {
        cout << "StlIO: Could not open file '" << filename << "'." << endl;
    }
}

} // namespace lssr
