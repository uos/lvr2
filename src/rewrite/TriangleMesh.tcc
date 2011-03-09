/*
 * TriangleMesh<VertexType, IndexType>.cpp
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename VertexType, typename IndexType>
TriangleMesh<VertexType, IndexType>::TriangleMesh()
{

}

template<typename VertexType, typename IndexType>
TriangleMesh<VertexType, IndexType>::~TriangleMesh<VertexType, IndexType>()
{
    m_normals.clear();
    m_vertices.clear();
    m_triangles.clear();
}

template<typename VertexType, typename IndexType>
void TriangleMesh<VertexType, IndexType>::addTriangle(IndexType v0, IndexType v1, IndexType v2)
{
	m_triangles.push_back(v0);
	m_triangles.push_back(v1);
	m_triangles.push_back(v2);
}

template<typename VertexType, typename IndexType>
VertexType TriangleMesh<VertexType, IndexType>::getVertex(IndexType index)
{
	assert(index < (int) m_vertices.size());
	return m_vertices[index];
}

template<typename VertexType, typename IndexType>
VertexType TriangleMesh<VertexType, IndexType>::getNormal(IndexType index)
{
	assert(index < (int)m_normals.size());
	return m_normals[index];
}

/// TODO: Re-Integrate normal interpolation
//template<typename VertexType, typename IndexType>
//void TriangleMesh<VertexType, IndexType>::interpolateNormal(Normal n, size_t index){
//
//	if(index < normal_buffer.size()){
//
//		Normal normal = normal_buffer[index];
//		normal += n;
//		normal.normalize();
//
//		normal_buffer[index] = normal;
//
//	} else {
//		cout << "WARNING: TriangleMesh<VertexType, IndexType>: Normal index out of range: " << index << endl;
//	}
//
//}


}
