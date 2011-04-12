/*
 * TriangleMesh.h
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef TRIANGLEMESH_H_
#define TRIANGLEMESH_H_

#include "BaseMesh.hpp"

#include <vector>
#include <list>
#include <cassert>
using namespace std;

namespace lssr
{

template<typename VertexType, typename IndexType>
class TriangleMesh : public BaseMesh<VertexType, IndexType>{
public:

	TriangleMesh();
	TriangleMesh(const TriangleMesh &other);

	virtual void addTriangle(IndexType v0, IndexType v1, IndexType v2);
	virtual void addVertex(VertexType v){ m_vertices.push_back(v);};
	virtual void addNormal(VertexType n) {m_normals.push_back(n);};


	virtual ~TriangleMesh();

	VertexType   getVertex(IndexType index);
	VertexType   getNormal(IndexType index);

	void save(string filename);
	virtual void finalize();

protected:

	float*						  m_normalBuffer;

	vector<VertexType>            m_normals;
	vector<VertexType>            m_vertices;
	list<IndexType>               m_triangles;

};

} // namepsace lssr

#include "TriangleMesh.tcc"

#endif /* TRIANGLEMESH_H_ */
