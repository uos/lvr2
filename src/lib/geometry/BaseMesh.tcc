#include "BaseMesh.hpp"
#include "../io/ObjIO.hpp"

namespace lssr
{

template<typename VertexT, typename IndexType>
BaseMesh<VertexT, IndexType>::BaseMesh()
{
	m_finalized = false;
	m_vertexBuffer = 0;
	m_normalBuffer = 0;
	m_colorBuffer = 0;
	m_textureCoordBuffer = 0;
	m_textureIndexBuffer = 0;
	m_indexBuffer = 0;
	m_textureBuffer = 0;
	m_nVertices = 0;
	m_nTextures = 0;
	m_nFaces = 0;
}

template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::save( string filename ) {

	PLYIO ply_writer;

	// Set data arrays
	ply_writer.setVertexArray( this->m_vertexBuffer, m_nVertices );
	ply_writer.setFaceArray( this->m_indexBuffer, m_nFaces );
	if ( this->m_colorBuffer ) {
		ply_writer.setVertexColorArray( this->m_colorBuffer, this->m_nVertices );
	}

	// Save
	ply_writer.save( filename );
}

template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::saveObj(string filename)
{
	ObjIO<float, uint> obj_writer;

	// Set data arrays
	obj_writer.setVertexArray(this->m_vertexBuffer, m_nVertices);
	obj_writer.setIndexArray(this->m_indexBuffer, m_nFaces);
	obj_writer.setNormalArray(this->m_normalBuffer, m_nVertices);
	obj_writer.setTextureCoords(this->m_textureCoordBuffer, m_nVertices);
	obj_writer.setTextureIndices(this->m_textureIndexBuffer, m_nVertices);
	obj_writer.setTextures(this->m_textureBuffer, m_nTextures);
	obj_writer.setColors(this->m_colorBuffer, m_nTextures);

	// Save
	obj_writer.write(filename);
}

}
