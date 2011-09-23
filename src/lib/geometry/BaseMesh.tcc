#include "BaseMesh.hpp"
#include "../io/ObjIO.hpp"

namespace lssr
{

template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::save( string filename ) {

	PLYIO ply_writer;

	// Set data arrays
	ply_writer.setVertexArray( this->m_vertexBuffer, m_nVertices );
	ply_writer.setIndexArray( this->m_indexBuffer, m_nFaces );
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

	// Save
	obj_writer.write(filename);
}

}
