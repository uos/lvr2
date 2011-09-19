#include "BaseMesh.hpp"

namespace lssr
{

template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::save( string filename ) {

	PLYIO ply_writer;

	// Set data arrays
	ply_writer.setVertexArray( this->m_vertexBuffer, m_nVertices );
	ply_writer.setIndexArray( this->m_indexBuffer, m_nFaces );
	if ( this->m_colorBuffer ) {
		ply_writer.setColorArray( this->m_colorBuffer, this->m_nVertices );
	}

	// Save
	ply_writer.save( filename );
}

}
