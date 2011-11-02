/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


#include "BaseMesh.hpp"
#include "io/ObjIO.hpp"
#include "io/PLYIO.hpp"

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
	m_regionSizeBuffer = 0;
	m_nRegions = 0;
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
MeshLoader* BaseMesh<VertexT, IndexType>::getMeshLoader()
{
    MeshLoader* l = 0;
    if(m_finalized)
    {
        // Create and setup new loader object
        l = new MeshLoader;
        l->setVertexArray(m_vertexBuffer, m_nVertices);
        l->setFaceArray(m_indexBuffer, m_nFaces);
        l->setVertexNormalArray(m_normalBuffer, m_nVertices);
        l->setVertexColorArray(m_colorBuffer, m_nVertices);
    }
    return l;
}

}
