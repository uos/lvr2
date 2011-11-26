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


/*
 * BaseMesh.tcc
 *
 *  @date 03.02.2011
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#include "BaseMesh.hpp"
#include "io/ModelFactory.hpp"

namespace lssr
{

template<typename VertexT, typename IndexType>
BaseMesh<VertexT, IndexType>::BaseMesh()
{
	m_finalized = false;
	m_meshBuffer.reset();
}

template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::save( string filename )
{

    if ( m_meshBuffer )
    {
        ModelPtr m( new Model( this->m_meshBuffer ) );
        ModelFactory::saveModel( m, filename );
    }

}


template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::load( string filename )
{


}


}
