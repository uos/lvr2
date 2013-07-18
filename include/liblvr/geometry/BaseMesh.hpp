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
 * BaseMesh.hpp
 *
 *  @date 03.02.2011
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef BASEMESH_H_
#define BASEMESH_H_

#include "io/Model.hpp"
#include <vector>

namespace lvr
{

/**
 * @brief     Abstract interface class for dynamic triangle meshes.
 *             The surface reconstruction algorithm can handle all
 *            all data structures that allow sequential insertion
 *            all of indexed triangles.
 */
template<typename VertexT, typename NormalT>
class BaseMesh
{
    public:

        BaseMesh();

        /**
         * @brief    This method should be called every time
         *           a new vertex is created.
         *
         * @param v  A supported vertex type. All used vertex types
         *           must support []-access.
         */
        virtual void addVertex(VertexT v) = 0;

        /**
         * @brief    This method should be called every time
         *           a new vertex is created to ensure that vertex
         *           and normal buffer always have the same size
         *
         * @param n  A supported vertex type. All used vertex types
         *           must support []-access.
         */
        virtual void addNormal(NormalT n) = 0;

        /**
         * @brief    Insert a new triangle into the mesh
         *
         * @param a  The first vertex of the triangle
         * @param b  The second vertex of the triangle
         * @param c  The third vertex of the triangle
         */
        virtual void addTriangle(uint a, uint b, uint c) = 0;

    	/**
    	 * @brief	Flip the edge between vertex index v1 and v2
    	 *
    	 * @param	v1	The index of the first vertex
    	 * @param	v2	The index of the second vertex
    	 */
    	virtual void flipEdge(uint v1, uint v2) = 0;

        /**
         * @brief    Finalizes a mesh, i.e. converts the template based buffers
         *           to OpenGL compatible buffers
         */
        virtual void finalize() = 0;


        /**
         * @brief    Creates a buffered mesh from the given file.
         *
         * @param    filename
         */
        //virtual void load( string filename );


        MeshBufferPtr meshBuffer()
        {
            return m_meshBuffer;
        }

    protected:

        /// True if mesh is finalized
        bool            m_finalized;

        MeshBufferPtr   m_meshBuffer;
};
}

#include "BaseMesh.tcc"

#endif /* BASEMESH_H_ */
