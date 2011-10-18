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
 * StlIO.h
 *
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef STLIO_H_
#define STLIO_H_

#include <string>
using std::string;

#include "Vertex.hpp"
#include "Normal.hpp"

namespace lssr
{

template class Normal<float>;

/***
 * @brief An Import / Export interface for ASCII STL files
 */

/// TODO: Write import for stl files.
template<typename CoordType, typename IndexType>
class StlIO
{
public:
    StlIO();

    void write(string filename);
    void setVertexArray(CoordType* array, size_t count);
    void setNormalArray(CoordType* array, size_t count);
    void setIndexArray(IndexType* array, size_t count);

private:
    CoordType*              m_vertices;
    CoordType*              m_normals;
    IndexType*              m_indices;

    size_t                  m_faceCount;
    size_t                  m_vertexCount;

};


} // namespace lssr

#include "StlIO.tcc"

#endif /* STLIO_H_ */
