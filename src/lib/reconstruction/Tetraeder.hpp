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

/**
 * Tetraeder.h
 *
 *  @date 23.11.2011
 *  @author Thomas Wiemann
 */

#ifndef TETRAEDER_H_
#define TETRAEDER_H_

#include "QueryPoint.hpp"

namespace lssr
{

template<typename VertexT, typename NormalT>
class Tetraeder
{
public:
    Tetraeder(
            QueryPoint<VertexT> p1,
            QueryPoint<VertexT> p2,
            QueryPoint<VertexT> p3,
            QueryPoint<VertexT> p4,
            int index1,
            int index2,
            int index3,
            int index4);

    void getSurface(BaseMesh<VertexT, NormalT> &mesh);

    virtual ~Tetraeder();

private:

    float       calcIntersection(float x1, float x2, float d1, float d2);
    VertexT     calcIntersection(int v1, int v2);


    QueryPoint<VertexT>     m_queryPoints[4];
    VertexT                 m_intersections[5];
    int                     m_globalIndex;

};

} /* namespace lssr */

#include "Tetraeder.tcc"

#endif /* TETRAEDER_H_ */
