/*Copyright (C) 2011 Uni Osnabrück
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
#include "HalfEdgeAccessExceptions.hpp"

namespace lvr
{

template<typename HVertexT, typename FaceT>
bool  HalfEdge<HVertexT, FaceT>::isBorderEdge()
{
    try
    {
        pair()->face();
    }
    catch(HalfEdgeAccessException& e)
    {
        // Current edge is a border edge
        return true;
    }

    try
    {
        return (pair()->face()->m_region != face()->m_region);
    }
    catch(HalfEdgeAccessException &e)
    {
        cout << "HalfEdge::isBorderEdge(): " << e.what() << endl;
        return false;
    }

    return false;
}


template<typename HVertexT, typename FaceT>
bool HalfEdge<HVertexT, FaceT>::hasNeighborFace()
{
    try
    {
        pair()->face();
    }
    catch(HalfEdgeAccessException &e)
    {
        // Face has no neighbor if no pair edge
        // or pair->face exists
        return false;
    }
    return true;
}

template<typename HVertexT, typename FaceT>
bool HalfEdge<HVertexT, FaceT>::hasPair()
{
    try
    {
        pair();
    }
    catch(HalfEdgeAccessException &e)
    {
        return false;
    }
    return true;
}

template<typename HVertexT, typename FaceT>
bool HalfEdge<HVertexT, FaceT>::hasFace()
{
    try
    {
        face();
    }
    catch(HalfEdgeAccessException &e)
    {
        return false;
    }
    return true;
}

template<typename HVertexT, typename FaceT>
HalfEdge<HVertexT, FaceT>* HalfEdge<HVertexT, FaceT>::next()
{
    if(n != 0)
    {
        return n;
    }
    else
    {

        throw HalfEdgeException("next");
    }
}

template<typename HVertexT, typename FaceT>
HalfEdge<HVertexT, FaceT>*  HalfEdge<HVertexT, FaceT>::pair()
{
    if(p != 0)
    {
        return p;
    }
    else
    {
        throw HalfEdgeException("pair");
    }
}

template<typename HVertexT, typename FaceT>
HVertexT* HalfEdge<HVertexT, FaceT>::start()
{
    if(s != 0)
    {
        return s;
    }
    else
    {
        throw HalfEdgeVertexException("start");
    }
}

template<typename HVertexT, typename FaceT>
HVertexT* HalfEdge<HVertexT, FaceT>::end()
{
    if(e != 0)
    {
        return e;
    }
    else
    {
        throw HalfEdgeVertexException("end");
    }
}

template<typename HVertexT, typename FaceT>
FaceT*  HalfEdge<HVertexT, FaceT>::face()
{
    if(f != 0)
    {
        return f;
    }
    else
    {
        throw HalfEdgeFaceException("face");
    }
}

}
