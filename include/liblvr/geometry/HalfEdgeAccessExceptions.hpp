/* Copyright (C) 2013 Uni Osnabrück
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
 * HalfEdgeAccessExceptions.hpp
 *
 *  @date Oct 8, 2013
 *  @author Thomas Wiemann
 *  @author Dominik Feldschnieders
 */
#ifndef HALFEDGEACCESSEXCEPTIONS_HPP_
#define HALFEDGEACCESSEXCEPTIONS_HPP_

#include <iostream>
#include <string>
using std::cout;
using std::endl;
using std::ostream;
using std::string;

namespace lvr
{

/**
 * @brief Marker base class for access exceptions in half edge meshes
 */
class HalfEdgeAccessException
{
public:
    HalfEdgeAccessException(string element = "unknown") : m_element(element){};
    virtual string what() const
    {
        return "General access exception in half edge mesh occurred in " + m_element;
    }
    virtual ~HalfEdgeAccessException() {};

protected:
    string m_element;
};

class HalfEdgeException : public HalfEdgeAccessException
{
public:
    HalfEdgeException(string element = "unknown") : HalfEdgeAccessException(element) {};
    virtual string what() const
    {
        return "Trying to acces non-existing half edge at " + m_element;
    }
    virtual ~HalfEdgeException() {};
};

class HalfEdgeFaceException : public HalfEdgeAccessException
{
public:
    HalfEdgeFaceException(string element = "unknown") : HalfEdgeAccessException(element) {};
    virtual string what() const
    {
        return "Trying to acces non-existing half edge face";
    }
    virtual ~HalfEdgeFaceException() {};
};

class HalfEdgeVertexException : public HalfEdgeAccessException
{
public:
    HalfEdgeVertexException(string element = "unknown") : HalfEdgeAccessException(element) {};
    virtual string what() const
    {
        return "Trying to acces non-existing half edge vertex at " + m_element;
    }
    virtual ~HalfEdgeVertexException() {};
};

/**
 * @brief General output operator for exception messages
 */
ostream& operator<<(ostream& os, const HalfEdgeAccessException e);


}


#endif /* HALFEDGEACCESSEXCEPTIONS_HPP_ */
