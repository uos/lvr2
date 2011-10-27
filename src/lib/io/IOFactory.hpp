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
 * IOFactory.h
 *
 *  @date 24.08.2011
 *  @author Thomas Wiemann
 */

#ifndef IOFACTORY_H_
#define IOFACTORY_H_

#include "BaseIO.hpp"
#include "MeshLoader.hpp"
#include "PointLoader.hpp"

#include <string>
using std::string;

namespace lssr
{

/**
 * @brief Factory class extract point cloud and mesh information
 *        from supported file formats. The instantiated MeshLoader
 *        and PointLoader instances are persistent, i.e. they will
 *        not be freed in the destructor of this class to prevent
 *        side effects.
 */
class IOFactory
{
public:

    /**
     * @brief Ctor.
     * @param filename  Full path to the file to parse.
     */
    IOFactory(string filename);

    /**
     * @brief Dtor.
     */
    virtual ~IOFactory() {}

    /**
     * @brief   Returns a point to a @ref{MeshLoader} instance or
     *          null if the parsed file does not contain mesh data.
     */
    MeshLoader* getMeshLoader() { return m_meshLoader;}

    /**
     * @brief   Returns a pointer to a @ref{PointLoader} instance or
     *          null if the parsed file does not contain point cloud
     *          data
     * @return
     */
    PointLoader* getPointLoader() { return m_pointLoader;}

private:

    /// The point loader associated with the given file
    PointLoader*    m_pointLoader;

    /// The mesh loader associated with the given file
    MeshLoader*     m_meshLoader;

    /// A BaseIO pointer for reading and writing
    BaseIO*         m_baseIO;
};

} // namespace lssr

#endif /* IOFACTORY_H_ */
