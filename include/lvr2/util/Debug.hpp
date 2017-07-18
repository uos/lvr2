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
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USAc
 */


/*
 * HalfEdgeMesh.hpp
 *
 *  @date 20.07.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <boost/iostreams/device/null.hpp>
#include <fstream>


#ifndef LVR2_UTIL_DEBUG
#define LVR2_UTIL_DEBUG

namespace lvr2
{

inline std::ostream& dout()
{
    // To have a "null" ostream, it's apparently a good idea to use an
    // unopened file, as no one ever checks the error state or sth. Source:
    //
    // https://stackoverflow.com/a/8244052/2408867
    static bool isDebug = getenv("LVR_MESH_DEBUG") != nullptr;
    static std::ofstream unopenedFile;

    return isDebug ? cout : unopenedFile;
}

#ifdef NDEBUG
#define DOINDEBUG(...) ;
#else
#define DOINDEBUG(...) __VA_ARGS__
#endif

} // namespace lvr2

#endif /* LVR2_UTIL_DEBUG */
