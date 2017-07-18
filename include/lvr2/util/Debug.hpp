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
 * Debug.hpp
 *
 *  @date 18.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_UTIL_DEBUG_H_
#define LVR2_UTIL_DEBUG_H_

#include <boost/iostreams/device/null.hpp>
#include <array>
#include <fstream>

using std::array;
using std::string;

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/algorithm/ClusterPainter.hpp>

namespace lvr2
{

// ==========================================================================
// Collection of functions to debug mesh generation.
// ==========================================================================
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

template<typename BaseVecT>
void writeDebugMesh(
    const BaseMesh<BaseVecT>& mesh,
    string filename = "debug.ply",
    ClusterPainter::Rgb8Color color = {255, 0, 0}
);

#ifdef NDEBUG
#define DOINDEBUG(...) ;
#else
#define DOINDEBUG(...) __VA_ARGS__
#endif

} // namespace lvr2

#include <lvr2/util/Debug.tcc>

#endif /* LVR2_UTIL_DEBUG_H_ */
