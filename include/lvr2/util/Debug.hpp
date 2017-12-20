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
 * Collection of functions to debug mesh generation.
 *
 * @date 18.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_UTIL_DEBUG_H_
#define LVR2_UTIL_DEBUG_H_

#include <boost/iostreams/device/null.hpp>
#include <array>
#include <fstream>
#include <vector>

using std::array;
using std::string;
using std::vector;

#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/algorithm/ClusterPainter.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>

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

/**
 * @brief Write a mesh to the given filename and color it with the given color
 */
template<typename BaseVecT>
void writeDebugMesh(
    const BaseMesh<BaseVecT>& mesh,
    string filename = "debug.ply",
    Rgb8Color color = {255, 0, 0}
);

/**
 * @brief Returns all handles of duplicate vertices from the given mesh
 *
 * The equality of two points is check via Point::operator==().
 *
 * @return duplicate vertex handles. The vertex handles for each duplicate point are stored in a seperate vector. The
 *         return value is a vector which consists of these vectors. In other words: each vector in the result vector
 *         is a set of vertex handles, which points to points with the same coordinates.
 */
template<typename BaseVecT>
vector<vector<VertexHandle>> getDuplicateVertices(const BaseMesh<BaseVecT>& mesh);

/**
 * @brief Writes a mesh to the given filename and colors it with the following meaning:
 *  - conntectedColor: connected mesh (vertices of edges with 2 connected faces)
 *  - contourColor: contour edges (vertices of edges with 1 connected face)
 *  - bugColor: edges with neither 1 or 2 conntected faces
 */
template<typename BaseVecT>
void writeDebugContourMesh(
    const BaseMesh<BaseVecT>& mesh,
    string filename = "debug-contours.ply",
    Rgb8Color connectedColor = {0, 255, 0},
    Rgb8Color contourColor = {0, 0, 255},
    Rgb8Color bugColor = {255, 0, 0}
);

#ifdef NDEBUG
#define DOINDEBUG(...) ;
#else
#define DOINDEBUG(...) __VA_ARGS__
#endif

} // namespace lvr2

#include <lvr2/util/Debug.tcc>

#endif /* LVR2_UTIL_DEBUG_H_ */
