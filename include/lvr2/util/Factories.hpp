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
 *  Factories.hpp
 *
 */

#ifndef LVR2_UTIL_FACTORIES_H_
#define LVR2_UTIL_FACTORIES_H_

#include <algorithm>
#include <memory>
#include <string>

#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/io/PointBuffer.hpp>

using std::string;


namespace lvr2
{

/**
 * @brief Returns the search tree implementation specified by `name`.
 *
 * If `name` doesn't contain a valid implementation, `nullptr` is returned.
 * Currently, the only supported implementation is "flann".
 */
template <typename BaseVecT>
SearchTreePtr<BaseVecT> getSearchTree(string name, PointBufferPtr<BaseVecT> buffer);

} // namespace lvr2

#include <lvr2/util/Factories.tcc>

#endif // LVR2_UTIL_FACTORIES_H_
