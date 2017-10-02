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
 * Cluster.hpp
 *
 *  @date 17.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_UTIL_CLUSTER_H_
#define LVR2_UTIL_CLUSTER_H_

#include <vector>

using std::vector;

namespace lvr2
{

/**
 * @brief Represents a group of handles, which are somehow connected.
 * @tparam HandleT Type of handles, which are connected by this cluster.
 */
template <typename HandleT>
struct Cluster
{
public:
    size_t size() { return handles.size(); }
    decltype(auto) begin() const { return handles.begin(); }
    decltype(auto) end() const { return handles.end(); }

    vector<HandleT> handles;
};

} // namespace lvr2

#endif /* LVR2_UTIL_CLUSTER_H_ */