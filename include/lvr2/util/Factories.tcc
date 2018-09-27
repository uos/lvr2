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
 *  Factories.tcc
 */

#include <algorithm>

#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/io/PointBuffer2.hpp>
#include <lvr2/util/Panic.hpp>

namespace lvr2
{


template <typename BaseVecT>
SearchTreePtr<BaseVecT> getSearchTree(string name, PointBuffer2Ptr buffer)
{
    // Transform name to lowercase (only works for ASCII, but this is not a
    // problem in our case).
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

#ifdef LVR_USE_PCL
    if(name == "pcl")
    {
        // TODO2
        // this->m_searchTree = search_tree::Ptr( new SearchTreeFlannPCL<BaseVecT>(buffer, this->m_numPoints, kn, ki, kd) );
    }
#endif

#ifdef LVR_USE_STANN
    if(name == "stann")
    {
        // TODO2
        // this->m_searchTree = std::make_shared<SearchTreeStann<BaseVecT>>(buffer, this->m_numPoints, kn, ki, kd);
    }
#endif

#ifdef LVR_USE_NABO
    if(name == "nabo")
    {
        // TODO2
        // this->m_searchTree = std::make_shared<SearchTreeNabo<BaseVecT>>(buffer, this->m_numPoints, kn, ki, kd);
    }
#endif

    if(name == "nanoflann")
    {
        // TODO2
        // this->m_searchTree = std::make_shared<SearchTreeNanoflann<BaseVecT>>(buffer, this->m_numPoints, kn, ki, kd);
    }

    if(name == "flann")
    {
        return std::make_shared<SearchTreeFlann<BaseVecT>>(buffer);
    }

    return nullptr;
}

} // namespace lvr2
