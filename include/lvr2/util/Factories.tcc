/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 *  Factories.tcc
 */

#include <algorithm>

#include "lvr2/reconstruction/SearchTree.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/util/Panic.hpp"

namespace lvr2
{


template <typename BaseVecT>
SearchTreePtr<BaseVecT> getSearchTree(string name, PointBufferPtr buffer)
{
    // Transform name to lowercase (only works for ASCII, but this is not a
    // problem in our case).
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

#ifdef LVR2_USE_PCL
    if(name == "pcl")
    {
        // TODO2
        // this->m_searchTree = search_tree::Ptr( new SearchTreeFlannPCL<BaseVecT>(buffer, this->m_numPoints, kn, ki, kd) );
    }
#endif

#ifdef LVR2_USE_STANN
    if(name == "stann")
    {
        // TODO2
        // this->m_searchTree = std::make_shared<SearchTreeStann<BaseVecT>>(buffer, this->m_numPoints, kn, ki, kd);
    }
#endif

#ifdef LVR2_USE_NABO
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
