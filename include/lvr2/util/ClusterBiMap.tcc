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

/*
 * ClusterBiMap.tcc
 *
 *  @date 17.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <algorithm>

using std::remove;

namespace lvr2
{

template <typename HandleT>
Cluster<HandleT>& ClusterBiMap<HandleT>::getC(ClusterHandle clusterHandle)
{
    return m_cluster[clusterHandle];
}

template <typename HandleT>
const Cluster<HandleT>& ClusterBiMap<HandleT>::getCluster(ClusterHandle clusterHandle) const
{
    return m_cluster[clusterHandle];
}

template <typename HandleT>
const Cluster<HandleT>& ClusterBiMap<HandleT>::operator[](ClusterHandle clusterHandle) const
{
    return m_cluster[clusterHandle];
}

template <typename HandleT>
ClusterHandle ClusterBiMap<HandleT>::createCluster()
{
    ClusterHandle newHandle(m_cluster.size());
    m_cluster.push(Cluster<HandleT>());

    return newHandle;
}

template <typename HandleT>
void ClusterBiMap<HandleT>::removeCluster(ClusterHandle clusterHandle)
{
    auto cluster = getC(clusterHandle);

    // Substract number of handles in removed cluster from number of all handles in set
    m_numHandles -= cluster.handles.size();

    // Remove handles in cluster from cluster map
    for (auto handle: cluster.handles)
    {
        m_clusterMap.erase(handle);
    }

    // Remove cluster
    m_cluster.erase(clusterHandle);
}

template <typename HandleT>
ClusterHandle ClusterBiMap<HandleT>::addToCluster(ClusterHandle clusterHandle, HandleT handle)
{
    getC(clusterHandle).handles.push_back(handle);
    m_clusterMap.insert(handle, clusterHandle);

    ++m_numHandles;

    return clusterHandle;
}

template <typename HandleT>
ClusterHandle ClusterBiMap<HandleT>::removeFromCluster(ClusterHandle clusterHandle, HandleT handle)
{
    auto& handles = getC(clusterHandle).handles;
    handles.erase(remove(handles.begin(), handles.end(), handle), handles.end());
    m_clusterMap.erase(handle);

    --m_numHandles;

    return clusterHandle;
}

template <typename HandleT>
ClusterHandle ClusterBiMap<HandleT>::getClusterH(HandleT handle) const
{
    return m_clusterMap[handle];
}

template <typename HandleT>
OptionalClusterHandle ClusterBiMap<HandleT>::getClusterOf(HandleT handle) const
{
    auto maybe = m_clusterMap.get(handle);
    if (maybe)
    {
        return *maybe;
    }
    return OptionalClusterHandle();
}

template <typename HandleT>
size_t ClusterBiMap<HandleT>::numCluster() const
{
    return m_cluster.numUsed();
}

template <typename HandleT>
size_t ClusterBiMap<HandleT>::numHandles() const
{
    return m_numHandles;
}

template <typename HandleT>
void ClusterBiMap<HandleT>::reserve(size_t newCap)
{
    m_cluster.reserve(newCap);
    m_clusterMap.reserve(newCap);
}

template<typename HandleT>
ClusterBiMapIterator<HandleT>& ClusterBiMapIterator<HandleT>::operator++()
{
    ++m_iterator;
    return *this;
}

template<typename HandleT>
bool ClusterBiMapIterator<HandleT>::operator==(const ClusterBiMapIterator& other) const
{
    return m_iterator == other.m_iterator;
}

template<typename HandleT>
bool ClusterBiMapIterator<HandleT>::operator!=(const ClusterBiMapIterator& other) const
{
    return m_iterator != other.m_iterator;
}

template<typename HandleT>
ClusterHandle ClusterBiMapIterator<HandleT>::operator*() const
{
    return *m_iterator;
}

template <typename HandleT>
ClusterBiMapIterator<HandleT> ClusterBiMap<HandleT>::begin() const
{
    return m_cluster.begin();
}

template <typename HandleT>
ClusterBiMapIterator<HandleT> ClusterBiMap<HandleT>::end() const
{
    return m_cluster.end();
}

} // namespace lvr2
