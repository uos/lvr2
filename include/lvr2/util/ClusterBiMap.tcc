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