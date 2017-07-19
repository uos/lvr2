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
 * ClusterSet.tcc
 *
 *  @date 17.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <algorithm>

using std::remove;

namespace lvr2
{

template <typename HandleT>
Cluster<HandleT>& ClusterSet<HandleT>::getC(ClusterHandle clusterHandle)
{
    return m_cluster[clusterHandle];
}

template <typename HandleT>
const Cluster<HandleT>& ClusterSet<HandleT>::getCluster(ClusterHandle clusterHandle) const
{
    return m_cluster[clusterHandle];
}

template <typename HandleT>
const Cluster<HandleT>& ClusterSet<HandleT>::operator[](ClusterHandle clusterHandle) const
{
    return m_cluster[clusterHandle];
}

template <typename HandleT>
ClusterHandle ClusterSet<HandleT>::createCluster()
{
    ClusterHandle newHandle(m_cluster.size());
    m_cluster.push_back(Cluster<HandleT>());

    return newHandle;
}

template <typename HandleT>
void ClusterSet<HandleT>::removeCluster(ClusterHandle clusterHandle)
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
ClusterHandle ClusterSet<HandleT>::addToCluster(ClusterHandle clusterHandle, HandleT handle)
{
    getC(clusterHandle).handles.push_back(handle);
    m_clusterMap.insert(handle, clusterHandle);

    ++m_numHandles;

    return clusterHandle;
}

template <typename HandleT>
ClusterHandle ClusterSet<HandleT>::removeFromCluster(ClusterHandle clusterHandle, HandleT handle)
{
    auto handles = getC(clusterHandle).handles;
    handles.erase(remove(handles.begin(), handles.end(), handle), handles.end());
    m_clusterMap.erase(handle);

    --m_numHandles;

    return clusterHandle;
}

template <typename HandleT>
ClusterHandle ClusterSet<HandleT>::getClusterH(HandleT handle) const
{
    return m_clusterMap[handle];
}

template <typename HandleT>
size_t ClusterSet<HandleT>::numCluster() const
{
    return m_cluster.sizeUsed();
}

template <typename HandleT>
size_t ClusterSet<HandleT>::numHandles() const
{
    return m_numHandles;
}

template <typename HandleT>
void ClusterSet<HandleT>::reserve(size_t newCap)
{
    m_cluster.reserve(newCap);
    m_clusterMap.reserve(newCap);
}

ClusterSetIterator& ClusterSetIterator::operator++()
{
    ++m_iterator;
    return *this;
}

bool ClusterSetIterator::operator==(const ClusterSetIterator& other) const
{
    return m_iterator == other.m_iterator;
}

bool ClusterSetIterator::operator!=(const ClusterSetIterator& other) const
{
    return m_iterator != other.m_iterator;
}

ClusterHandle ClusterSetIterator::operator*() const
{
    return *m_iterator;
}

template <typename HandleT>
ClusterSetIterator ClusterSet<HandleT>::begin() const
{
    return m_cluster.begin();
}

template <typename HandleT>
ClusterSetIterator ClusterSet<HandleT>::end() const
{
    return m_cluster.end();
}

} // namespace lvr2
