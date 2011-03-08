/*
 * FastBox.cpp
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType, typename IndexType>
FastBox<CoordType, IndexType>::FastBox(Vertex<CoordType> &center)
{
    // Init members
    for(int i = 0; i < 12; i++) m_intersections[i] = 0;
    m_center = center;
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::setVertex(int index, IndexType nb)
{
    m_vertices[index] = nb;
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::setNeighbor(int index, FastBox<CoordType, IndexType>* nb)
{
    m_neighbors[index] = nb;
}


template<typename CoordType, typename IndexType>
FastBox<CoordType, IndexType>* FastBox<CoordType, IndexType>::getNeighbor(int index)
{
    return m_neighbors[index];
}

template<typename CoordType, typename IndexType>
IndexType FastBox<CoordType, IndexType>::getVertex(int index)
{
    return m_vertices[index];
}


template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::calcVertices(vector<Vertex<CoordType> > &vertices)
{
}

} // namespace lssr
