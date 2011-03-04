/*
 * FastBox.cpp
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType, typename IndexType>
FastBox<CoordType, IndexType>::FastBox(Vertex<CoordType> &center, CoordType voxelsize)
{
    // Init members
    for(int i = 0; i < 12; i++) m_intersections[i] = 0;
    m_center = center;
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::calcVertices(vector<Vertex<CoordType> > &vertices)
{

}

} // namespace lssr
