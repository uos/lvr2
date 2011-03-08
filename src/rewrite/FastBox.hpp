/*
 * FastBox.h
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef FastBox_H_
#define FastBox_H_

#include "Vertex.hpp"
#include <vector>

using std::vector;

namespace lssr
{

template<typename CoordType, typename IndexType>
class FastBox
{
public:
    FastBox(Vertex<CoordType> &center);
    virtual ~FastBox() {};

    static CoordType            m_voxelsize;

    void setVertex(int index,  IndexType value);
    void setNeighbor(int index, FastBox<CoordType, IndexType>* value);

    IndexType getVertex(int index);
    FastBox<CoordType, IndexType>*     getNeighbor(int index);

private:

    void calcVertices(vector<Vertex<CoordType> > &vertices);

    Vertex<CoordType>               m_center;
    IndexType                       m_vertices[8];
    IndexType                       m_intersections[12];
    FastBox<CoordType, IndexType>*  m_neighbors[27];

};

} // namespace lssr

#include "FastBox.tcc"

#endif /* FastBox_H_ */
