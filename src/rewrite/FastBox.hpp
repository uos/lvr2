/*
 * FastBox.h
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef FastBox_H_
#define FastBox_H_

#include "Vertex.hpp"
#include "Normal.hpp"
#include "QueryPoint.hpp"
#include "MCTable.hpp"
#include "FastBoxTables.hpp"
#include <vector>
#include <limits>

using std::vector;
using std::numeric_limits;

namespace lssr
{

template<typename CoordType, typename IndexType>
class FastBox
{
public:
    FastBox(Vertex<CoordType> &center);
    virtual ~FastBox() {};

    void setVertex(int index,  IndexType value);
    void setNeighbor(int index, FastBox<CoordType, IndexType>* value);

    IndexType getVertex(int index);
    FastBox<CoordType, IndexType>*     getNeighbor(int index);

    void getSurface(BaseMesh<Vertex<CoordType>, IndexType> &mesh, vector<QueryPoint<CoordType> > &query_points, IndexType &globalIndex);
    static CoordType                m_voxelsize;

    static IndexType		   		INVALID_INDEX;
private:

    CoordType calcIntersection(CoordType x1, CoordType x2, CoordType d1, CoordType d2);
    int  getIndex(vector<QueryPoint<CoordType> > &query_points);
    void getCorners(Vertex<CoordType> corners[], vector<QueryPoint<CoordType> > &query_points);
    void getDistances(CoordType distances[], vector<QueryPoint<CoordType> > &query_points);
    void getIntersections(Vertex<CoordType> corners[], CoordType distance[], Vertex<CoordType> positions[]);

    Vertex<CoordType>               m_center;
    IndexType                       m_vertices[8];
    IndexType                       m_intersections[12];
    FastBox<CoordType, IndexType>*  m_neighbors[27];

};

} // namespace lssr

#include "FastBox.tcc"

#endif /* FastBox_H_ */
