#ifndef LAS_VEGAS_MPI_OCTREE_HPP
#define LAS_VEGAS_MPI_OCTREE_HPP

#include <vector>
// floatArr etc
#include <lvr/io/DataStruct.hpp>
#include <lvr/geometry/Vertex.hpp>
#include <lvr/geometry/BoundingBox.hpp>


#include "MPIDataManager.hpp"

using namespace std;
using namespace lvr;
template<typename BoundingBoxT, typename VertexT>
class MPIOctree : public MPIDataManager<BoundingBoxT, VertexT>
{
    public:
        MPIOctree(BoundingBoxT& bb);
        virtual void insert(VertexT point);
        virtual unsigned int getNumberOfCells();
        virtual floatArr getCellData(unsigned int cell_id);
        virtual unsigned int getCellDataSize(unsigned int cell_id);
        virtual BoundingBoxT getCellBoundingBox(unsigned int cell_id);
        virtual std::vector<unsigned int> getCellNeighbourIds(unsigned int cell_id);
    private:
};

#include "MPIOctree.tcc"



#endif //LAS_VEGAS_MPI_OCTREE_HPP
