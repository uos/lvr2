#ifndef LAS_VEGAS_MPI_DATA_HANDLER_HPP
#define LAS_VEGAS_MPI_DATA_HANDLER_HPP

#include <vector>
// floatArr etc
#include <lvr/io/DataStruct.hpp>

using namespace lvr;
template<typename BoundingBoxT, typename VertexT>
class MPIDataManager {
    public:
        MPIDataManager(BoundingBoxT& bb);
        virtual void insert(VertexT point) = 0;
        virtual unsigned int getNumberOfCells() = 0;
        virtual floatArr getCellData(unsigned int cell_id) = 0;
        virtual unsigned int getCellDataSize(unsigned int cell_id) = 0;
        virtual BoundingBoxT getCellBoundingBox(unsigned int cell_id) = 0;
        virtual std::vector<unsigned int> getCellNeighbourIds(unsigned int cell_id) = 0;
    private:
        BoundingBoxT m_bb;
};

#include "MPIDataManager.tcc"

#endif //LAS_VEGAS_MPI_DATA_HANDLER_HPP