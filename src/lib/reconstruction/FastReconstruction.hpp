/*
 * FastReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef FastReconstruction_H_
#define FastReconstruction_H_

#include "../geometry/Vertex.hpp"
#include "../geometry/BoundingBox.hpp"
#include "../reconstruction/Reconstructor.hpp"
#include "../reconstruction/LocalApproximation.hpp"
#include "../reconstruction/FastBox.hpp"
#include "../reconstruction/QueryPoint.hpp"

#include <ext/hash_map>
using namespace __gnu_cxx;

namespace lssr
{

/**
 * @brief A surface reconstruction object that implements the standard
 *        marching cubes algorithm using a hashed grid structure for
 *        parallel computation.
 */
template<typename VertexT, typename NormalT>
class FastReconstruction : public Reconstructor<VertexT, NormalT>
{
public:

    /**
     * @brief Constructor.
     *
     * @param manager       A point cloud manager instance
     * @param resolution    The number of intersections (on the longest side
     *                      of the volume taken by the data points) used by
     *                      the reconstruction.
     * @param isVoxelsize   If set to true, interpret resolution as voxelsize
     *                      instead of number of intersections
     */
    FastReconstruction(PointCloudManager<VertexT, NormalT> &manager,  float resolution, bool isVoxelsize = false);


    /**
     * @brief Destructor.
     */
    virtual ~FastReconstruction() {};

    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<VertexT, NormalT> &mesh);

private:

    /**
     * @brief Calculates the maximal grid indices
     */
    void calcIndices();

    /**
     * @brief Creates the needed query points for the reconstruction
     *        processs
     */
    void createGrid();

    /**
     * @brief Calculates the distance for all generated query points
     */
    void calcQueryPointValues();

    /**
     * @brief Tries to find an existing query point in the grid for
     *        the virtual box corner (1..8) for the  cell at (i, j, k)
     *        in the grid.
     *
     * @param position  The box corner index
     * @param i, j, k   A triple that identifies a cell in the grid
     *
     * @return The index of an existing query point or -1 if no point
     *         corresponding to the given position exists.
     */
    uint findQueryPoint(const int &position,
            const int &i, const int &j, const int &k);

    /**
     * @brief Calculates the hash value for the given index tripel
     */
    inline size_t hashValue(int i, int j, int k) const
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    /**
     * @brief Rounds the given value to the neares integer value
     */
    inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }

    /// The voxelsize used for reconstruction
    float                  		m_voxelsize;

    /// The absolute maximal index of the reconstruction grid
    size_t                      m_maxIndex;

    /// The squared maximal index of the reconstruction grid
    size_t                      m_maxIndexSquare;

    /// The maximal index in x direction
    size_t                      m_maxIndexX;

    /// The maximal index in y direction
    size_t                      m_maxIndexY;

    /// The maximal index in z direction
    size_t                      m_maxIndexZ;

    /// A hahs map to store the created grid cells
    hash_map<size_t, FastBox<VertexT, NormalT>* >  m_cells;

    /// A vector containing the query points for the reconstruction
    vector<QueryPoint<VertexT> > m_queryPoints;
};


} // namespace lssr


#include "FastReconstruction.tcc"

#endif /* FastReconstruction_H_ */
