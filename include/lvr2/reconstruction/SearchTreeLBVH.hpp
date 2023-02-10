#ifndef LVR2_RECONSTRUCTION_SEARCHTREELBVH_HPP_
#define LVR2_RECONSTRUCTION_SEARCHTREELBVH_HPP_

#include <vector>

#include "cuda/LBVHIndex.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"

namespace lvr2
{

/**
 * @brief SearchClass for point data.
 */

template<typename BaseVecT>
class SearchTreeLBVH : public SearchTree<BaseVecT>
{
private:
    using CoordT = typename BaseVecT::CoordType;

public:

    /**
     *  @brief Takes the point-data and initializes the underlying searchtree.
     *
     *  @param buffer  A PointBuffer point that holds the data.
     */
    SearchTreeLBVH(PointBufferPtr buffer);

    /// See interface documentation.
    virtual int kSearch(
        const BaseVecT& qp,
        int k,
        vector<size_t>& indices,
        vector<CoordT>& distances
    ) const override;

    /// See interface documentation.
    virtual int radiusSearch(
        const BaseVecT& qp,
        int k,
        float r,
        vector<size_t>& indices,
        vector<CoordT>& distances
    ) const override;

    /**
     * @brief Performs a parallel kNN Search on the GPU
     * 
     * @param query     Query points for which the neighbors are searched
     * @param n         Number of queries
     * @param k         Max Number of neighbors per query
     * @param indices   Indices of the found neighbors
     * @param distances Distances of the found neighbors
     */
    void kSearchParallel(
        const BaseVecT* query,
        int n,
        int k,
        vector<size_t>& indices,
        vector<CoordT>& distances
    ) const;
    
    /**
     * @brief Performs a parallel radius Search on the GPU
     * 
     * @param query     Query points for which the neighbors are searched
     * @param n         Number of queries
     * @param k         Max Number of neighbors per query
     * @param r         The max radius
     * @param indices   Indices of the found neighbors
     * @param distances Distances of the found neighbors
     * @param neighbors Number of found neighbors for each query
     */
    void radiusSearchParallel(
        const BaseVecT* query,
        int n,
        int k,
        float r,
        vector<size_t>& indices,
        vector<CoordT>& distances,
        vector<unsigned int>& neighbors
    ) const;

protected:

    lbvh::LBVHIndex m_tree;
};
} // namespace lvr2

#include "lvr2/reconstruction/SearchTreeLBVH.tcc"

#endif // LVR2_RECONSTRUCTION_SEARCHTREELBVH_HPP_