#ifndef LVR2_RECONSTRUCTION_SEARCHTREELBVH_HPP_
#define LVR2_RECONSTRUCTION_SEARCHTREELBVH_HPP_

#include <vector>

#include "cuda/LBVHIndex.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"

namespace lvr2
{

template<typename BaseVecT>
class SearchTreeLBVH : public SearchTree<BaseVecT>
{
private:
    using CoordT = typename BaseVecT::CoordType;

public:
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

    void kSearchParallel(
        const BaseVecT* query,
        int n,
        int k,
        vector<size_t>& indices,
        vector<CoordT>& distances
    ) const;
    
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