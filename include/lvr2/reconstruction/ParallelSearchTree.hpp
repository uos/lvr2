#ifndef LVR2_RECONSTRUCTION_PARALLELSEARCHTREE_HPP
#define LVR2_RECONSTRUCTION_PARALLELSEARCHTREE_HPP

namespace lvr2
{

template<typename BaseVecT>
class ParallelSearchTree : public SearchTree<BaseVecT>
{
private:
    using CoordT = typename BaseVecT::CoordType;

public:
    
    // virtual int kSearch(
    //     const BaseVecT& qp,
    //     int k,
    //     vector<size_t>& indices
    // ) const;

    // virtual int radiusSearch(
    //     const BaseVecT& qp,
    //     int k,
    //     float r,
    //     vector<size_t>& indices,
    //     vector<CoordT>& distances
    // ) const;

    virtual void kSearchParallel(
        const BaseVecT* query,
        int n,
        int k,
        size_t* indices,
        CoordT* distances
    ) const = 0;

     virtual void radiusSearchParallel(
        const BaseVecT* query,
        int k,
        float r,
        size_t* indices,
        vector<CoordT>& distances
    ) const = 0;

protected:
    protected:
    /// The number of neighbors used for normal interpolation
    int                         m_ki;

    using ParallelSearchTreePtr = std::shared_ptr<ParallelSearchTree<BaseVecT>>;

};

}


#endif // LVR2_RECONSTRUCTION_PARALLELSEARCHTREE_HPP