/*
 * SearchTreeFlann.hpp
 *
 *  Created on: Sep 22, 2015
 *      Author: Thomas Wiemann
 */

#ifndef LVR2_RECONSTRUCTION_SEARCHTREEFLANN_HPP_
#define LVR2_RECONSTRUCTION_SEARCHTREEFLANN_HPP_

#include <vector>
#include <memory>

#include <flann/flann.hpp>

#include <lvr2/io/Timestamp.hpp>

#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/io/PointBuffer.hpp>

using std::vector;
using std::unique_ptr;

namespace lvr2
{

/**
 * @brief SearchClass for point data.
 *
 *      This class uses the FLANN ( http://www.cs.ubc.ca/~mariusm/uploads/FLANN )
 *      library to implement a nearest neighbour search for point-data.
 */
template<typename BaseVecT>
class SearchTreeFlann : public SearchTree<BaseVecT>
{
private:
    using CoordT = typename BaseVecT::CoordType;

public:

    // typedef boost::shared_ptr< SearchTreeFlann< VertexT> > Ptr;


    /**
     *  @brief Takes the point-data and initializes the underlying searchtree.
     *
     *  @param buffer  A PointBuffer point that holds the data.
     */
    SearchTreeFlann(PointBufferPtr buffer);

    /// See interface documentation.
    virtual void kSearch(
        const Vector<BaseVecT>& qp,
        int k,
        vector<size_t>& indices,
        vector<CoordT>& distances
    ) const;

    /// See interface documentation.
    virtual void radiusSearch(
        const Vector<BaseVecT>& qp,
        CoordT r,
        vector<size_t>& indices
    ) const;

protected:

    /// The FLANN search tree structure.
    unique_ptr<flann::Index<flann::L2_Simple<float>>> m_tree;

    // /// FLANN matrix representation of the points
    // flann::Matrix<float>                                            m_flannPoints;

    // vector<int>                                                     m_ind;
    // vector<float>                                                   m_dst;



};

} // namespace lvr2

#include <lvr2/reconstruction/SearchTreeFlann.tcc>

#endif /* LVR2_RECONSTRUCTION_SEARCHTREEFLANN_HPP_ */
