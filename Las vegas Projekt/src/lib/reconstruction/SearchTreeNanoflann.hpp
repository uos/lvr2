/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/**
 * SearchTreeNanoflann.hpp
 *
 *  @date 29.08.2012
 *  @author Thomas Wiemann
 */

#ifndef SEARCHTREENANOFLANN_HPP_
#define SEARCHTREENANOFLANN_HPP_

#include "SearchTree.hpp"

#include <nanoflann/nanoflann.hpp>

namespace lssr
{

/**
 * @brief SearchClass for point data.
 *
 *      This class uses the nanoflann( https://code.google.com/p/nanoflann/ )
 *      library to implement a nearest neighbour search for point-data.
 */
template< typename VertexT >
class SearchTreeNanoflann : public SearchTree<VertexT>
{
public:


    /**
     *  @brief Constructor. Takes the point-data and initializes the underlying searchtree.
     *
     *  @param loader  A PointBuffer point that holds the data.
     *  @param kn      The number of neighbour points used for normal estimation.
     *  @param ki      The number of neighbour points used for normal interpolation.
     *  @param kd      The number of neighbour points used for distance value calculation.
     */
    SearchTreeNanoflann( PointBufferPtr points,
            size_t &n_points,
            const int &kn = 10,
            const int &ki = 10,
            const int &kd = 10,
            const bool &useRansac = false );

    /**
     * @brief This function performs a k-next-neighbor search on the
                       data that were given in the constructor.

     * @param qp          A float array which contains the query point for which the neighbours are searched.
     * @param neighbours  The number of neighbours that should be searched.
     * @param indices     A vector that stores the indices for the neighbours whithin the dataset.
     * @param distances   A vector that stores the distances for the neighbours that are found.
     */
    virtual void kSearch(
            coord < float >& qp,
            int neighbours, vector< ulong > &indices,
            vector< double > &distances );


    virtual void kSearch(VertexT qp, int k, vector< VertexT > &neighbors);


    virtual void radiusSearch( float              qp[3], double r, vector< ulong > &indices );
    virtual void radiusSearch( VertexT&              qp, double r, vector< ulong > &indices );
    virtual void radiusSearch( const VertexT&        qp, double r, vector< ulong > &indices );
    virtual void radiusSearch( coord< float >&       qp, double r, vector< ulong > &indices );
    virtual void radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices );

    /// Destructor
    virtual ~SearchTreeNanoflann() {};



private:

    /// Adaptor class for nanoflann
    template<typename T>
    class NFPointCloud
    {
    public:
        NFPointCloud(PointBufferPtr ptr) : m_ptr(ptr)
        {
            size_t num;
            m_numPoints = m_ptr->getNumPoints();
            m_points = m_ptr->getPointArray(num);
        };

        inline size_t kdtree_get_point_count() const { return m_numPoints;}

        inline T kdtree_get_pt(const size_t i, int dim) const
        {
            if(dim == 0)
            {
                return (T)m_points[3 * i];
            }
            else if(dim == 1)
            {
                return (T)m_points[3 * i + 1];
            }
            else
            {
                return (T)m_points[3 * i + 2];
            }
        }

        inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t size) const
        {
            const T d0 = p1[0] - m_points[idx_p2 * 3];
            const T d1 = p1[1] - m_points[idx_p2 * 3 + 1];
            const T d2 = p1[2] - m_points[idx_p2 * 3 + 2];
            return d0*d0 + d1*d1 + d2*d2;
        }

        template <typename BBOX>
        bool kdtree_get_bbox(BBOX &bb) const { return false; }

        PointBufferPtr  m_ptr;
        floatArr        m_points;
        size_t          m_numPoints;
    };

    /// Point cloud adator
    NFPointCloud<float>* m_pointCloud;

    /// kd-tree
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, NFPointCloud<float> > ,
            NFPointCloud<float>,
            3 > kd_tree_t;

    kd_tree_t* m_tree;
};

} /* namespace lssr */

#include "SearchTreeNanoflann.tcc"

#endif /* SEARCHTREENANOFLANN_HPP_ */
