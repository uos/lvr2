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


 /*
 * PointCloudManager.h
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef POINTCLOUDMANAGER_H_
#define POINTCLOUDMANAGER_H_

#include <vector>

#include "io/Model.hpp"

#include "geometry/Vertex.hpp"
#include "geometry/Normal.hpp"
#include "geometry/ColorVertex.hpp"
#include "geometry/BoundingBox.hpp"
#include "boost/shared_ptr.hpp"

using std::vector;

namespace lssr
{

/**
 * @brief    Abstract interface class for objects that are
 *             able to handle point cloud data with normals. It
 *             defines queries for nearest neighbor search.
 */
template<typename VertexT, typename NormalT>
class PointCloudManager
{
public:

    typedef boost::shared_ptr< PointCloudManager<VertexT, NormalT> > Ptr;

    /**
     * @brief Returns the k closest neighbor vertices to a given query point
     *
     * @param v  A query vertex
     * @param k  The (max) number of returned closest points to v
     * @param nb A vector containing the determined closest points
     */
    virtual void getkClosestVertices( const VertexT &v,
        const size_t &k, vector<VertexT> &nb ) = 0;


    /**
     * @brief Returns the k closest neighbor normals to a given query point
     *
     * @param n            A query vertex
     * @param k            The (max) number of returned closest points to v
     * @param nb        A vector containing the determined closest normals
     */
    virtual void getkClosestNormals( const VertexT &n,
        const size_t &k, vector<NormalT> &nb ) = 0;

    /**
     * @brief Returns the bounding box of the loaded point set.
     */
    virtual BoundingBox<VertexT>& getBoundingBox();


    /**
     * @brief Returns the points at index \ref{index} in the point array.
     *
     * @param index
     * @return
     */
    virtual VertexT getPoint( size_t index );


    /**
     * @brief Returns the number of managed points
     */
    virtual size_t getNumPoints();


    /**
     * @brief Returns the point at the given \ref{index}.
     */
    virtual const VertexT operator[]( const size_t &index ) const;


    /**
     * @brief Transfer color information from another pointcloud.
     *
     * The function “colorizePointCloud” takes another point cloud and
     * transferes the color information from that point cloud to this one using
     * a nearest neighbor search. Using the parameter “sqrMaxDist” you can specify
     * the maximum distance for the nearest neighbor search. Beware that the
     * parameter takes the squared maximum distance. Thus if you want a maximum
     * distance of 100 you have to set the parameter to 10000 (= 100 * 100).
     * You also can specify a color for these unmatched points using
     * “blankColor”. This parameter must be either a pointer to an array
     * containing three uchars (red, green and blue) or a NULL pointer. In the
     * latter case this option will be ignored.
     *
     * @param pcm        PointCloudManager containing the point cloud to get
     *                   the color information from.
     * @param sqrMaxDist Squared maximum distance for nearest neighbor search.
     * @param blankColor Color to set the point to if there are no near color
     *                   information.
     */
    virtual void colorizePointCloud( PointCloudManager<VertexT, NormalT>::Ptr pcm,
            const float &sqrMaxDist = std::numeric_limits<float>::max(), 
            const uchar* blankColor = NULL );


    /**
     * @brief Returns the distance of vertex v from the nearest tangent plane.
     **/
    virtual void distance(VertexT v, float &projectedDistance, float &euklideanDistance) = 0;


    virtual void radiusSearch(const VertexT &v, double r, vector<VertexT> &resV, vector<NormalT> &resN) = 0;

    void setKD( int kd )
    {
        m_kd = kd;
    }

    void setKI( int ki )
    {
        m_ki = ki;
    }

    void setKN( int kn )
    {
        m_kn = kn;
    }

    virtual void calcNormals() = 0;


	/// Model of the current pointcloud.

//#if 0
    /// The currently stored points
    coord3fArr                  m_points;

    /// The point normals
    coord3fArr                  m_normals;

    /// Color information for points
    uchar**                     m_colors;
//#endif
	boost::shared_ptr<Model>    m_model;

    /// The bounding box of the point set
    BoundingBox<VertexT>        m_boundingBox;

    size_t                      m_numPoints;

    /// The number of neighbors used for initial normal estimation
    int                         m_kn;

    /// The number of neighbors used for normal interpolation
    int                         m_ki;

    /// The number of tangent planes used for distance determination
    int                         m_kd;

};


template< typename VertexT >
struct VertexTraits { };


template< typename CoordType, typename ColorT >
struct VertexTraits< ColorVertex< CoordType, ColorT > > 
{
    static inline ColorVertex< CoordType, ColorT > vertex(
            const coord3fArr &p, ColorT** c, const unsigned int idx )
    {
        return c
            ? ColorVertex< CoordType, ColorT >(
                p[idx][0], p[idx][1], p[idx][2],
                c[idx][0], c[idx][1], c[idx][2] )
            : ColorVertex< CoordType, ColorT >(
                p[idx][0], p[idx][1], p[idx][2] );
        /* TODO: Make sure we always have color information if we have
         *       ColorVertex! */
    }
};


template< typename CoordType >
struct VertexTraits< Vertex< CoordType > > 
{
    static inline Vertex< CoordType > vertex(
            const coord3fArr &p, void** c, const unsigned int idx )
    {
        return Vertex< CoordType >(
                p[idx][0], p[idx][1], p[idx][2] );
    }
};

} // namespace lssr

#include "PointCloudManager.tcc"

#endif /* POINTCLOUDMANAGER_H_ */
