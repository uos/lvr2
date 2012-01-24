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

#ifndef M_POINTCLOUDMANAGER_H_
#define M_POINTCLOUDMANAGER_H_

#include "../io/Model.hpp"
#include "../io/Progress.hpp"
#include "../io/Timestamp.hpp"
#include "../io/PLYIO.hpp"
#include "../io/AsciiIO.hpp"
#include "../io/UosIO.hpp"

#include "geometry/Vertex.hpp"
#include "geometry/Normal.hpp"
#include "geometry/ColorVertex.hpp"
#include "geometry/BoundingBox.hpp"
#include "boost/shared_ptr.hpp"

// Stann
#include "../stann/sfcnn.hpp"

// SearchTreeStann
#include "SearchTreeStann.hpp"

// SearchTreePCL
#ifdef _USE_PCL_
    #include "SearchTreePCL.hpp"
#endif

// SearchTreeNabo
#ifdef _USE_NABO_
    #include "SearchTreeNabo.hpp"
#endif

// Standard C++ includes
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <limits>
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::numeric_limits;

namespace lssr
{

#ifndef STRUCT_PLANE_
#define STRUCT_PLANE_
/**
 * @brief A helper struct to represent a tangent plane
 *        of a query point. Used for normal estimation.
 */
template<typename VertexT, typename NormalT>
struct Plane
{
    float a, b, c;
    NormalT n;
    VertexT p;
};
#endif

/**
 * @brief A point cloud manager class that uses the STANN
 *        nearest neighbor search library to handle the data.
 *        This class calculates robust surface normals for the
 *        given point set as described in the SSRR2010 paper.
 */
template<typename VertexT, typename NormalT>
class PointCloudManager
{
public:

    typedef boost::shared_ptr< PointCloudManager<VertexT, NormalT> > Ptr;

	/**
	 * @brief Trys to read the given file to create a new PointCloudManager
	 *        instance.
	 *
	 * @param           The file to read from
     * @param searchTN  The of the searchTree type that shall be used
	 * @param kn        The number of neighbor points used for normal estimation
     * @param ki        The number of neighbor points used for normal interpolation
     * @param kd        The number of neighbor points used for distance value calculation
	 */
	PointCloudManager( PointBufferPtr loader,
                        std::string searchTreeName,
	                       const int &kn = 10,
	                       const int &ki = 10,
	                       const int &kd = 10,
						   const bool &useRansac = false );

	/**
	 * @brief   Destructor
	 */
	virtual ~PointCloudManager() {};

	/**
	 * @brief Returns the k closest neighbor vertices to a given queryy point
	 *
	 * @param v         A query vertex
	 * @param k         The (max) number of returned closest points to v
	 * @param nb        A vector containing the determined closest points
	 */
	virtual void getkClosestVertices(const VertexT &v,
	        const size_t &k, vector<VertexT> &nb);
    
    
    /**
     * @brief Returns the bounding box of the loaded point set.
     */
    virtual BoundingBox<VertexT>& getBoundingBox();
    
    
    /**
     * @brief Returns the number of managed points
     */
    virtual size_t getNumPoints();


    /**
     * @brief Returns the point buffer object
     */
    virtual PointBufferPtr pointBuffer() { return m_pointBuffer; }


    /**
     * @brief Returns the point at the given \ref{index}.
     */
    virtual const VertexT operator[]( const size_t &index ) const;
    
    
    /**
     * @brief Returns the points at index \ref{index} in the point array.
     *
     * @param index
     * @return
     */
    virtual VertexT getPoint( size_t index );
	

	/**
	 * @brief Return the k closest neighbor normals to a given query point
	 *
	 * @param n         A query vertex
	 * @param k         The (max) number of returned closest points to v
	 * @param nb        A vector containing the determined closest normals
	 */
    virtual void getkClosestNormals(const VertexT &n,
	        const size_t &k, vector<NormalT> &nb)
	{
	    /// TODO: Implement method
	}


    /**
     * @brief Returns the distance of vertex v from the nearest tangent plane
     * @param v                     A grid point
     * @param projectedDistance     Projected distance of the query point to the
     *                              isosurface
     * @param euclideanDistance     Euklidean Distance to the nearest data point
     */
    virtual void distance(VertexT v, float &projectedDistance, float &euklideanDistance);


    virtual void colorizePointCloud( PointCloudManager<VertexT, NormalT>::Ptr pcm,
          const float &sqrtMaxDist = std::numeric_limits<float>::max(),
          const uchar* blankColor = NULL );

    /**
     * @brief Calculates initial point normals using a least squares fit to
     *        the \ref m_kn nearest points
     */
    void calcNormals();

    /**
     * @brief If set to true, normals will be calculated using RANSAC instead of
     *        plane fitting
     */
    void useRansac(bool use_it) { m_useRANSAC = use_it;}
    
    
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

    bool haveNormals()
    {
        return m_normals != 0;
    }
    
    /// Color information for points public: TODO: This is not the best idea!
    color3bArr                  m_colors;

private:

    /**
     * @brief Helper function for constructors
     */
    void init();


	/**
	 * @brief Interpolate the initial normals with the \ref m_ki neighbors
	 */
	void interpolateSurfaceNormals();

	/**
	 * @brief Checks if the bounding box of a point set is "well formed",
	 *        i.e. no dimension is significantly larger than the other.
	 *
	 * This method is needed to achieve a better quality of the initial normal
	 * estimation in sparse scans. Details are described in the SRR2010 paper.
	 *
	 * @param dx, dy, dz The side lengths of the bounding box
	 *
	 * @return true if the given box has valid dimensions.
	 */
	bool boundingBoxOK(const float &dx, const float &dy, const float &dz);

	/**
	 * @brief Returns the mean distance of the given point set from
	 *        the given plane
	 *
	 * @param p             The query plane
	 * @param id            A list of point id's
	 * @param k             The number of points in the list
	 */
	float meanDistance(const Plane<VertexT, NormalT> &p, const vector<unsigned long> &id, const int &k);

	/**
	 * @brief Returns a vertex representation of the given point in the
	 *        point array
	 *
	 * @param i             A id of a point in the current point set
	 * @return              A vertex representation of the given point
	 */
	VertexT fromID(int i);

	/**
	 * @brief Returns the distance between vertex v and plane p
	 */
	float distance(VertexT v, Plane<VertexT, NormalT> p);


	void radiusSearch(const VertexT &v, double r, vector<VertexT> &resV, vector<NormalT> &resN){};

	/**
	 * @brief Calculates a tangent plane for the query point using the provided
	 *        k-neighborhood
	 *
	 * @param queryPoint    The point fpr which the tangent plane is created
	 * @param k             The size of the used k-neighborhood
	 * @param id            The positions of the neighborhood points in \ref m_points
	 * @param ok            True, if RANSAC interpolation was succesfull
	 */
	Plane<VertexT, NormalT> calcPlane(const VertexT &queryPoint,
	        const int &k,
	        const vector<unsigned long> &id);

	Plane<VertexT, NormalT> calcPlaneRANSAC(const VertexT &queryPoint,
	        const int &k,
	        const vector<unsigned long> &id, bool &ok );

    boost::shared_ptr< SearchTree< VertexT, NormalT > > m_searchTree;

	/// The centroid of the point set
	VertexT               		m_centroid;

    /// Should a randomized algorithm be used to determine planes?
	bool                        m_useRANSAC;
    
	/// A buffer containing the points that are managed here. 
    PointBufferPtr              m_pointBuffer;

    /// The currently stored points
    coord3fArr                  m_points;

    /// The point normals
    coord3fArr                  m_normals;

    /// A model of the current pointcloud
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
            const coord3fArr &p, color3bArr c, const unsigned int idx )
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
            const coord3fArr &p, color3bArr c, const unsigned int idx )
    {
        return Vertex< CoordType >(
                p[idx][0], p[idx][1], p[idx][2] );
    }
};

}

// Include template code
#include "PointCloudManager.tcc"

#endif /* STANNPOINTCLOUDMANAGER_H_ */
