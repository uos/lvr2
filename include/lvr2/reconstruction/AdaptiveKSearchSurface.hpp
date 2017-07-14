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
 * AdaptiveKSearchSurface.h
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 *   co-Author: Dominik Feldschnieders (dofeldsc@uos.de)
 */

#ifndef LVR2_RECONSTRUCTION_ADAPTIVEKSEARCHSURFACE_H_
#define LVR2_RECONSTRUCTION_ADAPTIVEKSEARCHSURFACE_H_

#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include <lvr/io/Model.hpp>
#include <lvr/io/Progress.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/PLYIO.hpp>
#include <lvr/io/AsciiIO.hpp>
#include <lvr/io/UosIO.hpp>

#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Plane.hpp>
// #include <lvr/geometry/ColorVertex.hpp>
// #include <lvr/geometry/BoundingBox.hpp>

#include "PointsetSurface.hpp"

// #ifdef LVR_USE_STANN
// // SearchTreeStann
// #include "SearchTreeStann.hpp"
// #endif

// #include "SearchTreeNanoflann.hpp"
#include "SearchTreeFlann.hpp"

// // SearchTreePCL
// #ifdef LVR_USE_PCL
//     #include "SearchTreeFlannPCL.hpp"
// #endif

// // SearchTreeNabo
// #ifdef LVR_USE_NABO
//     #include "SearchTreeNabo.hpp"
// #endif


// using std::cout;
// using std::endl;
// using std::ifstream;
using std::numeric_limits;
// using std::ofstream;
// using std::shared_ptr;
using std::string;
using std::unique_ptr;


namespace lvr2
{

/**
 * @brief A point cloud manager class that uses the STANN
 *        nearest neighbor search library to handle the data.
 *        This class calculates robust surface normals for the
 *        given point set as described in the SSRR2010 paper.
 */
template<typename BaseVecT>
class AdaptiveKSearchSurface : public PointsetSurface<BaseVecT>
{
public:

    // typedef shared_ptr<AdaptiveKSearchSurface<BaseVecT>> Ptr;

    /**
     * @brief Constructor.
     *
     * @param           The file to read from
     * @param searchTN  The of the searchTree type that shall be used
     * @param kn        The number of neighbor points used for normal estimation
     * @param ki        The number of neighbor points used for normal interpolation
     * @param kd        The number of neighbor points used for distance value calculation
     */
    AdaptiveKSearchSurface(
        PointBufferPtr<BaseVecT> loader,
        std::string searchTreeName,
        int kn = 10,
        int ki = 10,
        int kd = 10,
        bool useRansac = false,
        string poseFile = ""
    );

    /**
     * @brief standard Constructor
     *
     *      m_useRANSAC = true;
     *      m_ki = 10;
     *      m_kn = 10;
     *      m_kd = 10;
     *
     *      This Constructor can be used, if only the method "calcPlaneRANSACfromPoints"
     *      is required
     */
    AdaptiveKSearchSurface();

    /**
     * @brief   Destructor
     */
    virtual ~AdaptiveKSearchSurface() {};


    /// See interface documentation.
    virtual pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>
        distance(Point<BaseVecT> v) const;

    /**
     * @brief Calculates initial point normals using a least squares fit to
     *        the \ref m_kn nearest points
     */
    virtual void calculateSurfaceNormals();




    // /**
    //  * @brief Returns the number of managed points
    //  */
    // virtual size_t getNumPoints();

    // /**
    //  * @brief Calculates a tangent plane for the query point using the provided
    //  *        k-neighborhood
    //  *
    //  * @param queryPoint    The point for which the tangent plane is created
    //  * @param k             The size of the used k-neighborhood
    //  * @param points        The neighborhood points
    //  * @param ok            True, if RANSAC interpolation was succesfull
    //  *
    //  * @return the resulting plane
    //  */
    // Plane<BaseVecT> calcPlaneRANSACfromPoints(
    //     const Point<BaseVecT>& queryPoint,
    //     int k,
    //     vector<Point<BaseVecT>> points,
    //     NormalT c_normal,
    //     bool& ok
    // );



    // /**
    //  * @brief Returns the point at the given \ref{index}.
    //  */
    // virtual const Point<BaseVecT> operator[](size_t index) const;


    // virtual void colorizePointCloud( typename AdaptiveKSearchSurface<VertexT, NormalT>::Ptr pcm,
    //       const float &sqrtMaxDist = std::numeric_limits<float>::max(),
    //       const unsigned char* blankColor = NULL );

    // /**
    //  * @brief If set to true, normals will be calculated using RANSAC instead of
    //  *        plane fitting
    //  */
    // void useRansac(bool use_it) { m_useRANSAC = use_it; }


    // /// Color information for points public: TODO: This is not the best idea!
    // color3bArr                  m_colors;

    /**
     * @brief Interpolate the initial normals with the \ref m_ki neighbors
     */
    void interpolateSurfaceNormals();

private:

    // /**
    //  * @brief Parses the file with scan poses and creates a search tree to
    //  *        search for the nearest pose when flipping normals
    //  */
    // void parseScanPoses(string posefile);

    // /**
    //  * @brief Returns the k closest neighbor vertices to a given queryy point
    //  *
    //  * @param v         A query vertex
    //  * @param k         The (max) number of returned closest points to v
    //  * @param nb        A vector containing the determined closest points
    //  */
    // virtual void getkClosestVertices(const Point<BaseVecT> &v,
    //         const size_t &k, vector<Point<BaseVecT>> &nb);


    /**
     * @brief Helper function for constructors
     */
    void init();

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
    bool boundingBoxOK(float dx, float dy, float dz);

    // /**
    //  * @brief Returns the mean distance of the given point set from
    //  *        the given plane
    //  *
    //  * @param p             The query plane
    //  * @param id            A list of point id's
    //  * @param k             The number of points in the list
    //  */
    // float meanDistance(const Plane<VertexT, NormalT> &p, const vector<unsigned long> &id, const int &k);

    // /**
    //  * @brief Returns a vertex representation of the given point in the
    //  *        point array
    //  *
    //  * @param i             A id of a point in the current point set
    //  * @return              A vertex representation of the given point
    //  */
    // VertexT fromID(int i);

    // /**
    //  * @brief Returns the distance between vertex v and plane p
    //  */
    // float distance(VertexT v, Plane<VertexT, NormalT> p);


    // void radiusSearch(const VertexT &v, double r, vector<VertexT> &resV, vector<NormalT> &resN){};

    /**
     * @brief Calculates a tangent plane for the query point using the provided
     *        k-neighborhood
     *
     * @param queryPoint    The point for which the tangent plane is created
     * @param k             The size of the used k-neighborhood
     * @param id            The positions of the neighborhood points in \ref m_points
     * @param ok            True, if RANSAC interpolation was succesfull
     */
    Plane<BaseVecT> calcPlane(
        const Point<BaseVecT> &queryPoint,
        int k,
        const vector<size_t> &id
    );

    Plane<BaseVecT> calcPlaneRANSAC(
        const Point<BaseVecT> &queryPoint,
        int k,
        const vector<size_t> &id,
        bool &ok
    );


    /// The centroid of the point set
    Point<BaseVecT> m_centroid;

    /// Should a randomized algorithm be used to determine planes?
    bool m_useRANSAC;

    // /// The currently stored points
    // coord3fArr                  m_points;

    // /// The point normals
    // coord3fArr                  m_normals;

    // /// A model of the current pointcloud
    // boost::shared_ptr<Model>    m_model;

    // size_t                      m_numPoints;

    /// Search tree for scan poses
    std::shared_ptr<SearchTree<BaseVecT>> m_poseTree;

    /// Type of used search tree
    string m_searchTreeName;

};


} // namespace lvr2

#include <lvr2/reconstruction/AdaptiveKSearchSurface.tcc>

#endif // LVR2_RECONSTRUCTION_ADAPTIVEKSEARCHSURFACE_H_
