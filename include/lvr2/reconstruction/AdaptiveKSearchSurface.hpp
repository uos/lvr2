/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include <chrono>
#include <cmath>

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/geometry/Plane.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/geometry/BaseVector.hpp"



#include "PointsetSurface.hpp"

// #ifdef LVR2_USE_STANN
// // SearchTreeStann
// #include "SearchTreeStann.hpp"
// #endif

// #include "SearchTreeNanoflann.hpp"
#include "SearchTreeFlann.hpp"

// // SearchTreePCL
// #ifdef LVR2_USE_PCL
//     #include "SearchTreeFlannPCL.hpp"
// #endif

// // SearchTreeNabo
// #ifdef LVR2_USE_NABO
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
using std::isnan;


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
     * @param            The file to read from
     * @param searchTN   The of the searchTree type that shall be used
     * @param kn         The number of neighbor points used for normal estimation
     * @param ki         The number of neighbor points used for normal interpolation
     * @param kd         The number of neighbor points used for distance value calculation
     * @param calcMethod Normal calculation method. 0: PCA(default), 1: RANSAC, 2: Iterative
     */
    AdaptiveKSearchSurface(
        PointBufferPtr loader,
        std::string searchTreeName,
        int kn = 10,
        int ki = 10,
        int kd = 10,
        int calcMethod = 0,
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
        distance(BaseVecT v) const;

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
    //     const BaseVecT& queryPoint,
    //     int k,
    //     vector<BaseVecT> points,
    //     NormalT c_normal,
    //     bool& ok
    // );



    // /**
    //  * @brief Returns the point at the given \ref{index}.
    //  */
    // virtual const BaseVecT operator[](size_t index) const;


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

     /**
      * @brief Parses the file with scan poses and creates a search tree to
      *        search for the nearest pose when flipping normals
      */
     void parseScanPoses(string posefile);

    // /**
    //  * @brief Returns the k closest neighbor vertices to a given queryy point
    //  *
    //  * @param v         A query vertex
    //  * @param k         The (max) number of returned closest points to v
    //  * @param nb        A vector containing the determined closest points
    //  */
    // virtual void getkClosestVertices(const BaseVecT &v,
    //         const size_t &k, vector<BaseVecT> &nb);


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
        const BaseVecT &queryPoint,
        int k,
        const vector<size_t> &id
    );

    Plane<BaseVecT> calcPlaneRANSAC(
        const BaseVecT &queryPoint,
        int k,
        const vector<size_t> &id,
        bool &ok
    );

    Plane<BaseVecT> calcPlaneIterative(
        const BaseVecT &queryPoint,
        int k,
        const vector<size_t> &id
    );




    /// The centroid of the point set
    BaseVecT m_centroid;

    // 0: PCA
    // 1: RANSAC
    // 2: Iterative
    int m_calcMethod;

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

#include "lvr2/reconstruction/AdaptiveKSearchSurface.tcc"

#endif // LVR2_RECONSTRUCTION_ADAPTIVEKSEARCHSURFACE_H_
