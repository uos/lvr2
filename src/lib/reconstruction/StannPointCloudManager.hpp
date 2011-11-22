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
 * StannPointCloudManager.h
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef STANNPOINTCLOUDMANAGER_H_
#define STANNPOINTCLOUDMANAGER_H_

// Internal includes from lssr
#include "PointCloudManager.hpp"

#include "../io/Model.hpp"
#include "../io/Progress.hpp"
#include "../io/Timestamp.hpp"
#include "../io/PLYIO.hpp"
#include "../io/AsciiIO.hpp"
#include "../io/UosIO.hpp"

// Stann
#include "../stann/sfcnn.hpp"

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

/**
 * @brief A helper struct to represent a tangent plane
 *        of a query point. Used for normal estimation.
 */
template<typename VertexT, typename NormalT>
struct Plane{
    float a, b, c;
    NormalT n;
    VertexT p;
};

/**
 * @brief A point cloud manager class that uses the STANN
 *        nearest neighbor search library to handle the data.
 *        This class calculates robust surface normals for the
 *        given point set as described in the SSRR2010 paper.
 */
template<typename VertexT, typename NormalT>
class StannPointCloudManager : public PointCloudManager<VertexT, NormalT>
{
public:

	/**
	 * @brief Trys to read the given file to create a new StannPointCloudManager
	 *        instance.
	 *
	 * @param           The file to read from
	 * @param kn        The number of neighbor points used for normal estimation
     * @param ki        The number of neighbor points used for normal interpolation
     * @param kd        The number of neighbor points used for distance value calculation
	 */
	StannPointCloudManager(PointBuffer* loader,
	                       const int &kn = 10,
	                       const int &ki = 10,
	                       const int &kd = 10);

	/**
	 * @brief   Destructor
	 */
	virtual ~StannPointCloudManager() {};

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
	 * @brief Saves currently present data to a file
	 *
	 * Which data is saved depends on the file extension. Extensions
	 * .xyz, .pts and .3d will produce ASCII files that contain point
	 * cloud data only. Files with extension .nor will additionally
	 * contain normal defintions. Extension .ply will produce a
	 * binary .ply file with point and normal information.
	 *
	 * @param filename  The target file.
	 */
	void save(string filename);


    /**
     * @brief Returns the distance of vertex v from the nearest tangent plane
     * @TODO ///TODO: Doku
     */
    virtual void distance(VertexT v, float &projectedDistance, float &euklideanDistance);

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

	/**
	 * @brief Calculates a tangent plane for the query point using the provided
	 *        k-neighborhood
	 *
	 * @param queryPoint    The point fpr which the tangent plane is created
	 * @param k             The size of the used k-neighborhood
	 * @param id            The positions of the neighborhood points in \ref m_points
	 */
	Plane<VertexT, NormalT> calcPlane(const VertexT &queryPoint,
	        const int &k,
	        const vector<unsigned long> &id);

	Plane<VertexT, NormalT> calcPlaneRANSAC(const VertexT &queryPoint,
	        const int &k,
	        const vector<unsigned long> &id);

	/// STANN tree to manage the data points
	sfcnn< float*, 3, float>    m_pointTree;

	/// The centroid of the point set
	VertexT               		m_centroid;

	bool                        m_useRANSAC;
};

}

// Include template code
#include "StannPointCloudManager.tcc"

#endif /* STANNPOINTCLOUDMANAGER_H_ */
