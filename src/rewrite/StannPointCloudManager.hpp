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
#include "Progress.hpp"
#include "Timestamp.hpp"
#include "PLYIO.hpp"

// External libraries in lssr source tree
#include "../stann/sfcnn.hpp"
#include "../Eigen/Dense"
using namespace Eigen;

// Standard C++ includes
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;

// boost libraries
#include <boost/filesystem.hpp>

namespace lssr
{

/**
 * @brief A helper struct to represent a tangent plane
 *        of a query point. Used for normal estimation.
 */
template<typename T>
struct Plane{
    float a, b, c;
    Normal<T> n;
    Vertex<T> p;
};

/**
 * @brief A point cloud manager class that uses the STANN
 *        nearest neighbor search library to handle the data.
 *        This class calculates robust surface normals for the
 *        given point set as described in the SSRR2010 paper.
 */
template<typename T>
class StannPointCloudManager : public PointCloudManager<T>
{
public:

	/**
	 * @brief Creates a new instance using the given coordinate array.
	 *        The point data is supposed to be stored as a sequence
	 *        of n tripels that contain the point coordinates.
	 *
	 * @param points    An array of point coordinates
	 * @param normals   A normal array. If a null pointer is passed, normals
	 *                  are automatically calculated.
	 * @param n         The number of points in the data set
	 * @param kn        The number of neighbor points used for normal estimation
	 * @param ki        The number of neighbor points used for normal interpolation
	 */
	StannPointCloudManager(T **points,
	                       T **normals,
	                       size_t n,
	                       const Vertex<T> &center,
	                       const size_t &kn = 10,
	                       const size_t &ki = 10);

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
	virtual void getkClosestVertices(const Vertex<T> &v,
	        const size_t &k, vector<Vertex<T> > &nb)
	{
	    /// TODO: Implement method
	}

	/**
	 * @brief Return the k closest neighbor normals to a given query point
	 *
	 * @param n         A query vertex
	 * @param k         The (max) number of returned closest points to v
	 * @param nb        A vector containing the determined closest normals
	 */
	virtual void getkClosestNormals(const Vertex<T> &n,
	        const size_t &k, vector<Normal<T> > &nb)
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

private:

	/**
	 * @brief Save points and normals to a binary PLY file.
	 * @param filename The target file
	 */
	void savePLY(string filename);

	/**
	 * @brief Save points and normals to an ASCII file
	 */
	void savePointsAndNormals(string filename);

	/**
	 * @brief Save points to a ascii file
	 */
	void savePoints(string filename);

	/**
	 * @brief Calculates initial point normals using a least squares fit to
	 *        the \ref m_kn nearest points
	 */
	void estimateSurfaceNormals();

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
	bool boundingBoxOK(const T &dx, const T &dy, const T &dz);

	/**
	 * @brief Returns the mean distance of the given point set from
	 *        the given plane
	 *
	 * @param p             The query plane
	 * @param id            A list of point id's
	 * @param k             The number of points in the list
	 */
	T meanDistance(const Plane<T> &p, const vector<unsigned long> &id, const int &k);

	/**
	 * @brief Returns a vertex representation of the given point in the
	 *        point array
	 *
	 * @param i             A id of a point in the current point set
	 * @return              A vertex representation of the given point
	 */
	Vertex<T> fromID(int i);

	/**
	 * @brief Returns the distance between vertex v and plane p
	 */
	T distance(Vertex<T> v, Plane<T> p);

	/**
	 * @brief Returns the distance of vertex v from the nearest tangent plane
	 */
	T distance(Vertex<T> v);

	/**
	 * @brief Calculates a tangent plane for the query point using the provided
	 *        k-neighborhood
	 *
	 * @param queryPoint    The point fpr which the tangent plane is created
	 * @param k             The size of the used k-neighborhood
	 * @param id            The positions of the neighborhood points in \ref m_points
	 */
	Plane<T> calcPlane(const Vertex<T> &queryPoint,
	        const int &k,
	        const vector<unsigned long> &id);

	/// STANN tree to manage the data points
	sfcnn< T*, 3, T>            m_pointTree;

	/// The currently stored number of points
	size_t                      m_numPoints;

	/// The number of neighbors used for initial normal estimation
	int                         m_kn;

	/// The number of neighbors used for normal interpolation
	int                         m_ki;

	/// The centroid of the point set
	Vertex<T>               m_centroid;
};

}

// Include template code
#include "StannPointCloudManager.tcc"

#endif /* STANNPOINTCLOUDMANAGER_H_ */
