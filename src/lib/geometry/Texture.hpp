/*
 * Texture.hpp
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include "Region.hpp"
#include "../reconstruction/PointCloudManager.hpp"
#include "../io/PPMIO.hpp"
#include <boost/lexical_cast.hpp>

namespace lssr {

template<typename VertexT, typename NormalT>
class HalfEdgeVertex;

/**
 * @brief	This class represents a texture.
 */
template<typename VertexT, typename NormalT>
class Texture {
public:

	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;

	/**
	 * @brief 	Constructor.
	 *
	 * @param 	pm	a PointCloudManager containing a colored PointCloud
	 *
	 * @param	region	a region to generate a texture for.
	 *
	 */
	Texture(PointCloudManager<VertexT, NormalT>* pm, Region<VertexT, NormalT>* region,  vector<vector<HVertex*> > contours);

	/**
	 * @brief	computes texture coordinates corresponding to the give Vertex
	 *
	 * @param	v	the vertex to generate texture coordinates for
	 *
	 * @param	x	returns texture coordinates in x direction
	 *
	 * @param	y	returns texture coordinates in y direction
	 */
	void textureCoords(VertexT v, float &x, float &y);

	/**
	 *	@brief	Writes the texture to a file
	 */
	void save();

	/**
	 * Destructor.
	 */
	virtual ~Texture();

private:
	struct ColorT{
		uchar r, g, b;
	};

	Region<VertexT, NormalT>* m_region;

	///The texture data
	ColorT** m_data;

	///The coordinate system of the texture plane
	NormalT v1, v2;

	///A point in the texture plane
	VertexT p;

	///The bounding box of the texture plane
	float a_min, b_min, a_max, b_max;

	///The dimensions of the texture
	int m_sizeX, m_sizeY;

};

}

#include "Texture.tcc"

#endif /* TEXTURE_HPP_ */
