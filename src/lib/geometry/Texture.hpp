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
	Texture(PointCloudManager<VertexT, NormalT>* pm, Region<VertexT, NormalT>* region);

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

	///The dimensions of the texture
	int m_sizeX, m_sizeY;

};

}

#include "Texture.tcc"

#endif /* TEXTURE_HPP_ */
