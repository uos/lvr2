/*
 * PointCloudManager.tcc
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#include <cassert>
#include <string>
using std::string;

#include "../io/PLYIO.hpp"
#include "../io/AsciiIO.hpp"
#include "../io/UosIO.hpp"

#include <boost/filesystem.hpp>

namespace lssr
{

template<typename VertexT, typename NormalT>
BoundingBox<VertexT>& PointCloudManager<VertexT, NormalT>::getBoundingBox()
{
    return m_boundingBox;
}

template<typename VertexT, typename NormalT>
VertexT PointCloudManager<VertexT, NormalT>::getPoint(size_t index)
{
    assert(index < m_numPoints);
    return VertexT(m_points[index][0], m_points[index][1], m_points[index][2], m_colors[index][0], m_colors[index][1], m_colors[index][2]);
}

template<typename VertexT, typename NormalT>
size_t PointCloudManager<VertexT, NormalT>::getNumPoints()
{
    return m_numPoints;
}

template<typename VertexT, typename NormalT>
const VertexT PointCloudManager<VertexT, NormalT>::operator[](const size_t& index) const
{
    return VertexT(m_points[index][0], m_points[index][1], m_points[index][2]);
}


template<typename VertexT, typename NormalT>
void PointCloudManager<VertexT, NormalT>::readFromFile(string filename)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

    if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        AsciiIO asciiIO;
        asciiIO.read(filename);
        this->m_points = asciiIO.getIndexedPointArray( &(this->m_numPoints) );
        this->m_normals = 0;
        this->m_colors = asciiIO.getIndexedPointColorArray();
    }
    else if(extension == ".ply")
    {
        // Read given input file
        PLYIO plyio;
		  /* 
			* Read:
			*  color       yes
			*  confidence   no
			*  intensity    no
			*  normals     yes
			*  faces        no
			**/
        plyio.read( filename, true, false, false, true, false );

        this->m_points  = plyio.getIndexedPointArray( &this->m_numPoints );
		  size_t n(0);
        this->m_normals = plyio.getIndexedPointNormalArray( &n );
        if ( n != this->m_numPoints ) {
            m_normals = NULL;
        }
        this->m_colors  = plyio.getIndexedPointColorArray( &n );
        if ( n != this->m_numPoints ) {
            m_colors = NULL;
        }
    }
    else if(extension == "")
    {
        UosIO uosio;
        uosio.read(filename);
        this->m_points = uosio.getIndexedPointArray();
        this->m_numPoints = uosio.getNumPoints();
        this->m_normals = uosio.getIndexedPointNormalArray();
        this->m_colors = uosio.getIndexedPointColorArray();
    }
}

}

