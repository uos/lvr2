/**
 *
 * @file      MeshLoader.hpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110928
 * @date      09/22/2011 09:16:36 PM
 *
 **/

#ifndef MESHIO_HPP_
#define MESHIO_HPP_

#include <stdint.h>
#include <cstddef>
#include <cstdlib>

namespace lssr
{

class MeshLoader {

    public:
        MeshLoader();
        void setVertexArray( float* array, size_t n );
        void setVertexConfidenceArray( float* array, size_t n );
        void setVertexIntensityArray( float* array, size_t n );
        void setVertexNormalArray( float* array, size_t n );
        void setVertexColorArray( float* array, size_t n );
        void setVertexColorArray( uint8_t* array, size_t n );

        void setIndexedVertexArray( float** arr, size_t size );
        void setIndexedVertexNormalArray( float** arr, size_t size );

        float* getVertexArray( size_t &n );
        float* getVertexNormalArray( size_t &n );
        float* getVertexConfidenceArray( size_t &n );
        float* getVertexIntensityArray( size_t &n );
        uint8_t* getVertexColorArray( size_t &n );

        float** getIndexedVertexArray( size_t &n );
        float** getIndexedVertexNormalArray( size_t &n );
        float** getIndexedVertexConfidenceArray( size_t &n );
        float** getIndexedVertexIntensityArray( size_t &n );
        uint8_t** getIndexedVertexColorArray( size_t &n );

        unsigned int* getFaceArray( size_t &n );
        void setFaceArray( unsigned int* array, size_t n );

        void freeBuffer();

    protected:

        float*        m_vertices;
        uint8_t*      m_vertexColors;
        float*        m_vertexConfidence;
        float*        m_vertexIntensity;
        float*        m_vertexNormals;

        float**       m_indexedVertices;
        uint8_t**     m_indexedVertexColors;
        float**       m_indexedVertexConfidence;
        float**       m_indexedVertexIntensity;
        float**       m_indexedVertexNormals;

        unsigned int* m_faceIndices;

        uint32_t      m_numVertex;
        uint32_t      m_numVertexNormals;
        uint32_t      m_numVertexColors;
        uint32_t      m_numVertexConfidence;
        uint32_t      m_numVertexIntensity;
        uint32_t      m_numFace;

};

} /* namespace lssr */
#endif /* MESHIO_HPP_ */
