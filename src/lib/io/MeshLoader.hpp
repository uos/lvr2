/**
 *
 * @file      MeshLoader.hpp
 * @brief     Interface for all mesh loading classes.
 * @details   The MeshLoader class specifies the storage and access to all
 *            available mesh data by implementing the get and set methods for
 *            these data.
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

/**
 * \class MeshLoader MeshLoader.hpp "io/MeshLoader.hpp"
 * \brief Interface for all mesh loading classes.
 *
 * The MeshLoader class specifies the storage and access to all available mesh
 * data by implementing the get and set methods for these data. This has to be
 * the superclass of all mesh loading I/O classes.
 **/
class MeshLoader {

    public:
        /**
         * \brief Constructor.
         *
         * The default constructor of this class. This clears all internal
         * data.
         **/
        MeshLoader();


        /**
         * \brief Set the vertex array.
         *
         * By using setVertexArray the internal vertex buffer can be set. The
         * vertex array has to be a one dimensional float array containing sets
         * of \c x, \c y and \c z values.
         *
         * \param array  Pointer to interlaced vertex data.
         * \param n      Amount of vertices in the array.
         **/
        void setVertexArray( float* array, size_t n );


        /**
         * \brief Set the vertex confidence array.
         *
         * By using setVertexConfidenceArray the internal confidence buffer for
         * vertices can be set. The array has to be a one dimensional float
         * array.
         *
         * \param array  Pointer to vertex confidence data.
         * \param n      Amount of data in the array.
         **/
        void setVertexConfidenceArray( float* array, size_t n );


        /**
         * \brief Set the vertex intensity array.
         *
         * By using setVertexIntensityArray the internal intensity buffer for
         * vertices can be set. The array has to be a one dimensional float
         * array.
         *
         * \param array  Pointer to vertex intensity data.
         * \param n      Amount of data in the array.
         **/
        void setVertexIntensityArray( float* array, size_t n );


        /**
         * \brief Set the vertex normal array.
         *
         * By using setVertexNormalArray the internal vertex normal buffer can
         * be set. The array has to be a one dimensional float array containing
         * sets of \c x, \c y and \c z values.
         *
         * \param array  Pointer to interlaced vertex normal data.
         * \param n      Amount of normals in the array.
         **/
        void setVertexNormalArray( float* array, size_t n );


        /**
         * \brief Set the vertex color array.
         *
         * By using setVertexColorArray the internal vertex color buffer can be
         * set. The array has to be a one dimensional float array containing
         * sets of three values for \c red, \c green and \c blue. The values
         * have to be in the range of [0..1]. These vales are automatically
         * converted to uint8_t values in the range of [0..255].
         *
         * \param array  Pointer to interlaced vertex color data.
         * \param n      Amount of color information in the array.
         **/
        void setVertexColorArray( float* array, size_t n );


        /**
         * \brief Set the vertex color array.
         *
         * By using setVertexColorArray the internal vertex color buffer can be
         * set. The array has to be a one dimensional uint8_t array containing
         * sets of three values for \c red, \c green and \c blue. The values
         * can be in the range of [0..255].
         *
         * \param array  Pointer to interlaced vertex color data.
         * \param n      Amount of color information in the array.
         **/
        void setVertexColorArray( uint8_t* array, size_t n );


        /**
         * \brief Set the vertex array.
         *
         * By using setIndexedVertexArray the internal vertex buffer can be set. The
         * vertex array has to be a two dimensional float array containing sets
         * of \c x, \c y and \c z values. \n
         * The two dimensional array is automatically converted to an
         * interlaced one dimensional vertex array.
         *
         * \param array  Pointer to indexed vertex data.
         * \param n      Amount of vertices in the array.
         **/
        void setIndexedVertexArray( float** arr, size_t size );


        /**
         * \brief Set the vertex normal array.
         *
         * By using setIndexedVertexNormalArray the internal vertex normal
         * buffer can be set. The array has to be a two dimensional float array
         * containing sets of \c x, \c y and \c z values. \n
         * The two dimensional array is automatically converted to an
         * interlaced one dimensional vertex  normal array.
         *
         * \param array  Pointer to indexed vertex data.
         * \param n      Amount of vertices in the array.
         **/
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
