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

#ifdef __GNUC__
#define DEPRECATED __attribute__ ((deprecated))
#else
#define DEPRECATED 
#endif

namespace lssr
{

/**
 * \class MeshLoader MeshLoader.hpp "io/MeshLoader.hpp"
 * \brief Interface for all mesh loading classes.
 * \todo  At the moment this class comes along with a lot of possible memory
 *        leaks. To prevent those all data should be stored as \c shared_ptr as
 *        introduced by C++11.
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
         * \deprecated This method is deprecated. To be consistent, all
         *             internal color data should be unsigned 8bit integers. At
         *             the moment however some parts of the lssr toolkit still
         *             use float values in the range of [0..1] to describe
         *             color information. So this function is still available
         *             for compatibility reasons. But it might be removed
         *             anytime.
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
        void setVertexColorArray( float* array, size_t n ) DEPRECATED;


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


        /**
         * \brief Get the vertex array.
         *
         * Using getVertexArray the vertex data can be retrieved. The returned
         * array is a one dimensional interlaced float array containing sets of
         * \c x, \c y and \c z values. Additionally the passed reference of a
         * size_t variable is set to the amount of vertices stored in the
         * array. Thus \c n is set to one third of the array length.
         *
         * \param n  Amount of vertices in array.
         * \return   %Vertex array.
         **/
        float* getVertexArray( size_t &n );


        /**
         * \brief Get the vertex normal array.
         *
         * getVertexNormalArray returns the vertex normal data. The returned
         * array is a one dimensional interlaced float array containing sets of
         * \c x, \c y and \c z values. Additionally the passed reference of a
         * size_t variable is set to the amount of vertex normals stored in the
         * array. Thus \c n is set to one third of the array length.
         *
         * \param n  Amount of vertex normals in array.
         * \return   %Vertex normal array.
         **/
        float* getVertexNormalArray( size_t &n );


        /**
         * \brief Get the vertex confidence array.
         *
         * getVertexConfidenceArray returns the vertex confidence
         * informations. The returned array is a one dimensional float array.
         * Additionally the passed reference of a size_t variable is set to the
         * amount of confidence values stored in the array. Thus \c n is set to
         * the array length.
         *
         * \param n  Amount of confidence values in array.
         * \return   %Vertex confidence array.
         **/
        float* getVertexConfidenceArray( size_t &n );


        /**
         * \brief Get the vertex intensity array.
         *
         * getVertexiIntensityArray returns the vertex intensity informations.
         * The returned array is a one dimensional float array.  Additionally
         * the passed reference of a size_t variable is set to the amount of
         * intensity values stored in the array. Thus \c n is set to the array
         * length.
         *
         * \param n  Amount of intensity values in array.
         * \return   %Vertex intensity array.
         **/
        float* getVertexIntensityArray( size_t &n );


        /**
         * \brief Get the vertex color array.
         *
         * getVertexColorArray returns the vertex color data. The returned
         * array is a one dimensional interlaced uint8_t array containing sets
         * of \c red, \c green and \c blue values. Additionally the passed
         * reference of a size_t variable is set to the amount of vertex color
         * sets stored in the array. Thus \c n is set to one third of the array
         * length.
         *
         * \param n  Amount of vertex color sets in array.
         * \return   %Vertex color array.
         **/
        uint8_t* getVertexColorArray( size_t &n );


        /**
         * \brief Get indexed vertex array.
         *
         * This method return a two dimensional float array containing the
         * vertex data. The actual data is the same as returned by
         * \ref getVertexArray. Additionally \c n is set to the amount of
         * vertices and thus to the length of the array.
         *
         * \param n  Amount of vertices in array.
         * \return   Indexed vertex array.
         **/
        float** getIndexedVertexArray( size_t &n );


        /**
         * \brief Get indexed vertex normal array.
         *
         * This method return a two dimensional float array containing the
         * vertex normals. The actual data is the same as returned by
         * \ref getVertexNormalArray. Additionally \c n is set to the amount of
         * vertex normals and thus to the length of the array.
         *
         * \param n  Amount of vertex normals in array.
         * \return   Indexed vertex normal array.
         **/
        float** getIndexedVertexNormalArray( size_t &n );


        /**
         * \brief Get indexed vertex confidence array.
         * \note  As each of the internal arrays only contain one value it is
         *        strongly suggested to use \ref getVertexConfidenceArray
         *        instead. That way the overhead introduced by this method is
         *        omitted and the data can be accessed even more easier as with
         *        this method.
         *
         * This method return a two dimensional float array containing the
         * vertex confidence data. The actual data is the same as returned by
         * \ref getVertexConfidenceArray. Additionally \c n is set to the
         * amount of data and thus to the length of the array.
         *
         * \param n  Amount of vertex confidence data in array.
         * \return   Indexed vertex confidence array.
         **/
        float** getIndexedVertexConfidenceArray( size_t &n );


        /**
         * \brief Get indexed vertex intensity array.
         * \note  As each of the internal arrays only contain one value it is
         *        strongly suggested to use \ref getVertexIntensityArray
         *        instead. That way the overhead introduced by this method is
         *        omitted and the data can be accessed even more easier as with
         *        this method.
         *
         * This method return a two dimensional float array containing the
         * vertex intensity information. The actual data is the same as
         * returned by \ref getVertexIntensityArray. Additionally \c n is set
         * to the amount of data and thus to the length of the array.
         *
         * \param n  Amount of vertex intensities in array.
         * \return   Indexed vertex intensity array.
         **/
        float** getIndexedVertexIntensityArray( size_t &n );


        /**
         * \brief Get indexed vertex color array.
         *
         * This method return a two dimensional float array containing the
         * vertex color information. The actual data is the same as returned by
         * \ref getVertexColorArray. Additionally \c n is set to the amount of
         * data and thus to the length of the array.
         *
         * \param n  Amount of vertex color sets in array.
         * \return   Indexed vertex color array.
         **/
        uint8_t** getIndexedVertexColorArray( size_t &n );


        /**
         * \brief Set the face index array.
         *
         * This method is used to set the face index array. The array passed as
         * argument is a one dimensional interlaced array of unsigned integers.
         * Each set of three integers specifies one face. The parameter \c n
         * specifies the amount of faces. Thus n has to be one third of the
         * array length.
         *
         * \param array  %Face index array.
         * \param n      Amount of faces in array.
         **/
        void setFaceArray( unsigned int* array, size_t n );


        /**
         * \brief Get the face index array.
         *
         * This method returns the vertex indices defining the faces. The
         * returned array is a one dimensional interlaced uint8_t array
         * containing sets of of three vertex indices. Additionally the passed
         * reference of a size_t variable is set to the amount of faces.
         * Because every face is a triangle, \c n is set to one third of the
         * array length.
         *
         * \param n  Amount of faces defined in the array.
         * \return   %Face index array.
         **/
        unsigned int* getFaceArray( size_t &n );


        /**
         * \brief Clear internal vertex and face buffers.
         *
         * This method clears all internal buffers and the stored buffer
         * length. That means all pointers are set to null-pointers and all
         * length variables are set to zero.
         * \warning This method does not free any allocated memory.
         **/
        void freeBuffer();

    protected:

        /// %Vertex buffer.
        float*        m_vertices;
        /// %Vertex color buffer.
        uint8_t*      m_vertexColors;
        /// %Vertex confidence buffer.
        float*        m_vertexConfidence;
        /// %Vertex intensity buffer.
        float*        m_vertexIntensity;
        /// %Vertex normal buffer.
        float*        m_vertexNormals;

        /// Indexed vertex buffer.
        float**       m_indexedVertices;
        /// Indexed vertex color buffer.
        uint8_t**     m_indexedVertexColors;
        /// Indexed vertex confidence buffer.
        float**       m_indexedVertexConfidence;
        /// Indexed vertex intensity buffer.
        float**       m_indexedVertexIntensity;
        /// Indexed vertex normal buffer.
        float**       m_indexedVertexNormals;

        /// Buffer of face indices
        unsigned int* m_faceIndices;

        /// Number of vertices in internal buffer.
        uint32_t      m_numVertex;
        /// Number of vertex normals in internal buffer.
        uint32_t      m_numVertexNormals;
        /// Number of vertex colors sets in internal buffer.
        uint32_t      m_numVertexColors;
        /// Number of vertex confidence values in internal buffer.
        uint32_t      m_numVertexConfidence;
        /// Number of vertex intensities in internal buffer.
        uint32_t      m_numVertexIntensity;
        /// Number of faces in internal buffer.
        uint32_t      m_numFace;

};

} /* namespace lssr */
#endif /* MESHIO_HPP_ */
