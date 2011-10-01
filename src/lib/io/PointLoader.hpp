/**
 *
 * @file      PointLoader.hpp
 * @brief     Interface for all point loading classes.
 * @details   The PointLoader class specifies storage and access to all
 *            available point data by implementing the appropriate  get and set
 *            methods for these data.
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   111001
 * @date      Recreated:     2011-09-22 23:23:57
 * @date      Last modified: 2011-10-01 15:22:49
 *
 **/

#ifndef POINTIO_HPP_
#define POINTIO_HPP_

#include <stdint.h>
#include <cstddef>
#include <cstdlib>

namespace lssr
{

/**
 * \class PointLoader PointLoader.hpp "io/PointLoader.hpp"
 * \brief Interface for all point loading classes.
 * \todo  At the moment this class comes along with a lot of possible memory
 *        leaks. To prevent those all data should be stored as \c shared_ptr as
 *        introduced by C++11.
 *
 * The PointLoader class specifies storage and access to all available point
 * data by implementing the appropriate get and set methods for these data.
 * This class is supposed to be the superclass of all point loading I/O
 * classes.
 **/
class PointLoader {

    public:
        /**
         * \brief Constructor.
         *
         * The default constructor. This clears all internal data.
         **/
        PointLoader();


        /**
         * \brief Set the point array.
         *
         * By using setPointArray the internal point buffer can be set. The
         * point array has to be a one dimensional float array containing sets
         * of \c x, \c y and \c z values.
         *
         * \param array  Pointer to interlaced point data.
         * \param n      Amount of points in the array.
         **/
        void setPointArray( float* array, size_t n );


        /**
         * \brief Set the point color array.
         *
         * By using setPointColorArray the internal point color buffer can be
         * set. The array has to be a one dimensional uint8_t array containing
         * sets of three values for \c red, \c green and \c blue. The values
         * can be in the range of [0..255].
         *
         * \param array  Pointer to interlaced point color data.
         * \param n      Amount of color information in the array.
         **/
        void setPointColorArray( uint8_t* array, size_t n );


        /**
         * \brief Set the point normal array.
         *
         * By using setPointNormalArray the internal point normal buffer can
         * be set. The array has to be a one dimensional float array containing
         * sets of \c x, \c y and \c z values.
         *
         * \param array  Pointer to interlaced point normal data.
         * \param n      Amount of normals in the array.
         **/
        void setPointNormalArray( float* array, size_t n );


        /**
         * \brief Set the point intensity array.
         *
         * By using setPointIntensityArray the internal intensity buffer for
         * points can be set. The array has to be a one dimensional float
         * array.
         *
         * \param array  Pointer to point intensity data.
         * \param n      Amount of data in the array.
         **/
        void setPointIntensityArray( float* array, size_t n );


        /**
         * \brief Set the point confidence array.
         *
         * By using setPointConfidenceArray the internal confidence buffer for
         * points can be set. The array has to be a one dimensional float
         * array.
         *
         * \param array  Pointer to point confidence data.
         * \param n      Amount of data in the array.
         **/
        void setPointConfidenceArray( float* array, size_t n );


        /************************* Get *************************/


        /**
         * \brief Get the point array.
         *
         * Using getPointArray the point data can be retrieved. The returned
         * array is a one dimensional interlaced float array containing sets of
         * \c x, \c y and \c z values. Additionally the passed reference of a
         * size_t variable is set to the amount of points stored in the array.
         * Thus \c n is set to one third of the array length.
         *
         * \param n  Amount of points in array.
         * \return   %Point array.
         **/
        virtual float* getPointArray( size_t &n );


        /**
         * \brief Get the point color array.
         *
         * getPointColorArray returns the point color data. The returned array
         * is a one dimensional interlaced uint8_t array containing sets of \c
         * red, \c green and \c blue values. Additionally the passed reference
         * of a size_t variable is set to the amount of point color sets stored
         * in the array. Thus \c n is set to one third of the array length.
         *
         * \param n  Amount of point color sets in array.
         * \return   %Point color array.
         **/
        virtual uint8_t* getPointColorArray( size_t &n );


        /**
         * \brief Get the point normal array.
         *
         * getPointNormalArray returns the point normal data. The returned
         * array is a one dimensional interlaced float array containing sets of
         * \c x, \c y and \c z values. Additionally the passed reference of a
         * size_t variable is set to the amount of point normals stored in the
         * array. Thus \c n is set to one third of the array length.
         *
         * \param n  Amount of point normals in array.
         * \return   %Point normal array.
         **/
        virtual float* getPointNormalArray( size_t &n );


        /**
         * \brief Get the point intensity array.
         *
         * getPointiIntensityArray returns the point intensity informations.
         * The returned array is a one dimensional float array.  Additionally
         * the passed reference of a size_t variable is set to the amount of
         * intensity values stored in the array. Thus \c n is set to the array
         * length.
         *
         * \param n  Amount of intensity values in array.
         * \return   %Point intensity array.
         **/
        virtual float* getPointIntensityArray( size_t &n );


        /**
         * \brief Get the point confidence array.
         *
         * getPointConfidenceArray returns the point confidence informations.
         * The returned array is a one dimensional float array. Additionally
         * the passed reference of a size_t variable is set to the amount of
         * confidence values stored in the array. Thus \c n is set to the array
         * length.
         *
         * \param n  Amount of confidence values in array.
         * \return   %Point confidence array.
         **/
        virtual float* getPointConfidenceArray( size_t &n );


        /************************* Indexed Get *************************/


        /**
         * \brief Get indexed point array.
         *
         * This method return a two dimensional float array containing the
         * point data. The actual data is the same as returned by
         * getPointArray. Additionally \c n is set to the amount of vertices
         * and thus to the length of the array.
         *
         * \param n  Amount of points in array.
         * \return   Indexed point array.
         **/
        float** getIndexedPointArray( size_t &n );


        /**
         * \brief Get indexed point color array.
         *
         * This method return a two dimensional float array containing the
         * point color information. The actual data is the same as returned by
         * \ref getPointColorArray. Additionally \c n is set to the amount of
         * data and thus to the length of the array.
         *
         * \param n  Amount of point color sets in array.
         * \return   Indexed point color array.
         **/
        uint8_t** getIndexedPointColorArray( size_t &n );


        /**
         * \brief Get indexed point normal array.
         *
         * This method return a two dimensional float array containing the
         * point normals. The actual data is the same as returned by
         * getPointNormalArray. Additionally \c n is set to the amount of
         * point normals and thus to the length of the array.
         *
         * \param n  Amount of point normals in array.
         * \return   Indexed point normal array.
         **/
        float** getIndexedPointNormalArray( size_t &n );


        /**
         * \brief Get indexed point intensity array.
         * \note  As each of the internal arrays only contain one value it is
         *        strongly suggested to use \ref getPointIntensityArray
         *        instead. That way the overhead introduced by this method is
         *        omitted and the data can be accessed even more easier as with
         *        this method.
         *
         * This method return a two dimensional float array containing the
         * point intensity information. The actual data is the same as
         * returned by \ref getPointIntensityArray. Additionally \c n is set to
         * the amount of data and thus to the length of the array.
         *
         * \param n  Amount of point intensities in array.
         * \return   Indexed point intensity array.
         **/
        float** getIndexedPointIntensityArray( size_t &n );


        /**
         * \brief Get indexed point confidence array.
         * \note  As each of the internal arrays only contain one value it is
         *        strongly suggested to use \ref getPointConfidenceArray
         *        instead. That way the overhead introduced by this method is
         *        omitted and the data can be accessed even more easier as with
         *        this method.
         *
         * This method return a two dimensional float array containing the
         * point confidence data. The actual data is the same as returned by
         * getPointConfidenceArray. Additionally \c n is set to the amount of
         * data and thus to the length of the array.
         *
         * \param n  Amount of point confidence data in array.
         * \return   Indexed point confidence array.
         **/
        float** getIndexedPointConfidenceArray( size_t &n );

        
        /**
         * \brief Clear internal point buffers.
         *
         * This method clears all internal buffers and the stored buffer
         * length. That means all pointers are set to null-pointers and all
         * length variables are set to zero.
         * \warning This method does not free any allocated memory.
         **/
        void freeBuffer();


        /**
         * \brief Get the amount of points.
         * \return Amount of points.
         **/
        size_t getNumPoints();

    protected:


        /**
         * \brief Create and return indexed array for given data.
         *
         * This method returns and if necessary generates a two dimensional
         * float array from given data.
         *
         * \param n      Amount of data in 2d-array.
         * \param num    Amount of data in 1d array.
         * \param arr1d  Pointer to one dimensional float array containing the
         *               data.
         * \param arr2d  Pointer to two dimensional float array to be created.
         * \param step   Amount of values in one set of data.
         * \return       Indexed array.
         **/
        float** getIndexedArrayf( size_t &n, const size_t num, float** arr1d,
                float*** arr2d, const int step = 3 );

        /// %Point buffer.
        float*    m_points;
        /// %Point normal buffer.
        float*    m_pointNormals;
        /// %Point color buffer.
        uint8_t*  m_pointColors;
        /// %Point intensity buffer.
        float*    m_pointIntensities;
        /// %Point confidence buffer.
        float*    m_pointConfidence;

        /// Indexed point buffer.
        float**   m_indexedPoints;
        /// Indexed point normal buffer.
        float**   m_indexedPointNormals;
        /// Indexed point intensity buffer.
        float**   m_indexedPointIntensities;
        /// Indexed point confidence buffer.
        float**   m_indexedPointConfidence;
        /// Indexed point color buffer.
        uint8_t** m_indexedPointColors;

        /// Number of points in internal buffer.
        size_t    m_numPoints;
        /// Number of point color sets in internal buffer.
        size_t    m_numPointColors;
        /// Number of point normals in internal buffer.
        size_t    m_numPointNormals;
        /// Number of point intensity values in internal buffer.
        size_t    m_numPointIntensities;
        /// Number of point confidence values in internal buffer.
        size_t    m_numPointConfidence;

};

} /* namespace lssr */

#endif /* POINTIO_HPP_ */
