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


 /**
 *
 * @file      PointLoader.hpp
 * @brief     Interface for all point loading classes.
 * @details   The PointLoader class specifies storage and access to all
 *            available point data by implementing the appropriate  get and set
 *            methods for these data.
 *
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @author    Thomas Wiemann, twiemann@uos.de, Universität Osnabrück
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

#include <vector>
#include <algorithm>
#include "DataStruct.hpp"

#include <boost/shared_ptr.hpp>

namespace lvr
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
class PointBuffer
{

public:
    /**
     * \brief Constructor.
     *
     * The default constructor. This clears all internal data.
     **/
    PointBuffer();

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
    void setPointArray( floatArr array, size_t n );


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
    void setPointColorArray( ucharArr array, size_t n );


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
    void setPointNormalArray( floatArr array, size_t n );


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
    void setPointIntensityArray( floatArr array, size_t n );


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
    void setPointConfidenceArray( floatArr array, size_t n );


    /**
     * \brief Set the point spectral channel array.
     *
     * By using setPointSpectralChannelsArray the internal spectral channel
     * buffer for points can be set. The array has to be a one dimensional
     * float array.
     *
     * \param array         Pointer to point spectral channel data.
     * \param n             Amount of data in the array.
     * \param n_channels    Number of channels per Point.
     * \param minWavelength smallest Wavelength
     * \param maxWavelength largest Wavelength
     **/
    void setPointSpectralChannelsArray( floatArr array, size_t n, size_t n_channels, int minWavelength, int maxWavelength );


    /************************* Indexed Set *************************/


    /**
     * \brief Set the point color array.
     *
     * By using setIndexedPointColorArray the internal point color buffer
     * can be set. The array has to be a shared color<uchar> array
     * containing sets of three values for \c red, \c green and \c blue.
     * The values can be in the range of [0..255].
     *
     * \param array  Pointer to point color data.
     * \param n      Number of color triples in the array
     **/
    void setIndexedPointColorArray( color3bArr array, size_t n );

    /**
     * \brief Set the point array.
     *
     * By using setIndexedPointArray the internal point color buffer
     * can be set.
     * \param array  Pointer to a point array
     * \param n      Number of points in the array
     **/
    void setIndexedPointArray( coord3fArr array, size_t n);

    /**
     * \brief Set the point array.
     *
     * By using setIndexedPointArray the internal point color buffer
     * can be set.
     * \param array  Pointer to a point normal array
     * \param n      Number of points in the array
     **/
    void setIndexedPointNormalArray( coord3fArr array, size_t n);



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
    virtual floatArr getPointArray( size_t &n );


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
    virtual ucharArr getPointColorArray( size_t &n );


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
    virtual floatArr getPointNormalArray( size_t &n );


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
    virtual floatArr getPointIntensityArray( size_t &n );


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
    floatArr getPointConfidenceArray( size_t &n );


    /**
     * \brief Get the point spectral channels array.
     *
     * getPointSpectralChannelsArray returns the spectral channel informations.
     * The returned array is a one dimensional unsigned char array.
     * Additionally the passed reference of a size_t variable is set to the
     * amount of spectral channel values stored in the array. Thus \c n is
     * set to the array length. \c n_channels is set to the number of channels
     * that each Point has.
     *
     * \param n          Amount of spectral channel values in array.
     * \param n_channels Number of Channels per Point.
     * \return           %Point spectral channel array.
     **/
    floatArr getPointSpectralChannelsArray( size_t &n, size_t &n_channels );

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
    coord3fArr getIndexedPointArray( size_t &n );


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
    color3bArr getIndexedPointColorArray( size_t &n );


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
    coord3fArr getIndexedPointNormalArray( size_t &n );


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
    idx1fArr getIndexedPointIntensityArray( size_t &n );


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
    idx1fArr getIndexedPointConfidenceArray( size_t &n );


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

    virtual std::vector<indexPair>& getSubClouds()
                {
        return m_subClouds;
                }

    /**
     * @brief   The pair of given indices defines a sub point cloud.
     */
    void defineSubCloud( indexPair &range );

    /**
     * @brief   Returns true if the stored data contains point normal
     *          information
     */
    bool hasPointNormals() { return m_numPointNormals != 0;}


    /**
     * @brief   Returns true if the stored data contains point color
     *          information
     */
    bool hasPointColors() { return m_numPointColors != 0;}

    /**
     * @brief   Returns true if the stored data contains point spectral
     *          channel information
     */
    bool hasPointSpectralChannels() { return m_numPointSpectralChannels != 0;}

    /**
     * @brief   Returns the Wavelength of channel 0 in nm
     */
    int getMinWavelength() { return m_minWavelength; }

    /**
     * @brief   Returns the Wavelength that channel m_numSpectralChannels would have
     *          in nm
     */
    int getMaxWavelength() { return m_maxWavelength; }

    /**
     * @brief   Returns the Wavelength difference in nm between two adjacent channels
     */
    int numWavelengthsPerChannel();

    /**
     * @brief   Returns the channel index of the given wavelength, or -1 if wavelength
     *          is not in [minWavelength, maxWavelength)
     *
     * @param wavelength the wavelength in nm
     */
    int getChannel(int wavelength);

    /**
     * @brief   Returns the wavelength index of the given channel, or -1 if channel
     *          is not in [0, n_channels)
     *
     * @param channel the channel
     */
    int getWavelength(int channel);


protected:

    /// %Point buffer.
    floatArr        m_points;
    /// %Point normal buffer.
    floatArr        m_pointNormals;
    /// %Point color buffer.
    ucharArr        m_pointColors;
    /// %Point intensity buffer.
    floatArr        m_pointIntensities;
    /// %Point confidence buffer.
    floatArr        m_pointConfidences;
    /// %Point spectral channel buffer.
    floatArr        m_pointSpectralChannels;


    /// Number of points in internal buffer.
    size_t          m_numPoints;
    /// Number of point color sets in internal buffer.
    size_t          m_numPointColors;
    /// Number of point normals in internal buffer.
    size_t          m_numPointNormals;
    /// Number of point intensity values in internal buffer.
    size_t          m_numPointIntensities;
    /// Number of point confidence values in internal buffer.
    size_t          m_numPointConfidence;
    /// Number of spectral channel Datasets in internal buffer.
    size_t          m_numPointSpectralChannels;
    /// Number of spectral channels in each spectral channel Dataset.
    size_t          m_numSpectralChannels;

    /// Wavelength of channel 0 in nm
    int m_minWavelength;
    /// Wavelength of channel m_numSpectralChannels in nm
    int m_maxWavelength;

    /// Vector to save the indices of the first and last points of single scans
    std::vector<indexPair> m_subClouds;

};

typedef boost::shared_ptr<PointBuffer> PointBufferPtr;

} /* namespace lvr */

#endif /* POINTIO_HPP_ */
