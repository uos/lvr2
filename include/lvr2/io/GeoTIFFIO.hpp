/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef GEOTIFFIO_HPP
#define GEOTIFFIO_HPP

#include <gdal_priv.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace lvr2
{
/**
 * @brief class providing and encapsulating GDAL GeoTIFF I/O functions
 * @author Niklas Dettmer <ndettmer@uos.de>
 */
class GeoTIFFIO
{
public:
    /**
     * @param filename filename of output GeoTIFF file
     * @param cols number of columns / width of the image
     * @param rows number of rows / length of the image
     * @param bands number of bands
     */
    GeoTIFFIO(std::string filename, int cols, int rows, int bands);

    /**
     * @param filename
     */
    GeoTIFFIO(std::string filename);

    /**
     * @brief Writing given band into open GeoTIFF file
     * @param mat cv::Mat containing the band data
     * @param band number of band to be written
     * @return standard C++ return value
     */
    int writeBand(cv::Mat *mat, int band);

    /**
     * @return width of dataset in number of pixels
     */
    int getRasterWidth();

    /**
     * @return height of dataset in number of pixels
     */
    int getRasterHeight();

    /**
     * @return number of bands of dataset
     */
    int getNumBands();

    /**
     * @param band_index index of the band to be read
     * @return indexed band of the dataset as cv::Mat *
     */
    cv::Mat *readBand(int band_index);

    ~GeoTIFFIO();

private:
    GDALDataset *m_gtif_dataset;
    GDALDriver *m_gtif_driver;
    int m_cols, m_rows, m_bands;
};
}


#endif //GEOTIFFIO_HPP
