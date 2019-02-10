//
// Created by ndettmer on 07.02.19.
//

#ifndef GEOTIFFIO_HPP
#define GEOTIFFIO_HPP

#include <gdal/gdal_priv.h>
#include <cv.h>

#include <string>

namespace lvr2
{
    class GeoTIFFIO
    {
    public:
        GeoTIFFIO(std::string filename, int cols, int rows, int bands);

        int writeBand(cv::Mat *mat, int band);

        ~GeoTIFFIO();

    private:
        GDALDataset *m_gtif_dataset;
        GDALDriver *m_gtif_driver;
        int m_cols, m_rows, m_bands;
    };
}


#endif //GEOTIFFIO_HPP
