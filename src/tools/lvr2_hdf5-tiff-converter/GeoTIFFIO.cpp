//
// Created by ndettmer on 07.02.19.
//

#include <iostream>

#include "GeoTIFFIO.hpp"

using namespace lvr2;

GeoTIFFIO::GeoTIFFIO(std::string filename, int cols, int rows, int bands) : m_cols(cols), m_rows(rows), m_bands(bands)
{
    GDALAllRegister();
    m_gtif_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    m_gtif_dataset = m_gtif_driver->Create(filename.c_str(), m_cols, m_rows, m_bands, GDT_UInt16, NULL);
    // TODO: SetGeoTransform , SetGeoProjection ??
}

int GeoTIFFIO::writeBand(cv::Mat *mat, int band)
{
    if (!m_gtif_dataset)
    {
        std::cout << "GeoTIFF dataset not initialized!" << std::endl;
        return -1;
    }

    uint16_t *rowBuff = (uint16_t *) CPLMalloc(sizeof(uint16_t) * m_cols);
    for (int row = 0; row < m_rows; row++)
    {
        for (int col = 0; col < m_cols; col++)
        {
            rowBuff[col] = mat->at<uint16_t>(row, col);
        }
        if (m_gtif_dataset->GetRasterBand(band)->RasterIO(
                GF_Write, 0, row, m_cols, 1, rowBuff, m_cols, 1, GDT_UInt16, 0, 0) != CPLE_None)
        {
            std::cout << "An error occurred in GDAL while writing band "
                << band << " in row " << row << "." << std::endl;
            return -1;
        }
    }
    return 0;
}

GeoTIFFIO::~GeoTIFFIO()
{
    GDALClose(m_gtif_dataset);
    GDALDestroyDriverManager();
}

