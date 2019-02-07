//
// Created by hitech-gandalf on 05.02.19.
//

#include <iostream>

#include "TIFFIO.hpp"

using namespace lvr2;

TIFFIO::TIFFIO(std::string filename)
{
    m_tiffile = TIFFOpen(filename.c_str(), "w");
}

TIFFIO::~TIFFIO()
{
    if (m_tiffile)
    {
        TIFFClose(m_tiffile);
    }
}

int TIFFIO::writePage(cv::Mat *mat, int page)
{
    int sampleperpixel = 1;

    tsize_t linebytes = sampleperpixel * mat->cols;

/*    TIFFSetField(m_tiffile, TIFFTAG_PAGENUMBER, page, 150);*/

    // write
    for (uint32_t row = 0; row < mat->rows; row++)
    {
        uint8_t *row_content = &mat->row(row).at<uint8_t>(0);
        if(TIFFWriteScanline(m_tiffile, &row_content[mat->rows * linebytes], row, 0) < 0)
        {
            std::cout << "An error occurred while writing the tiff file in row " << row << "." << std::endl;
            return -1;
        }
    }
    return 0;
}

int TIFFIO::setFields(int height, int width, int sampleperpixel)
{
    if (!m_tiffile)
    {
        return -1;
    }

    TIFFSetField(m_tiffile, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(m_tiffile, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(m_tiffile, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel); // number of channels per pixel
    TIFFSetField(m_tiffile, TIFFTAG_BITSPERSAMPLE, 8); // size of the channel
    TIFFSetField(m_tiffile, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(m_tiffile, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(m_tiffile, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(m_tiffile, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(m_tiffile, width * sampleperpixel));
    TIFFSetField(m_tiffile, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);

    return 0;
}

