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

int TIFFIO::writeLevel(cv::Mat *mat)
{
    // TODO: make configurable?
    int sampleperpixel = 1;

    setFields(mat, sampleperpixel);

    tsize_t linebytes = sampleperpixel * mat->cols;
    unsigned int *buf = NULL;

    if(TIFFScanlineSize(m_tiffile))
    {
        buf = (unsigned int *)_TIFFmalloc(linebytes);
    } else {
        buf = (unsigned int *)_TIFFmalloc(TIFFScanlineSize(m_tiffile));
    }

    // write
    for (uint32 row = 0; row < mat->rows; row++)
    {
        unsigned int *row_content = (unsigned int *)mat->row(row).data;
        memcpy(buf, &row_content, linebytes);
        if(TIFFWriteScanline(m_tiffile, buf, row, 0) < 0)
        {
            std::cout << "An error occurred while writing the tiff file in row " << row << "." << std::endl;
            break;
        }
    }

    if(buf)
    {
        _TIFFfree(buf);
    }
}

int TIFFIO::setFields(cv::Mat *mat, int sampleperpixel)
{
    if (!m_tiffile)
    {
        return -1;
    }

    // TODO: check if fields are set already
    TIFFSetField(m_tiffile, TIFFTAG_IMAGEWIDTH, mat->cols);
    TIFFSetField(m_tiffile, TIFFTAG_IMAGELENGTH, mat->rows);
    TIFFSetField(m_tiffile, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel); // number of channels per pixel
    TIFFSetField(m_tiffile, TIFFTAG_BITSPERSAMPLE, 8); // size of the channel
    TIFFSetField(m_tiffile, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    // TODO: understand / change RGB to gray
    TIFFSetField(m_tiffile, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(m_tiffile, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(m_tiffile, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(m_tiffile, mat->cols * sampleperpixel));
}

