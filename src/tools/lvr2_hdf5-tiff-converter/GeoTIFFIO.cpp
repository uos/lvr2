//
// Created by ndettmer on 07.02.19.
//

#include <geotiff.h>
#include <xtiffio.h>

#include "GeoTIFFIO.hpp"

using namespace lvr2;

GeoTIFFIO::GeoTIFFIO(std::string filename)
{
    m_tiffile = XTIFFOpen(filename.c_str(), "w");
    m_gtiffile = GTIFNew(m_tiffile);
}

GeoTIFFIO::~GeoTIFFIO()
{
    XTIFFClose(m_tiffile);
    GTIFFree(m_gtiffile);
}

