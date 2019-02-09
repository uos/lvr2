//
// Created by ndettmer on 07.02.19.
//

#ifndef GEOTIFFIO_HPP
#define GEOTIFFIO_HPP

#include <geotiffio.h>
/*#include <tiffio.h>*/
#include <string>
#include "TIFFIO.hpp"

namespace lvr2
{
    class GeoTIFFIO
    {
    public:
        GeoTIFFIO(std::string filename);

        ~GeoTIFFIO();

    private:
        GTIF *m_gtiffile;
        TIFF *m_tiffile;
    };
}


#endif //GEOTIFFIO_HPP
