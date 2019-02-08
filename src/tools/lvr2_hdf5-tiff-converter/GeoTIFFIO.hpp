//
// Created by ndettmer on 07.02.19.
//

#ifndef GEOTIFFIO_HPP
#define GEOTIFFIO_HPP

#include <geotiff.h>
#include <geotiffio.h>

namespace lvr2
{
    class GeoTIFFIO
    {
    public:
        GeoTIFFIO();

    private:
        GTIF *m_geofile;
    };
}


#endif //GEOTIFFIO_HPP
