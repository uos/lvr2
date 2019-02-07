//
// Created by hitech-gandalf on 05.02.19.
//

#ifndef TIFFIO_HPP
#define TIFFIO_HPP

#include <tiffio.h>
#include <opencv2/core/core.hpp>
#include <string>

namespace lvr2
{

/**
 * @brief wrapper class for TIFFIO functions
 * @author ndettmer <ndettmer@uos.de>
 */
    class TIFFIO {
    public:
        TIFFIO(std::string filename);

        ~TIFFIO();

        /**
         * @brief write the next scanline to the tiff file
         * @return true if no errors occurred
         */
        int writePage(cv::Mat *mat, int page);

        int setFields(int height, int width, int sampleperpixel);

    private:
        TIFF* m_tiffile;
    };

}

#endif //TIFFIO_HPP
