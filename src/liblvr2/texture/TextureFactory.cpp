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

/**
 * TextureFactory.cpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 *  @date 15.02.2012
 *  @author Denis Meyer
 */

#include "lvr2/texture/TextureFactory.hpp"

#include "lvr2/io/PPMIO.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/texture/Texture.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
using std::cout;
using std::endl;

namespace lvr2
{

Texture TextureFactory::readTexture(std::string filename)
{
    // returns 8-bit image with BGR order
    cv::Mat mat = cv::imread(filename);


    // if unable to read file
    if (mat.data == NULL)
    {
        cout << timestamp << "TextureFactory: Unable to read file '"
            << filename << "'. Returning empty Texture." << endl;

        // return empty Texture
        return Texture();
    }

    // convert it to RGB order
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    Texture ret(0, mat.cols, mat.rows, 3, 1, 1.0);
    std::copy(mat.datastart, mat.dataend, ret.m_data);

    return ret;
}

void TextureFactory::saveTexture(const Texture& tex, std::string filename)
{
    // if Texture has no data to be saved
    if (tex.m_data == NULL || tex.m_width == 0 || tex.m_height == 0 ||
        tex.m_numChannels == 0 || tex.m_numBytesPerChan == 0)
    {
        cout << timestamp << "TextureFactory: Texture will not be saved to file '"
            << filename << "' because the texture has no data." << endl;

        return;
    }

    // TODO convert the data instead of only allowing 1 byte channels
    if (tex.m_numBytesPerChan != 1)
    {
        cout << timestamp << "TextureFactory: Texture will not be saved to file '"
            << filename << "' because texture has more than 1 byte \
            per channel (currently only 1-byte channels are supported)." << endl;

        return;
    }

    if (tex.m_numChannels != 1 && tex.m_numChannels != 3 && tex.m_numChannels != 4)
    {
        cout << timestamp << "TextureFactory: Texture will not be saved to file '"
            << filename << "' because the texture has an unsupported amount of channels \
            (currently only 1, 3 and 4 channels per pixel are supported)." << endl;

        return;
    }

    // convert texture data to be saved with OpenCV
    int mat_type = CV_8UC1;
    int conversionCode = cv::COLOR_GRAY2BGR;

    if (tex.m_numChannels == 3)
    {
        mat_type = CV_8UC3;
        conversionCode = cv::COLOR_RGB2BGR;
    }
    else if (tex.m_numChannels == 4)
    {
        // currently we just ignore the alpha channel but it would be
        // possible to save the alpha channel with OpenCV for PNG images.
        mat_type = CV_8UC3;
        conversionCode = cv::COLOR_RGBA2BGR;
    }

    cv::Mat mat;
    mat.create(tex.m_height, tex.m_width, mat_type);

    size_t bytesToCopy = tex.m_width * tex.m_height * tex.m_numChannels * tex.m_numBytesPerChan;
    std::copy(tex.m_data, tex.m_data + bytesToCopy, mat.data);

    if (!(mat_type == CV_8UC1))
    {
        cv::cvtColor(mat, mat, conversionCode);
    }

    // todo include params like binary mode for ppm files for example...
    if (!cv::imwrite(filename, mat))
    {
        cout << timestamp << "TextureFactory: Unable to save texture to file '"
            << filename << "'." << endl;
    };
}

} // namespace lvr2
