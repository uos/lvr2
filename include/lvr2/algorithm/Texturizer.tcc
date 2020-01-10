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

/*
* Texturizer.tcc
*
*  @date 15.09.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/algorithm/ColorAlgorithms.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


namespace lvr2
{

template<typename BaseVecT>
Texturizer<BaseVecT>::Texturizer(
    float texelSize,
    int texMinClusterSize,
    int texMaxClusterSize
) :
    m_texelSize(texelSize),
    m_texMinClusterSize(texMinClusterSize),
    m_texMaxClusterSize(texMaxClusterSize)
{
}



template<typename BaseVecT>
Texture Texturizer<BaseVecT>::getTexture(TextureHandle h)
{
    return m_textures[h];
}

template<typename BaseVecT>
StableVector<TextureHandle, Texture> Texturizer<BaseVecT>::getTextures()
{
    return m_textures;
}

template<typename BaseVecT>
int Texturizer<BaseVecT>::getTextureIndex(TextureHandle h)
{
    return m_textures[h].m_index;
}

template<typename BaseVecT>
void Texturizer<BaseVecT>::saveTextures()
{
    string comment = timestamp.getElapsedTime() + "Saving textures ";
    ProgressBar progress(m_textures.numUsed(), comment);
    for (auto h : m_textures)
    {
        m_textures[h].save();
        ++progress;
    }
    std::cout << std::endl;
}

template<typename BaseVecT>
TexCoords Texturizer<BaseVecT>::calculateTexCoords(
    TextureHandle h,
    const BoundingRectangle<typename BaseVecT::CoordType>& br,
    BaseVecT point
)
{
    //return m_textures[h].calcTexCoords(boundingRect, v);
    auto texelSize = m_textures[h].m_texelSize;
    auto width = m_textures[h].m_width;
    auto height = m_textures[h].m_height;

    BaseVecT w =  point - ((br.m_vec1 * br.m_minDistA) + (br.m_vec2 * br.m_minDistB)
            + br.m_supportVector);
    float u = (br.m_vec1 * (w.dot(br.m_vec1))).length() / texelSize / width;
    float v = (br.m_vec2 * (w.dot(br.m_vec2))).length() / texelSize / height;

    return TexCoords(u,v);
}

template<typename BaseVecT>
BaseVecT Texturizer<BaseVecT>::calculateTexCoordsInv(
    TextureHandle h,
    const BoundingRectangle<typename BaseVecT::CoordType>& br,
    const TexCoords& coords
)
{
    return br.m_supportVector + (br.m_vec1 * br.m_minDistA)
                              + br.m_vec1 * coords.u
                                          * (br.m_maxDistA - br.m_minDistA + m_texelSize / 2.0)
                              + (br.m_vec2 * br.m_minDistB)
                              + br.m_vec2 * coords.v
                                          * (br.m_maxDistB - br.m_minDistB - m_texelSize / 2.0);
}

template<typename BaseVecT>
TextureHandle Texturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>& surface,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
)
{
    cout << "Wrong" << endl;
    // Calculate the texture size
    unsigned short int sizeX = ceil((boundingRect.m_maxDistA - boundingRect.m_minDistA) / m_texelSize);
    unsigned short int sizeY = ceil((boundingRect.m_maxDistB - boundingRect.m_minDistB) / m_texelSize);

    // Create texture
    Texture texture(index, sizeX, sizeY, 3, 1, m_texelSize);

    string comment = timestamp.getElapsedTime() + "Computing texture pixels ";
    ProgressBar progress(sizeX * sizeY, comment);

    if (surface.pointBuffer()->hasColors())
    {
        UCharChannel colors = *(surface.pointBuffer()->getUCharChannel("colors"));

        // For each texel find the color of the nearest point
        #pragma omp parallel for schedule(dynamic,1) collapse(2)
        for (int y = 0; y < sizeY; y++)
        {
            for (int x = 0; x < sizeX; x++)
            {
                int k = 1; // k-nearest-neighbors

                vector<size_t> cv;

                BaseVecT currentPos =
                    boundingRect.m_supportVector
                    + boundingRect.m_vec1 * (x * m_texelSize + boundingRect.m_minDistA - m_texelSize / 2.0)
                    + boundingRect.m_vec2 * (y * m_texelSize + boundingRect.m_minDistB - m_texelSize / 2.0);

                surface.searchTree()->kSearch(currentPos, k, cv);

                uint8_t r = 0, g = 0, b = 0;

                for (size_t pointIdx : cv)
                {
                    auto cur_color = colors[pointIdx];
                    r += cur_color[0];
                    g += cur_color[1];
                    b += cur_color[2];
                }

                r /= k;
                g /= k;
                b /= k;

                texture.m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = r;
                texture.m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = g;
                texture.m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = b;

                ++progress;
            }
        }
        std::cout << std::endl;
    }
    else
    {
        for (int i = 0; i < sizeX * sizeY; i++)
        {
            texture.m_data[i] = 0;
        }
    }

    return m_textures.push(texture);
}


template<typename BaseVecT>
void Texturizer<BaseVecT>::findKeyPointsInTexture(const TextureHandle texH,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
        const cv::Ptr<cv::Feature2D>& detector,
        std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    const Texture texture = m_textures[texH];
    if (texture.m_height <= 32 && texture.m_width <= 32)
    {
        return;
    }

    const unsigned char* img_data = texture.m_data;
    cv::Mat image(texture.m_height, texture.m_width, CV_8UC3, (void*)img_data);

    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

template<typename BaseVecT>
std::vector<BaseVecT> Texturizer<BaseVecT>::keypoints23d(const std::vector<cv::KeyPoint>&
        keypoints, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, const TextureHandle& h)
{
    const size_t N = keypoints.size();
    std::vector<BaseVecT> keypoints3d(N);
    const int width            = m_textures[h].m_width;
    const int height           = m_textures[h].m_height;

    for (size_t p_idx = 0; p_idx < N; ++p_idx)
    {
        const cv::Point2f keypoint = keypoints[p_idx].pt;
        // Calculate texture coordinates from pixel locations and then calculate backwards
        // to 3D coordinates
        const float u = keypoint.x / width;
        // I'm not sure why we need to mirror this coordinate, but it works like
        // this
        const float v      = 1 - keypoint.y / height;
        BaseVecT location  = calculateTexCoordsInv(h, boundingRect, TexCoords(u, v));
        keypoints3d[p_idx] = location;
    }
    return keypoints3d;
}

} // namespace lvr2
