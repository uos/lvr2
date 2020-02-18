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
 * ModelToImage.h
 *
 *  Created on: Jan 25, 2017
 *      Author: Thomas Wiemann (twiemann@uos.de)
 */

#ifndef SRC_LIBLVR2_RECONSTRUCTION_MODELTOIMAGE_HPP_
#define SRC_LIBLVR2_RECONSTRUCTION_MODELTOIMAGE_HPP_

#include "lvr2/io/Model.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <vector>
#include <tuple>
using std::vector;
using std::tuple;

namespace lvr2
{

class Projection;

///
/// \brief  The ModelToImage class provides methods to re-project 3D laser scans
///         to image planes.
///
class ModelToImage {
public:

    /// Pixelcoordinates with depth value
    typedef struct
    {
        int i;
        int j;
        float depth;
    } DepthPixel;

    typedef struct PanoPoint
    {
        PanoPoint(size_t index_) : index(index_) {}
        size_t index;
    } PanoramaPoint;

    /// Image with single depth information
    typedef struct DI
    {
        vector<vector<float> > pixels;
        float   maxRange;
        float   minRange;
        DI() :
            maxRange(std::numeric_limits<float>::lowest()),
            minRange(std::numeric_limits<float>::max()) {}

    } DepthImage;

    /// Image with list of projected points at each pixel
    typedef struct PLI
    {
        vector<vector<vector<PanoramaPoint> > > pixels;
        float   maxRange;
        float   minRange;
        PLI() :
            maxRange(std::numeric_limits<float>::lowest()),
            minRange(std::numeric_limits<float>::max()) {}
    } DepthListMatrix;


    ///
    /// \brief The ProjectionType enum
    ///
    enum ProjectionType {
        CYLINDRICAL, CONICAL, EQUALAREACYLINDRICAL,
        RECTILINEAR, PANNINI, STEREOGRAPHIC,
        ZAXIS, AZIMUTHAL
    };

    enum ProjectionPolicy
    {
        FIRST, LAST, MINRANGE, MAXRANGE, AVERAGE, COLOR, INTENSITY
    };

    enum CoordinateSystem
    {
        NATIVE, SLAM6D, UOS
    };


    ///
    /// \brief Constructor
    /// \param buffer               A point buffer containing a point cloud
    /// \param projection           Type of used projection
    /// \param width                Desired image with. May be changed to fit
    ///                             panorama angles
    /// \param height               Desired image height. May be changed to fit
    ///                             panorama angles
    /// \param minZ                 Minimum depth value
    /// \param maxZ                 Maximum depth value
    /// \param minHorizontenAngle   Start of horizontal field of view in degrees
    /// \param maxHorizontalAngle   End of horizontal field of view in degrees
    /// \param mainVerticalAngle    Start of vertical field of view in degrees
    /// \param maxVerticalAngle     End of vertical field of view in degrees
    /// \param imageOptimization    If true, the aspect ration will be adapted to
    ///                             the chosen field of view
    /// \param leftHandedInputData  Set this to true of the scan points are in a
    ///                             left handed coordinate system (like 3dtk)
    ///
    ModelToImage(
            PointBufferPtr buffer,
            ProjectionType projection,
            int width, int height,
            float minZ, float maxZ,
            int minHorizontenAngle, int maxHorizontalAngle,
            int mainVerticalAngle, int maxVerticalAngle,
            bool imageOptimization = true,
            CoordinateSystem system = NATIVE);

    ///
    /// \brief Writes the scan panaroma to an pgm file
    ///
    /// \param filename     Filename of the bitmap
    /// \param cutoff       Max range cutoff. Reduce this to enhance contrast on
    ///                     pixels with low depths.
    ///
    void writePGM(string filename, float cutoff);

    /// Destructor
    virtual ~ModelToImage();

    //// Returns an OpenCV image representation of the panarama
    void getCVMatrix(cv::Mat& image);


    ///
    /// \brief  Computes a depth image from the given scan using the specified
    ///         policy
    /// \param img          A DepthImage to store the projection result.
    ///
    /// \param policy       Specifies how the entries are generated. FIRST stores the
    ///                     first projected distance. LAST the last distance. MINRANGE and
    ///                     MAXRANGE the minimal and maximal projected distances. AVERAGE
    ///                     averages over all encountered distances.
    ///
    void computeDepthImage(DepthImage& img, ProjectionPolicy policy = LAST);

    ///
    /// \brief  Computes a DepthListMatrix, i.e., an image matrix where each
    ///         entry holds a vector of all points that where projected to that
    ///         image position.
    ///
    /// \param mat          The generated DepthListMatrix
    ///
    void computeDepthListMatrix(DepthListMatrix& mat);


    ///
    /// \brief  Retruns the point buffer
    ///
    PointBufferPtr pointBuffer() { return m_points;}

private:


    /// Pointer to projection
    Projection*         m_projection;

    /// Pointer to the initial point cloud
    PointBufferPtr      m_points;

    /// Image width
    int                 m_width;

    /// Image height
    int                 m_height;

    /// Min horizontal opening angle
    int                 m_minHAngle;

    /// Max horizontal opening angle
    int                 m_maxHAngle;

    /// Min horizontal opening angle
    int                 m_minVAngle;

    /// Max horizontal opening angle
    int                 m_maxVAngle;

    /// Image optimization flag
    bool                m_optimize;

    /// Set this to true if you are using a left-handed coordinate system
    CoordinateSystem    m_coordinateSystem;

    /// Maximal z value that will be projected
    float               m_maxZ;

    /// Minimal z value that will be projected
    float               m_minZ;


};

} /* namespace lvr2 */

#endif /* SRC_LIBLVR2_RECONSTRUCTION_MODELTOIMAGE_HPP_ */
