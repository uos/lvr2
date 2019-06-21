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

#ifndef IOUTILS_HPP
#define IOUTILS_HPP

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/Model.hpp"
#include "lvr2/io/CoordinateTransform.hpp"

#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <vector>

namespace lvr2
{

/**
 * @brief Transforms a registration matrix according to the given
 *        transformation matrix, i.e., applies @ref transform to @ref registration
 * 
 * @param transform             A transformation matrix
 * @param registration          A matrix representing a registration (i.e. transformation) that
 * @return Eigen::Matrix4d      The transformed registration matrix
 */
Eigen::Matrix4d transformRegistration(const Eigen::Matrix4d& transform, const Eigen::Matrix4d& registration);

/**
 * @brief   Loads an Euler representation of from a pose file
 * 
 * @param position      Will contain the postion
 * @param angles        Will contain the rotation angles in degrees
 * @param file          The pose file
 */
void getPoseFromFile(BaseVector<float>& position, BaseVector<float>& angles, const boost::filesystem::path file);

/**
 * @brief   Computes a Euler representation from the given transformation matrix
 * 
 * @param  position     Will contain the position
 * @param  angles       Will contain the rotation angles in radians
 * @param  mat          The transformation matrix
 */
void getPoseFromMatrix(BaseVector<float>& position, BaseVector<float>& angles, const Eigen::Matrix4d& mat);

/**
 * @brief   Returns a Eigen 4x4 maxtrix representation of the transformation
 *          represented in the given frame file.
 */
Eigen::Matrix4d getTransformationFromFrames(boost::filesystem::path& frames);

/**
 * @brief   Returns a Eigen 4x4 maxtrix representation of the transformation
 *          represented in the given pose file.
 */
Eigen::Matrix4d getTransformationFromPose(boost::filesystem::path& pose);

/**
 * @brief   Returns a Eigen 4x4 maxtrix representation of the transformation
 *          represented in the given dat file.
 */
Eigen::Matrix4d getTransformationFromDat(boost::filesystem::path& frames);

/**
 * @brief   Transforms an slam6d transformation matrix into an Eigen 4x4 matrix.
 */
Eigen::Matrix4d buildTransformation(double* alignxf);

/***
 * @brief   Counts the number of points (i.e., lines) in the given file. We
 *          assume that it is an plain ASCII with one point per line.
 */
size_t countPointsInFile(boost::filesystem::path& inFile);

/**
 * @brief   Writes a Eigen transformation into a .frames file
 *
 * @param   transform   The transformation
 * @param   framesOut   The target file.
 */
void writeFrame(Eigen::Matrix4d transform, const boost::filesystem::path& framesOut);

/**
 * @brief               Writes pose information in Euler representation to the given file
 * 
 * @param position      Position
 * @param angles        Rotation angles in degrees
 */
void writePose(const BaseVector<float>& position, const BaseVector<float>& angles, const boost::filesystem::path& out);

/**
 * @brief   Writes the given model to the given file
 *
 * @param   model       A LVR model
 * @param   outfile     The target file.
 * @return  The number of points writen to the target file.
 */
size_t writeModel( ModelPtr model, const  boost::filesystem::path& outfile);



/**
 * @brief   Computes the reduction factor for a given target size (number of
 *          points) when reducing a point cloud using a modulo filter.
 *
 * @param   model       A model containing point cloud data
 * @param   targetSize  The desired number of points in the reduced model
 *
 * @return  The parameter n for the modulo filter (which means that you only
 *          have to write only every nth point to have approximately \ref
 *          targetSize points in the reduced point cloud.
 */
size_t getReductionFactor(ModelPtr model, size_t targetSize);

/**
 * @brief   Computes the reduction factor for a given target size (number of
 *          points) when reducing a point cloud loaded from an ASCII file
 *          using a modulo filter.
 *
 * @param   inFile      An ASCII file containing point cloud data
 * @param   targetSize  The desired number of points in the reduced model
 *
 * @return  The parameter n for the modulo filter (which means that you only
 *          have to write only every nth point to have approximately \ref
 *          targetSize points in the reduced point cloud.
 */
size_t getReductionFactor(boost::filesystem::path& inFile, size_t targetSize);

/**
 * @brief   Transforms (scale and switch coordinates) and reduces a model
 *          containing point cloud data using a modulo filter. Use this
 *          function the convert between different coordinate systems.
 *
 * @param   model       A model containing point cloud data
 * @param   modulo      The reduction factor for the modulo filter. Set to
 *                      1 to keep the original resolution.
 * @param   sx          Scaling factor in x direction
 * @param   sy          Scaling factor in y direction
 * @param   sz          Scaling factor in z direction
 * @param   xPos        Position of the x position in the input data, i.e,
 *                      "which array position has the x coordinate that is written
 *                      to the output data in the input data"
 * @param   yPos        Same as with xPos for y.
 * @param   zPos        Same as with xPos for z.
 */
template<typename T>
void transformAndReducePointCloud(ModelPtr model, int modulo, 
        const T& sx, const T& sy, const T& sz, 
        const unsigned char& xPos, 
        const unsigned char& yPos, 
        const unsigned char& zPos);

/**
 * @brief  Transforms (scale and switch coordinates) and reduces a model
 *         containing point cloud data using a modulo filter. Use this
 *         function the convert between different coordinate systems.          
 * 
 * @param model         A model containing point cloud data 
 * @param modulo        The reduction factor for the modulo filter. Set to
 *                      1 to keep the original resolution.
 * @param c             The coordinate transformation applied to the \ref model
 */
template<typename T>
void transformAndReducePointCloud(ModelPtr& model, int modulo, const CoordinateTransform<T>& c);

/**
 * @brief   Transforms a model containing a point cloud according to the given
 *          transformation (usually from a .frames file)
 * @param   A model containing point cloud data.
 * @param   A transformation.
 */
void transformPointCloud(ModelPtr model, Eigen::Matrix4d transformation);

/**
 * @brief   Transforms the given point buffer according to the transformation
 *          stored in \ref transformFile and appends the transformed points and
 *          normals to \ref pts and \ref nrm.
 *
 * @param   buffer          A PointBuffer
 * @param   transformFile   The input file name. The fuction will search for transformation information
 *                          (.pose or .frames)
 * @param   pts             The transformed points are added to this vector
 * @param   nrm             The transformed normals are added to this vector
 */
void transformPointCloudAndAppend(PointBufferPtr& buffer,
        boost::filesystem::path& transfromFile,
        std::vector<float>& pts,
        std::vector<float>& nrm);

/**
 * @brief   Writes the points and normals (float triples) stored in \ref p and \ref n
 *          to the given output file. Attention: The data is converted to a PointBuffer
 *          structure to be able to use the IO library, which results in a considerable
 *          memory overhead.
 *
 * @param   p               A vector containing point definitions.
 * @param   n               A vector containing normal information.
 * @param   outfile         The target file.
 */
void writePointsAndNormals(std::vector<float>& p, std::vector<float>& n, std::string outfile);

/**
 * @brief   Transforms the given source frame according to the given coordinate
 *          transform struct 
 * 
 * @param   frame           Source frame
 * @param   ct               Coordinate system transformation
 * @return                  The transformed frame
 */
template<typename T>
Eigen::Matrix4d transformFrame(Eigen::Matrix4d frame, const CoordinateTransform<T>& ct);


/**
 * @brief   Writes the points stored in the given model to the fiven output
 *          stream. This function is used to apend point cloud data to an
 *          already existing ASCII file..
 *
 * @param   model       A model containing point cloud data
 * @param   out         A output stream
 * @param   nocolor     If set to true, the color information in the model
 *                      is ignored.
 * @return  The number of points written to the output stream.
 */
size_t writePointsToStream(ModelPtr model, std::ofstream& out, bool nocolor = false);

/**
 * @brif    Computes the inverse transformation from the given 
 *          transformation matrix, which means if transform encodes
 *          the transformation A->B, the return will transform from 
 *          B to A.
 * 
 * @param transform             A transformation matrix
 * @return Eigen::Matrix4d      The inverse transformation
 */
Eigen::Matrix4d inverseTransform(const Eigen::Matrix4d& transform);

/**
 * @brief   Converts a Pose to a Matrix.
 * 
 * @param position  The position of the Pose
 * @param rotation  The rotation in radians
 * @return          The Matrix representation of the Pose
 */
Eigen::Matrix4f poseToMatrix(const Eigen::Vector3f& position, const Eigen::Vector3f& rotation);

/**
 * @brief   Extracts the Pose from a Matrix
 * 
 * @param pose      A Matrix representing a Pose
 * @param position  Output for the position of the Pose
 * @param rotation  Output for the rotation in radians
 */
void matrixToPose(const Eigen::Matrix4f& mat, Eigen::Vector3f& position, Eigen::Vector3f& rotation);

} // namespace lvr2

// #include "IOUtils.tcc"

#endif // IOUTILS_HPP
