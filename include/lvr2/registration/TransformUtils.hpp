#ifndef __TRANSFORM_UTILS__
#define __TRANSFORM_UTILS__

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
 * TransformUtils.hpp
 *
 *  @date August 08, 2019
 *  @author Thomas Wiemann
 */

#include "lvr2/io/Model.hpp"
#include "lvr2/io/CoordinateTransform.hpp"

#include <Eigen/Dense>

namespace lvr2
{

/**
 * @brief   Computes a Euler representation from the given transformation matrix
 * 
 * @param  position     Will contain the position
 * @param  angles       Will contain the rotation angles in radians
 * @param  mat          The transformation matrix
 */
template<typename T>
void getPoseFromMatrix(BaseVector<T>& position, BaseVector<T>& angles, const Eigen::Matrix<T, 4, 4>& mat);

/**
 * @brief Transforms a registration matrix according to the given
 *        transformation matrix, i.e., applies @ref transform to @ref registration
 * 
 * @param transform             A transformation matrix
 * @param registration          A matrix representing a registration (i.e. transformation) that
 * @return Eigen::Matrix4d      The transformed registration matrix
 */
template<typename T>
Eigen::Matrix<T, 4, 4> transformRegistration(const Eigen::Matrix<T, 4, 4>& transform, const Eigen::Matrix<T, 4, 4>& registration);

/**
 * @brief   Transforms an slam6d transformation matrix into an Eigen 4x4 matrix.
 */
template<typename T>
Eigen::Matrix<T, 4, 4> buildTransformation(T* alignxf);

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
template<typename T>
void transformPointCloud(ModelPtr model, const Eigen::Matrix<T, 4, 4>& transformation);

/**
 * @brief   Transforms the given source frame according to the given coordinate
 *          transform struct 
 * 
 * @param   frame           Source frame
 * @param   ct               Coordinate system transformation
 * @return                  The transformed frame
 */
template<typename T>
Eigen::Matrix<T, 4, 4> transformFrame(const Eigen::Matrix<T, 4, 4>& frame, const CoordinateTransform<T>& ct);

/**
 * @brief    Computes the inverse transformation from the given 
 *          transformation matrix, which means if transform encodes
 *          the transformation A->B, the return will transform from 
 *          B to A.
 * 
 * @param transform             A transformation matrix
 * @return Eigen::Matrix4d      The inverse transformation
 */
template<typename T>
Eigen::Matrix<T, 4, 4> inverseTransform(const Eigen::Matrix<T, 4, 4>& transform);

/**
 * @brief   Converts a Pose to a Matrix.
 * 
 * @param position  The position of the Pose
 * @param rotation  The rotation in radians
 * @return          The Matrix representation of the Pose
 */
template<typename T>
Eigen::Matrix<T, 4, 4> poseToMatrix(const Eigen::Matrix<T, 3, 1>& position, const Eigen::Matrix<T, 3, 1>& rotation);

/**
 * @brief   Extracts the Pose from a Matrix
 * 
 * @param pose      A Matrix representing a Pose
 * @param position  Output for the position of the Pose
 * @param rotation  Output for the rotation in radians
 */
template<typename T>
void matrixToPose(const Eigen::Matrix<T, 4, 4>& mat, Eigen::Matrix<T, 3, 1>& position, Eigen::Matrix<T, 3, 1>& rotation);

} // namespace lvr2

#include "TransformUtils.tcc"

#endif
