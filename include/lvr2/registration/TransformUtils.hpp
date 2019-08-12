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
void getPoseFromMatrix(BaseVector<T>& position, BaseVector<T>& angles, const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& mat);

/**
 * @brief Transforms a registration matrix according to the given
 *        transformation matrix, i.e., applies @ref transform to @ref registration
 * 
 * @param transform             A transformation matrix
 * @param registration          A matrix representing a registration (i.e. transformation) that
 * @return Eigen::Matrix4d      The transformed registration matrix
 */
template<typename T>
Eigen::Matrix<T, 4, 4, Eigen::RowMajor> transformRegistration(const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& transform, const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& registration);

/**
 * @brief   Transforms an slam6d transformation matrix into an Eigen 4x4 matrix.
 */
template<typename T>
Eigen::Matrix<T, 4, 4, Eigen::RowMajor> buildTransformation(T* alignxf);

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
void transformPointCloud(ModelPtr model, const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& transformation);

/**
 * @brief   Transforms the given source frame according to the given coordinate
 *          transform struct 
 * 
 * @param   frame           Source frame
 * @param   ct               Coordinate system transformation
 * @return                  The transformed frame
 */
template<typename T>
Eigen::Matrix<T, 4, 4, Eigen::RowMajor> transformFrame(const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& frame, const CoordinateTransform<T>& ct);

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
Eigen::Matrix<T, 4, 4, Eigen::RowMajor> inverseTransform(const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& transform);

/**
 * @brief   Converts a Pose to a Matrix.
 * 
 * @param position  The position of the Pose
 * @param rotation  The rotation in radians
 * @return          The Matrix representation of the Pose
 */
template<typename T>
Eigen::Matrix<T, 4, 4, Eigen::RowMajor> poseToMatrix(const Eigen::Matrix<T, 3, 1>& position, const Eigen::Matrix<T, 3, 1>& rotation);

/**
 * @brief   Extracts the Pose from a Matrix
 * 
 * @param pose      A Matrix representing a Pose
 * @param position  Output for the position of the Pose
 * @param rotation  Output for the rotation in radians
 */
template<typename T>
void matrixToPose(const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& mat, Eigen::Matrix<T, 3, 1>& position, Eigen::Matrix<T, 3, 1>& rotation);

/**
 * @brief Converts a transformation matrix that is used in riegl coordinate system into
 *        a transformation matrix that is used in slam6d coordinate system.
 *
 * @param    in    The transformation matrix in riegl coordinate system
 *
 * @return The transformation matrix in slam6d coordinate system
 */
template <typename T>
static Eigen::Matrix<T, 4, 4, Eigen::RowMajor> rieglToSLAM6DTransform(const Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& in)
{
        Eigen::Matrix<T, 4, 4> ret;

        ret(0) = in(5);
        ret(1) = -in(9);
        ret(2) = -in(1);
        ret(3) = -in(13);
        ret(4) = -in(6);
        ret(5) = in(10);
        ret(6) = in(2);
        ret(7) = in(14);
        ret(8) = -in(4);
        ret(9) = in(8);
        ret(10) = in(0);
        ret(11) = in(12);
        ret(12) = -100*in(7);
        ret(13) = 100*in(11);
        ret(14) = 100*in(3);
        ret(15) = in(15);

        return ret;
}

/**
 * @brief Converts a transformation matrix that is used in slam6d coordinate system into
 *        a transformation matrix that is used in riegl coordinate system.
 *
 * @param    in    The transformation matrix in slam6d coordinate system
 *
 * @return The transformation matrix in riegl coordinate system
 */
template <typename T>
static Eigen::Matrix<T, 4, 4, Eigen::RowMajor> slam6dToRieglTransform(const Eigen::Matrix<T, 4, 4, Eigen::RowMajor> &in)
{
        Eigen::Matrix<T, 4, 4, Eigen::RowMajor> ret;

        ret(0) = in(10);
        ret(1) = -in(2);
        ret(2) = in(6);
        ret(3) = in(14)/100.0;
        ret(4) = -in(8);
        ret(5) = in(0);
        ret(6) = -in(4);
        ret(7) = -in(12)/100.0;
        ret(8) = in(9);
        ret(9) = -in(1);
        ret(10) = in(5);
        ret(11) = in(13)/100.0;
        ret(12) = in(11);
        ret(13) = -in(3);
        ret(14) = in(7);
        ret(15) = in(15);

        return ret;
}

template <typename T>
static Eigen::Matrix<T, 3, 1> slam6DToRieglPoint(const Eigen::Matrix<T, 3, 1> &in)
{
        return {
                in.z   / (T) 100.0,
                - in.x / (T) 100.0,
                in.y   / (T) 100.0
        };
}

template<typename T>
void eigenToEuler(Eigen::Matrix<T, 4, 4, Eigen::RowMajor>& mat, T* pose)
{
        T* m = mat.data();
        if(pose != 0){
                float _trX, _trY;
                if(m[0] > 0.0) {
                        pose[4] = asin(m[8]);
                } else {
                        pose[4] = (float)M_PI - asin(m[8]);
                }
                // rPosTheta[1] =  asin( m[8]);      // Calculate Y-axis angle

                float  C    =  cos( pose[4] );
                if ( fabs( C ) > 0.005 )  {          // Gimball lock?
                        _trX      =  m[10] / C;          // No, so get X-axis angle
                        _trY      =  -m[9] / C;
                        pose[3]  = atan2( _trY, _trX );
                        _trX      =  m[0] / C;           // Get Z-axis angle
                        _trY      = -m[4] / C;
                        pose[5]  = atan2( _trY, _trX );
                } else {                             // Gimball lock has occurred
                        pose[3] = 0.0;                   // Set X-axis angle to zero
                        _trX      =  m[5];  //1          // And calculate Z-axis angle
                        _trY      =  m[1];  //2
                        pose[5]  = atan2( _trY, _trX );
                }

                // cout << pose[3] << " " << pose[4] << " " << pose[5] << endl;

                pose[0] = m[12];
                pose[1] = m[13];
                pose[2] = m[14];
        }
}

} // namespace lvr2

#include "TransformUtils.tcc"

#endif
