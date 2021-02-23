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

#include "lvr2/types/MatrixTypes.hpp"
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
void getPoseFromMatrix(BaseVector<T>& position, BaseVector<T>& angles, const Transform<T>& mat);

/**
 * @brief Transforms a registration matrix according to the given
 *        transformation matrix, i.e., applies @ref transform to @ref registration
 * 
 * @param transform             A transformation matrix
 * @param registration          A matrix representing a registration (i.e. transformation) that
 * @return Transform<T>         The transformed registration matrix
 */
template<typename T>
Transform<T> transformRegistration(const Transform<T>& transform, const Transform<T>& registration);

/**
 * @brief   Transforms an slam6d transformation matrix into an Eigen 4x4 matrix.
 */
template<typename T>
Transform<T> buildTransformation(T* alignxf);

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
void transformPointCloud(ModelPtr model, const Transform<T>& transformation);

/**
 * @brief   Transforms the given source frame according to the given coordinate
 *          transform struct 
 * 
 * @param   frame           Source frame
 * @param   ct               Coordinate system transformation
 * @return                  The transformed frame
 */
template<typename T>
Transform<T> transformFrame(const Transform<T>& frame, const CoordinateTransform<T>& ct);

/**
 * @brief    Computes the inverse transformation from the given 
 *          transformation matrix, which means if transform encodes
 *          the transformation A->B, the return will transform from 
 *          B to A.
 * 
 * @param transform             A transformation matrix
 * @return Transform<T>         The inverse transformation
 */
template<typename T>
Transform<T> inverseTransform(const Transform<T>& transform);

/**
 * @brief   Converts a Pose to a Matrix.
 * 
 * @param position  The position of the Pose
 * @param rotation  The rotation in radians
 * @return          The Matrix representation of the Pose
 */
template<typename T>
Transform<T> poseToMatrix(const Vector3<T>& position, const Vector3<T>& rotation);

/**
 * @brief   Extracts the Pose from a Matrix
 * 
 * @param pose      A Matrix representing a Pose
 * @param position  Output for the position of the Pose
 * @param rotation  Output for the rotation in radians
 */
template<typename T>
void matrixToPose(const Transform<T>& mat, Vector3<T>& position, Vector3<T>& rotation);

/**
 * @brief Converts a transformation matrix that is used in riegl coordinate system into
 *        a transformation matrix that is used in slam6d coordinate system.
 *
 * @param    in    The transformation matrix in riegl coordinate system
 *
 * @return The transformation matrix in slam6d coordinate system
 */
template <typename T>
static Transform<T> rieglToSLAM6DTransform(const Transform<T>& in)
{
       Transform<T> ret;

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
static Transform<T> slam6dToRieglTransform(const Transform<T> &in)
{
    Transform<T> ret;

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
static Vector3<T> slam6DToRieglPoint(const Vector3<T> &in)
{
    return Vector3<T>(
        in.coeff(2)   / static_cast<T>(100.0),
        - in.coeff(0) / static_cast<T>(100.0),
        in.coeff(1)   / static_cast<T>(100.0)
    );
}

/////////////////////////////////
// Change Coodinate Systems
/////////////////////////////////

///////////////////////
// 1) CS -> LVR
//////////////

/**
 * @brief Slam6D to LVR coordinate change: Point
 * 
 *  Slam6D
 * 
 *     y  z    
 *     | /
 *     |/___ x
 *     
 *  - x: right
 *  - y: up
 *  - z: front
 *  - scale: cm
 * 
 *  LVR / ROS
 *        z  x    
 *        | /
 *   y ___|/ 
 * 
 *  - x: front
 *  - y: left
 *  - z: up
 *  - scale: m
 * 
 */
template <typename T>
static Vector3<T> slam6dToLvr(const Vector3<T>& in)
{
    return Vector3<T>(
        in.coeff(2) / static_cast<T>(100.0),
        - in.coeff(0) / static_cast<T>(100.0),
        in.coeff(1) / static_cast<T>(100.0)
    );
}

template<typename T>
static Rotation<T> slam6dToLvr(const Rotation<T>& in)
{
    Rotation<T> ret;
    
    // use coeff functions to access elements without range check
    // should be faster?

    // OLD CODE: why transpose?
    // ret.coeffRef(0, 0) =  in.coeff(2, 2);
    // ret.coeffRef(0, 1) = -in.coeff(0, 2);
    // ret.coeffRef(0, 2) =  in.coeff(1, 2);
    // ret.coeffRef(1, 0) = -in.coeff(2, 0);
    // ret.coeffRef(1, 1) =  in.coeff(0, 0);
    // ret.coeffRef(1, 2) = -in.coeff(1, 0);
    // ret.coeffRef(2, 0) =  in.coeff(2, 1);
    // ret.coeffRef(2, 1) = -in.coeff(0, 1);
    // ret.coeffRef(2, 2) =  in.coeff(1, 1);

    ret.coeffRef(0, 0) =  in.coeff(2, 2);
    ret.coeffRef(1, 0) = -in.coeff(0, 2);
    ret.coeffRef(2, 0) =  in.coeff(1, 2);
    ret.coeffRef(0, 1) = -in.coeff(2, 0);
    ret.coeffRef(1, 1) =  in.coeff(0, 0);
    ret.coeffRef(2, 1) = -in.coeff(1, 0);
    ret.coeffRef(0, 2) =  in.coeff(2, 1);
    ret.coeffRef(1, 2) = -in.coeff(0, 1);
    ret.coeffRef(2, 2) =  in.coeff(1, 1);
    
    return ret;
}

template <typename T>
static Transform<T> slam6dToLvr(const Transform<T> &in)
{
    Transform<T> ret;

    // Rotation
    const Rotation<T> R = in.template block<3,3>(0,0);
    ret.template block<3,3>(0,0) = slam6dToLvr(R);

    // Translation
    ret.coeffRef(0, 3) =  in.coeff(2, 3)/static_cast<T>(100.0);
    ret.coeffRef(1, 3) = -in.coeff(0, 3)/static_cast<T>(100.0);
    ret.coeffRef(2, 3) =  in.coeff(1, 3)/static_cast<T>(100.0);
    ret.coeffRef(3, 0) =  in.coeff(3, 2)/static_cast<T>(100.0);
    ret.coeffRef(3, 1) = -in.coeff(3, 0)/static_cast<T>(100.0);
    ret.coeffRef(3, 2) =  in.coeff(3, 1)/static_cast<T>(100.0);
    ret.coeffRef(3, 3) =  in.coeff(3, 3);

    return ret;
}

/**
 * @brief OpenCV to Lvr coordinate change: Point
 * 
 *  OpenCV
 * 
 *        z    
 *       /
 *      /___ x
 *     |
 *     |
 *     y
 *  
 *  - x: right
 *  - y: down
 *  - z: front
 *  - scale: m
 * 
 *  LVR / ROS
 *        z  x    
 *        | /
 *   y ___|/ 
 * 
 *  - x: front
 *  - y: left
 *  - z: up
 *  - scale: m
 * 
 */
template <typename T>
static Vector3<T> openCvToLvr(const Vector3<T>& in)
{
    return Vector3<T>(
        in.coeff(2),
        -in.coeff(0),
        -in.coeff(1)
    );
}

template <typename T>
static Rotation<T> openCvToLvr(const Rotation<T> &in)
{
    Rotation<T> ret;

    ret.coeffRef(0,0) =  in.coeff(2,2);
    ret.coeffRef(0,1) = -in.coeff(2,0);
    ret.coeffRef(0,2) = -in.coeff(2,1);
    ret.coeffRef(1,0) = -in.coeff(0,2);
    ret.coeffRef(1,1) =  in.coeff(0,0);
    ret.coeffRef(1,2) =  in.coeff(0,1);
    ret.coeffRef(2,0) = -in.coeff(1,2);
    ret.coeffRef(2,1) =  in.coeff(1,0);
    ret.coeffRef(2,2) =  in.coeff(1,1);

    return ret;
}

template <typename T>
static Transform<T> openCvToLvr(const Transform<T> &in)
{
    Transform<T> ret;

    const Rotation<T> R = in.template block<3,3>(0,0);
    ret.template block<3,3>(0,0) = openCvToLvr(R);

    ret.coeffRef(0,3) =  in.coeff(2,3);
    ret.coeffRef(1,3) = -in.coeff(0,3);
    ret.coeffRef(2,3) = -in.coeff(1,3);

    ret.coeffRef(3,0) =  in.coeff(3,2);
    ret.coeffRef(3,1) = -in.coeff(3,0);
    ret.coeffRef(3,2) = -in.coeff(3,1);

    ret.coeffRef(3,3) =  in.coeff(3,3);

    return ret;
}


////////////////
// 2) Lvr -> CS
////////////////



/**
 * @brief Lvr to Slam6D coordinate change: Point
 * 
 *  Slam6D
 * 
 *     y  z    
 *     | /
 *     |/___ x
 *     
 *  - x: right
 *  - y: up
 *  - z: front
 *  - scale: cm
 * 
 *  LVR / ROS
 *        z  x    
 *        | /
 *   y ___|/ 
 * 
 *  - x: front
 *  - y: left
 *  - z: up
 *  - scale: m
 */
template <typename T>
static Vector3<T> lvrToSlam6d(const Vector3<T>& in)
{
    return Vector3<T>(
        -in.coeff(1) * static_cast<T>(100.0),
        in.coeff(2) * static_cast<T>(100.0),
        in.coeff(0) * static_cast<T>(100.0)
    );
}

template<typename T>
static Rotation<T> lvrToSlam6d(const Rotation<T>& in)
{
    Rotation<T> ret;
    
    // use coeff functions to access elements without range check
    // should be faster?

    // OLD CODE: why transpose?
    // ret.coeffRef(0, 0) =  in.coeff(1, 1); // in.coeff(1, 1)
    // ret.coeffRef(0, 1) = -in.coeff(2, 1); // in.coeff(1, 2)
    // ret.coeffRef(0, 2) = -in.coeff(0, 1); // in.coeff(1, 0)
    // ret.coeffRef(1, 0) = -in.coeff(1, 2);
    // ret.coeffRef(1, 1) =  in.coeff(2, 2);
    // ret.coeffRef(1, 2) =  in.coeff(0, 2);
    // ret.coeffRef(2, 0) = -in.coeff(1, 0);
    // ret.coeffRef(2, 1) =  in.coeff(2, 0);
    // ret.coeffRef(2, 2) =  in.coeff(0, 0);

    ret.coeffRef(0, 0) =  in.coeff(1, 1); // in.coeff(1, 1)
    ret.coeffRef(1, 0) = -in.coeff(2, 1); // in.coeff(1, 2)
    ret.coeffRef(2, 0) = -in.coeff(0, 1); // in.coeff(1, 0)
    ret.coeffRef(0, 1) = -in.coeff(1, 2);
    ret.coeffRef(1, 1) =  in.coeff(2, 2);
    ret.coeffRef(2, 1) =  in.coeff(0, 2);
    ret.coeffRef(0, 2) = -in.coeff(1, 0);
    ret.coeffRef(1, 2) =  in.coeff(2, 0);
    ret.coeffRef(2, 2) =  in.coeff(0, 0);
    
    return ret;
}

template <typename T>
static Transform<T> lvrToSlam6d(const Transform<T> &in)
{
    Transform<T> ret;

    // Rotation
    const Rotation<T> R = in.template block<3,3>(0,0);
    ret.template block<3,3>(0,0) = lvrToSlam6d(R);

    // Translation
    ret.coeffRef(0, 3) = -static_cast<T>(100.0) * in.coeff(1, 3);
    ret.coeffRef(1, 3) =  static_cast<T>(100.0) * in.coeff(2, 3);
    ret.coeffRef(2, 3) =  static_cast<T>(100.0) * in.coeff(0, 3);
    ret.coeffRef(3, 0) = -static_cast<T>(100.0) * in.coeff(3, 1);
    ret.coeffRef(3, 1) =  static_cast<T>(100.0) * in.coeff(3, 2);
    ret.coeffRef(3, 2) =  static_cast<T>(100.0) * in.coeff(3, 0);

    ret.coeffRef(3, 3) = in.coeff(3, 3);

    return ret;
}



/**
 * @brief Lvr to OpenCV coordinate change: Point
 * 
 *  OpenCV
 * 
 *        z    
 *       /
 *      /___ x
 *     |
 *     |
 *     y
 *  
 * - x: right
 * - y: down
 * - z: front
 * - scale: m
 * 
 *  LVR / ROS
 *        z  x    
 *        | /
 *   y ___|/ 
 * 
 *  - x: front
 *  - y: left
 *  - z: up
 *  - scale: m
 * 
 */
template <typename T>
static Vector3<T> lvrToOpenCv(const Vector3<T>& in)
{
    return Vector3<T>(
        -in.coeff(1),
        -in.coeff(2),
        in.coeff(0)
    );
}

template <typename T>
static Rotation<T> lvrToOpenCv(const Rotation<T>& in)
{
    Rotation<T> ret;

    ret.coeffRef(0,0) =  in.coeff(1,1);
    ret.coeffRef(0,1) =  in.coeff(1,2);
    ret.coeffRef(0,2) = -in.coeff(1,0);
    ret.coeffRef(1,0) =  in.coeff(2,1);
    ret.coeffRef(1,1) =  in.coeff(2,2);
    ret.coeffRef(1,2) = -in.coeff(2,0);
    ret.coeffRef(2,0) = -in.coeff(0,1);
    ret.coeffRef(2,1) = -in.coeff(0,2);
    ret.coeffRef(2,2) =  in.coeff(0,0);

    return ret;
}

template <typename T>
static Transform<T> lvrToOpenCv(const Transform<T> &in)
{
    Transform<T> ret;

    const Rotation<T> R = in.template block<3,3>(0,0);
    ret.template block<3,3>(0,0) = lvrToOpenCv(R);

    ret.coeffRef(0,3) = -in.coeff(1,3);
    ret.coeffRef(1,3) = -in.coeff(2,3);
    ret.coeffRef(2,3) =  in.coeff(0,3);

    ret.coeffRef(3,0) = -in.coeff(3,1);
    ret.coeffRef(3,1) = -in.coeff(3,2);
    ret.coeffRef(3,2) =  in.coeff(3,0);

    ret.coeffRef(3,3) =  in.coeff(3,3);

    return ret;
}

template<typename T>
void extrinsicsToEuler(Extrinsics<T> mat, T* pose)
{
    T *m = mat.data();
    if (pose != 0)
    {
        float _trX, _trY;
        if (m[0] > 0.0)
        {
            pose[4] = asin(m[8]);
        }
        else
        {
            pose[4] = (float)M_PI - asin(m[8]);
        }
        // rPosTheta[1] =  asin( m[8]);      // Calculate Y-axis angle

        float C = cos(pose[4]);
        if (fabs(C) > 0.005)
        {                     // Gimball lock?
            _trX = m[10] / C; // No, so get X-axis angle
            _trY = -m[9] / C;
            pose[3] = atan2(_trY, _trX);
            _trX = m[0] / C; // Get Z-axis angle
            _trY = -m[4] / C;
            pose[5] = atan2(_trY, _trX);
        }
        else
        {                  // Gimball lock has occurred
            pose[3] = 0.0; // Set X-axis angle to zero
            _trX = m[5];   //1          // And calculate Z-axis angle
            _trY = m[1];   //2
            pose[5] = atan2(_trY, _trX);
        }

        // cout << pose[3] << " " << pose[4] << " " << pose[5] << endl;

        pose[0] = m[12];
        pose[1] = m[13];
        pose[2] = m[14];
    }
}

template<typename T>
void eigenToEuler(Transform<T>& mat, T* pose)
{
    T *m = mat.data();
    if (pose != 0)
    {
        float _trX, _trY;
        if (m[0] > 0.0)
        {
            pose[4] = asin(m[8]);
        }
        else
        {
            pose[4] = (float)M_PI - asin(m[8]);
        }
        // rPosTheta[1] =  asin( m[8]);      // Calculate Y-axis angle

        float C = cos(pose[4]);
        if (fabs(C) > 0.005)
        {                     // Gimball lock?
            _trX = m[10] / C; // No, so get X-axis angle
            _trY = -m[9] / C;
            pose[3] = atan2(_trY, _trX);
            _trX = m[0] / C; // Get Z-axis angle
            _trY = -m[4] / C;
            pose[5] = atan2(_trY, _trX);
        }
        else
        {                  // Gimball lock has occurred
            pose[3] = 0.0; // Set X-axis angle to zero
            _trX = m[5];   //1          // And calculate Z-axis angle
            _trY = m[1];   //2
            pose[5] = atan2(_trY, _trX);
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
