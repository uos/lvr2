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
 * TransformUtils.tcc
 *
 *  @date August 08, 2019
 *  @author Thomas Wiemann
 */

namespace lvr2
{

template<typename T>
void getPoseFromMatrix(BaseVector<T>& position, BaseVector<T>& angles, const Transform<T>& mat)
{
    double m[16];

    m[0]  = mat(0, 0);
    m[1]  = mat(0, 1);
    m[2]  = mat(0, 2);
    m[3]  = mat(0, 3);

    m[4]  = mat(1, 0);
    m[5]  = mat(1, 1);
    m[6]  = mat(1, 2);
    m[7]  = mat(1, 3);

    m[8]  = mat(2, 0);
    m[9]  = mat(2, 1);
    m[10] = mat(2, 2);
    m[11] = mat(2, 3);

    m[12] = mat(3, 0);
    m[13] = mat(3, 1);
    m[14] = mat(3, 2);
    m[15] = mat(3, 3);

    float _trX, _trY;
    if(m[0] > 0.0) {
       angles.y = asin(m[8]);
    } else {
       angles.y = (float)M_PI - asin(m[8]);
    }
    // rPosTheta[1] =  asin( m[8]);      // Calculate Y-axis angle

    float  C    =  cos(angles.y );
    if ( fabs( C ) > 0.005 )  {          // Gimball lock?
        _trX      =  m[10] / C;          // No, so get X-axis angle
        _trY      =  -m[9] / C;
        angles.x  = atan2( _trY, _trX );
        _trX      =  m[0] / C;           // Get Z-axis angle
        _trY      = -m[4] / C;
        angles.z  = atan2( _trY, _trX );
    } else {                             // Gimball lock has occurred
        angles.x = 0.0;                   // Set X-axis angle to zero
        _trX      =  m[5];  //1          // And calculate Z-axis angle
        _trY      =  m[1];  //2
        angles.z  = atan2( _trY, _trX );
    }

    //cout << angles.x << " " <<angles.y << " " << angles.z << endl;

    position.x = m[12];
    position.y = m[13];
    position.z = m[14];

}

template<typename T>
Transform<T> transformRegistration(const Transform<T>& transform, const Transform<T>& registration)
{
    Rotation<T> rotation_trans;
    Rotation<T> rotation_registration;

    rotation_trans = transform.template block<3,3>(0, 0);
    rotation_registration = registration.template block<3,3>(0, 0);

    Rotation<T> rotation = rotation_trans * rotation_registration;

    Transform<T> result = Transform<T>::Identity();
    result.template block<3,3>(0, 0) = rotation;

    Vector3<T> tmp;
    tmp = registration.template block<3,1>(0,3);
    tmp = rotation_trans * tmp;

    result.template block<3, 1>(0, 3) = transform.template block<3, 1>(0, 3) + tmp;

    return result;
}

template<typename T>
Transform<T> buildTransformation(T* alignxf)
{
    Rotation<T> rotation;
    Vector4<T> translation;

    rotation  << alignxf[0],  alignxf[4],  alignxf[8],
                 alignxf[1],  alignxf[5],  alignxf[9],
                 alignxf[2],  alignxf[6],  alignxf[10];

    translation << alignxf[12], alignxf[13], alignxf[14], 1.0;

    Transform<T> transformation;
    transformation.setIdentity();
    transformation.template block<3,3>(0,0) = rotation;
    transformation.template rightCols<1>() = translation;

    return transformation;
}

template<typename T>
void transformAndReducePointCloud(ModelPtr& model, int modulo, const CoordinateTransform<T>& c)
{
    transformAndReducePointCloud(model, modulo, c.sx, c.sy, c.sz, c.x, c.y, c.z);
}

template<typename T>
void transformAndReducePointCloud(
    ModelPtr model, int modulo, 
    const T& sx, const T& sy, const T& sz,
    const unsigned char& xPos, const unsigned char& yPos, const unsigned char& zPos)
{
    size_t n_ip, n_colors;
    size_t cntr = 0;
    size_t w_colors;

    n_ip = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();
    ucharArr colors = model->m_pointCloud->getUCharArray("colors", n_colors, w_colors);

    // Plus one because it might differ because of the 0-index
    // better waste memory for one float than having not enough space.
    // TO-DO think about exact calculation.
    size_t targetSize = (3 * ((n_ip)/modulo)) + modulo;
    size_t targetSizeColors = (w_colors * ((n_ip)/modulo)) + modulo;
    floatArr points(new float[targetSize ]);
    ucharArr newColorsArr;

    if(n_colors)
    {
        newColorsArr = ucharArr(new unsigned char[targetSizeColors]);
    }

    for(int i = 0; i < n_ip; i++)
    {
        if(i % modulo == 0)
        {
            if(sx != 1)
            {
                arr[i * 3]         *= sx;
            }

            if(sy != 1)
            {
                arr[i * 3 + 1]     *= sy;
            }

            if(sz != 1)
            {
                arr[i * 3 + 2]     *= sz;
            }

            if((cntr * 3) < targetSize)
            {
                points[cntr * 3]     = arr[i * 3 + xPos];
                points[cntr * 3 + 1] = arr[i * 3 + yPos];
                points[cntr * 3 + 2] = arr[i * 3 + zPos];
            }
            else
            {
                std::cout << "The following is for debugging purpose" << std::endl;
                std::cout << "Cntr: " << (cntr * 3) << " targetSize: " << targetSize << std::endl;
                std::cout << "nip : " << n_ip << " modulo " << modulo << std::endl;
                break;
            }

            if(n_colors)
            {
                for (unsigned j = 0; j < w_colors; j++)
                {
                    newColorsArr[cntr * w_colors + j] = colors[i * w_colors + j];
                }
            }

            cntr++;
        }
    }

    // Pass counter because it is the actual number of points used after reduction
    // it might be 1 less than the size
    model->m_pointCloud->setPointArray(points, cntr);

    if(n_colors)
    {
        model->m_pointCloud->setColorArray(newColorsArr, cntr, w_colors);
    }
}

template<typename T>
void transformPointCloud(ModelPtr model, const Transform<T>& transformation)
{
    std::cout << timestamp << "Transforming model." << std::endl;

    size_t numPoints = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();

    for(int i = 0; i < numPoints; i++)
    {
        float x = arr[3 * i];
        float y = arr[3 * i + 1];
        float z = arr[3 * i + 2];

        Vector4<T> v(x,y,z,1);
        Vector4<T> tv = transformation * v;

        arr[3 * i]     = tv[0];
        arr[3 * i + 1] = tv[1];
        arr[3 * i + 2] = tv[2];
    }
}

template<typename T>
Transform<T> transformFrame(const Transform<T>& frame, const CoordinateTransform<T>& ct)
{
    Eigen::Matrix<T, 3, 3> basisTrans;
    Eigen::Matrix<T, 3, 3> reflection;
    Vector3<T> tmp;
    std::vector<Vector3<T>> xyz;
    xyz.push_back(Vector3<T>(1,0,0));
    xyz.push_back(Vector3<T>(0,1,0));
    xyz.push_back(Vector3<T>(0,0,1));

    reflection.setIdentity();

    if(ct.sx < 0)
    {
        reflection.template block<3,1>(0,0) = (-1) * xyz[0];
    }

    if(ct.sy < 0)
    {
        reflection.template block<3,1>(0,1) = (-1) * xyz[1];
    }

    if(ct.sz < 0)
    {
        reflection.template block<3,1>(0,2) = (-1) * xyz[2];
    }

    // axis reflection
    frame.template block<3,3>(0,0) *= reflection;

    // We are always transforming from the canonical base => T = (B')^(-1)
    basisTrans.col(0) = xyz[ct.x];
    basisTrans.col(1) = xyz[ct.y];
    basisTrans.col(2) = xyz[ct.z];

    // Transform the rotation matrix
    frame.template block<3,3>(0,0) = basisTrans.inverse() * frame.template block<3,3>(0,0) * basisTrans;

    // Setting translation vector
    tmp = frame.template block<3,1>(0,3);
    tmp = basisTrans.inverse() * tmp;

    (frame.template rightCols<1>())(0) = tmp(0);
    (frame.template rightCols<1>())(1) = tmp(1);
    (frame.template rightCols<1>())(2) = tmp(2);
    (frame.template rightCols<1>())(3) = 1.0;

    return frame;
}

template<typename T>
Transform<T> inverseTransform(const Transform<T>& transform)
{
    Eigen::Matrix<T, 3, 3> rotation = transform.template block<3, 3>(0, 0);
    rotation.transposeInPlace();

    Transform<T> inv;
    inv.template block<3, 3>(0, 0) = rotation;

    (inv.template rightCols<1>())(0) = -transform.col(3)(0);
    (inv.template rightCols<1>())(1) = -transform.col(3)(1);
    (inv.template rightCols<1>())(2) = -transform.col(3)(2);
    (inv.template rightCols<1>())(3) = 1.0;

    return inv;
}

template<typename T>
Transform<T> poseToMatrix(const Vector3<T>& position, const Vector3<T>& rotation)
{
    Transform<T> mat = Transform<T>::Identity();
    mat.template block<3, 3>(0, 0) =  Eigen::AngleAxis<T>(rotation.x(), Vector3<T>::UnitX()).matrix()
                                    * Eigen::AngleAxis<T>(rotation.y(), Vector3<T>::UnitY())
                                    * Eigen::AngleAxis<T>(rotation.z(), Vector3<T>::UnitZ());

    mat.template block<3, 1>(0, 3) = position.template cast<T>();
    return mat;
}

template<typename T>
void matrixToPose(const Transform<T>& mat, Vector3<T>& position, Vector3<T>& rotation)
{
    // Calculate Y-axis angle
    if (mat(0, 0) > 0.0)
    {
        rotation.y() = asin(mat(2, 0));
    }
    else
    {
        rotation.y() = M_PI - asin(mat(2, 0));
    }

    double C = cos(rotation.y());
    if (fabs(C) < 0.005) // Gimbal lock?
    {
        // Gimbal lock has occurred
        rotation.x() = 0.0;
        rotation.z() = atan2(mat(0, 1), mat(1, 1));
    }
    else
    {
        rotation.x() = atan2(-mat(2, 1) / C, mat(2, 2) / C);
        rotation.z() = atan2(-mat(1, 0) / C, mat(0, 0) / C);
    }

    position = mat.template block<3, 1>(0, 3).template cast<T>();
}



} // namespace lvr2