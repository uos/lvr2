// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see PMP_LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cassert>

#include <iostream>
#include <limits>
#include <initializer_list>

#include <Eigen/Dense>

#include "Exceptions.h"

namespace pmp {

//! \addtogroup core
//!@{

using Eigen::Matrix;

//! template specialization for Vector as Nx1 matrix
template <typename Scalar, int M>
using Vector = Matrix<Scalar, M, 1>;

//! template specialization for 4x4 matrices
template <typename Scalar>
using Mat4 = Matrix<Scalar, 4, 4>;

//! template specialization for 3x3 matrices
template <typename Scalar>
using Mat3 = Matrix<Scalar, 3, 3>;

//! template specialization for 2x2 matrices
template <typename Scalar>
using Mat2 = Matrix<Scalar, 2, 2>;

//! template specialization for a vector of two float values
typedef Vector<float, 2> vec2;
//! template specialization for a vector of two double values
typedef Vector<double, 2> dvec2;
//! template specialization for a vector of two bool values
typedef Vector<bool, 2> bvec2;
//! template specialization for a vector of two int values
typedef Vector<int, 2> ivec2;
//! template specialization for a vector of two unsigned int values
typedef Vector<unsigned int, 2> uvec2;

//! template specialization for a vector of three float values
typedef Vector<float, 3> vec3;
//! template specialization for a vector of three double values
typedef Vector<double, 3> dvec3;
//! template specialization for a vector of three bool values
typedef Vector<bool, 3> bvec3;
//! template specialization for a vector of three int values
typedef Vector<int, 3> ivec3;
//! template specialization for a vector of three unsigned int values
typedef Vector<unsigned int, 3> uvec3;

//! template specialization for a vector of four float values
typedef Vector<float, 4> vec4;
//! template specialization for a vector of four double values
typedef Vector<double, 4> dvec4;
//! template specialization for a vector of four bool values
typedef Vector<bool, 4> bvec4;
//! template specialization for a vector of four int values
typedef Vector<int, 4> ivec4;
//! template specialization for a vector of four unsigned int values
typedef Vector<unsigned int, 4> uvec4;

//! template specialization for a vector of four float values
typedef Vector<float, 8> vec8;
//! template specialization for a vector of four double values
typedef Vector<double, 8> dvec8;
//! template specialization for a vector of four bool values
typedef Vector<bool, 8> bvec8;
//! template specialization for a vector of four int values
typedef Vector<int, 8> ivec8;
//! template specialization for a vector of four unsigned int values
typedef Vector<unsigned int, 8> uvec8;

//! template specialization for a 2x2 matrix of float values
typedef Mat2<float> mat2;
//! template specialization for a 2x2 matrix of double values
typedef Mat2<double> dmat2;
//! template specialization for a 3x3 matrix of float values
typedef Mat3<float> mat3;
//! template specialization for a 3x3 matrix of double values
typedef Mat3<double> dmat3;
//! template specialization for a 4x4 matrix of float values
typedef Mat4<float> mat4;
//! template specialization for a 4x4 matrix of double values
typedef Mat4<double> dmat4;

//! OpenGL viewport matrix with parameters left, bottom, width, height
template <typename Scalar>
Mat4<Scalar> viewport_matrix(Scalar l, Scalar b, Scalar w, Scalar h)
{
    Mat4<Scalar> m(Scalar(0));

    m(0, 0) = 0.5 * w;
    m(0, 3) = 0.5 * w + l;
    m(1, 1) = 0.5 * h;
    m(1, 3) = 0.5 * h + b;
    m(2, 2) = 0.5;
    m(2, 3) = 0.5;
    m(3, 3) = 1.0f;

    return m;
}

//! inverse of OpenGL viewport matrix with parameters left, bottom, width, height
//! \sa viewport_matrix
template <typename Scalar>
Mat4<Scalar> inverse_viewport_matrix(Scalar l, Scalar b, Scalar w, Scalar h)
{
    Mat4<Scalar> m(Scalar(0));

    m(0, 0) = 2.0 / w;
    m(0, 3) = -1.0 - (l + l) / w;
    m(1, 1) = 2.0 / h;
    m(1, 3) = -1.0 - (b + b) / h;
    m(2, 2) = 2.0;
    m(2, 3) = -1.0;
    m(3, 3) = 1.0f;

    return m;
}

//! OpenGL frustum matrix with parameters left, right, bottom, top, near, far
template <typename Scalar>
Mat4<Scalar> frustum_matrix(Scalar l, Scalar r, Scalar b, Scalar t, Scalar n,
                            Scalar f)
{
    Mat4<Scalar> m(Scalar(0));

    m(0, 0) = (n + n) / (r - l);
    m(0, 2) = (r + l) / (r - l);
    m(1, 1) = (n + n) / (t - b);
    m(1, 2) = (t + b) / (t - b);
    m(2, 2) = -(f + n) / (f - n);
    m(2, 3) = -f * (n + n) / (f - n);
    m(3, 2) = -1.0f;

    return m;
}

//! inverse of OpenGL frustum matrix with parameters left, right, bottom, top, near, far
//! \sa frustum_matrix
template <typename Scalar>
Mat4<Scalar> inverse_frustum_matrix(Scalar l, Scalar r, Scalar b, Scalar t,
                                    Scalar n, Scalar f)
{
    Mat4<Scalar> m(Scalar(0));

    const Scalar nn = n + n;

    m(0, 0) = (r - l) / nn;
    m(0, 3) = (r + l) / nn;
    m(1, 1) = (t - b) / nn;
    m(1, 3) = (t + b) / nn;
    m(2, 3) = -1.0;
    m(3, 2) = (n - f) / (nn * f);
    m(3, 3) = (n + f) / (nn * f);

    return m;
}

//! OpenGL perspective matrix with parameters field of view in y-direction,
//! aspect ratio, and distance of near and far planes
template <typename Scalar>
Mat4<Scalar> perspective_matrix(Scalar fovy, Scalar aspect, Scalar zNear,
                                Scalar zFar)
{
    Scalar t = Scalar(zNear) * tan(fovy * M_PI / 360.0);
    Scalar b = -t;
    Scalar l = b * aspect;
    Scalar r = t * aspect;

    return frustum_matrix(l, r, b, t, Scalar(zNear), Scalar(zFar));
}

//! inverse of perspective matrix
//! \sa perspective_matrix
template <typename Scalar>
Mat4<Scalar> inverse_perspective_matrix(Scalar fovy, Scalar aspect,
                                        Scalar zNear, Scalar zFar)
{
    Scalar t = zNear * tan(fovy * M_PI / 360.0);
    Scalar b = -t;
    Scalar l = b * aspect;
    Scalar r = t * aspect;

    return inverse_frustum_matrix(l, r, b, t, zNear, zFar);
}

//! OpenGL orthogonal projection matrix with parameters left, right, bottom,
//! top, near, far
template <typename Scalar>
Mat4<Scalar> ortho_matrix(Scalar left, Scalar right, Scalar bottom, Scalar top,
                          Scalar zNear = -1, Scalar zFar = 1)
{
    Mat4<Scalar> m(0.0);

    m(0, 0) = Scalar(2) / (right - left);
    m(1, 1) = Scalar(2) / (top - bottom);
    m(2, 2) = -Scalar(2) / (zFar - zNear);
    m(0, 3) = -(right + left) / (right - left);
    m(1, 3) = -(top + bottom) / (top - bottom);
    m(2, 3) = -(zFar + zNear) / (zFar - zNear);
    m(3, 3) = Scalar(1);

    return m;
}

//! OpenGL look-at camera matrix with parameters eye position, scene center, up-direction
template <typename Scalar>
Mat4<Scalar> look_at_matrix(const Vector<Scalar, 3>& eye,
                            const Vector<Scalar, 3>& center,
                            const Vector<Scalar, 3>& up)
{
    Vector<Scalar, 3> z = (eye - center).normalized();
    Vector<Scalar, 3> x = up.cross(z).normalized();
    Vector<Scalar, 3> y = z.cross(x).normalized();

    // clang-format off
    Mat4<Scalar> m;
    m(0, 0) = x[0]; m(0, 1) = x[1]; m(0, 2) = x[2]; m(0, 3) = -x.dot(eye);
    m(1, 0) = y[0]; m(1, 1) = y[1]; m(1, 2) = y[2]; m(1, 3) = -y.dot(eye);
    m(2, 0) = z[0]; m(2, 1) = z[1]; m(2, 2) = z[2]; m(2, 3) = -z.dot(eye);
    m(3, 0) = 0.0;  m(3, 1) = 0.0;  m(3, 2) = 0.0;  m(3, 3) = 1.0;
    // clang-format on

    return m;
}

//! OpenGL matrix for translation by vector t
template <typename Scalar>
Mat4<Scalar> translation_matrix(const Vector<Scalar, 3>& t)
{
    Mat4<Scalar> m(Scalar(0));
    m(0, 0) = m(1, 1) = m(2, 2) = m(3, 3) = 1.0f;
    m(0, 3) = t[0];
    m(1, 3) = t[1];
    m(2, 3) = t[2];

    return m;
}

//! OpenGL matrix for scaling x/y/z by s
template <typename Scalar>
Mat4<Scalar> scaling_matrix(const Scalar s)
{
    Mat4<Scalar> m(Scalar(0));
    m(0, 0) = m(1, 1) = m(2, 2) = s;
    m(3, 3) = 1.0f;

    return m;
}

//! OpenGL matrix for scaling x/y/z by the components of s
template <typename Scalar>
Mat4<Scalar> scaling_matrix(const Vector<Scalar, 3>& s)
{
    Mat4<Scalar> m(Scalar(0));
    m(0, 0) = s[0];
    m(1, 1) = s[1];
    m(2, 2) = s[2];
    m(3, 3) = 1.0f;

    return m;
}

//! OpenGL matrix for rotation around x-axis by given angle (in degrees)
template <typename Scalar>
Mat4<Scalar> rotation_matrix_x(Scalar angle)
{
    Scalar ca = cos(angle * (M_PI / 180.0));
    Scalar sa = sin(angle * (M_PI / 180.0));

    Mat4<Scalar> m(0.0);
    m(0, 0) = 1.0;
    m(1, 1) = ca;
    m(1, 2) = -sa;
    m(2, 2) = ca;
    m(2, 1) = sa;
    m(3, 3) = 1.0;

    return m;
}

//! OpenGL matrix for rotation around y-axis by given angle (in degrees)
template <typename Scalar>
Mat4<Scalar> rotation_matrix_y(Scalar angle)
{
    Scalar ca = cos(angle * (M_PI / 180.0));
    Scalar sa = sin(angle * (M_PI / 180.0));

    Mat4<Scalar> m(0.0);
    m(0, 0) = ca;
    m(0, 2) = sa;
    m(1, 1) = 1.0;
    m(2, 0) = -sa;
    m(2, 2) = ca;
    m(3, 3) = 1.0;

    return m;
}

//! OpenGL matrix for rotation around z-axis by given angle (in degrees)
template <typename Scalar>
Mat4<Scalar> rotation_matrix_z(Scalar angle)
{
    Scalar ca = cos(angle * (M_PI / 180.0));
    Scalar sa = sin(angle * (M_PI / 180.0));

    Mat4<Scalar> m(0.0);
    m(0, 0) = ca;
    m(0, 1) = -sa;
    m(1, 0) = sa;
    m(1, 1) = ca;
    m(2, 2) = 1.0;
    m(3, 3) = 1.0;

    return m;
}

//! OpenGL matrix for rotation around given axis by given angle (in degrees)
template <typename Scalar>
Mat4<Scalar> rotation_matrix(const Vector<Scalar, 3>& axis, Scalar angle)
{
    Mat4<Scalar> m(Scalar(0));
    Scalar a = angle * (M_PI / 180.0f);
    Scalar c = cosf(a);
    Scalar s = sinf(a);
    Scalar one_m_c = Scalar(1) - c;
    Vector<Scalar, 3> ax = axis.normalized();

    m(0, 0) = ax[0] * ax[0] * one_m_c + c;
    m(0, 1) = ax[0] * ax[1] * one_m_c - ax[2] * s;
    m(0, 2) = ax[0] * ax[2] * one_m_c + ax[1] * s;

    m(1, 0) = ax[1] * ax[0] * one_m_c + ax[2] * s;
    m(1, 1) = ax[1] * ax[1] * one_m_c + c;
    m(1, 2) = ax[1] * ax[2] * one_m_c - ax[0] * s;

    m(2, 0) = ax[2] * ax[0] * one_m_c - ax[1] * s;
    m(2, 1) = ax[2] * ax[1] * one_m_c + ax[0] * s;
    m(2, 2) = ax[2] * ax[2] * one_m_c + c;

    m(3, 3) = 1.0f;

    return m;
}

//! OpenGL matrix for rotation specified by unit quaternion
template <typename Scalar>
Mat4<Scalar> rotation_matrix(const Vector<Scalar, 4>& quat)
{
    Mat4<Scalar> m(0.0f);
    Scalar s1(1);
    Scalar s2(2);

    m(0, 0) = s1 - s2 * quat[1] * quat[1] - s2 * quat[2] * quat[2];
    m(1, 0) = s2 * quat[0] * quat[1] + s2 * quat[3] * quat[2];
    m(2, 0) = s2 * quat[0] * quat[2] - s2 * quat[3] * quat[1];

    m(0, 1) = s2 * quat[0] * quat[1] - s2 * quat[3] * quat[2];
    m(1, 1) = s1 - s2 * quat[0] * quat[0] - s2 * quat[2] * quat[2];
    m(2, 1) = s2 * quat[1] * quat[2] + s2 * quat[3] * quat[0];

    m(0, 2) = s2 * quat[0] * quat[2] + s2 * quat[3] * quat[1];
    m(1, 2) = s2 * quat[1] * quat[2] - s2 * quat[3] * quat[0];
    m(2, 2) = s1 - s2 * quat[0] * quat[0] - s2 * quat[1] * quat[1];

    m(3, 3) = 1.0f;

    return m;
}

//! return upper 3x3 matrix from given 4x4 matrix, corresponding to the
//! linear part of an affine transformation
template <typename Scalar>
Mat3<Scalar> linear_part(const Mat4<Scalar>& m)
{
    Mat3<Scalar> result;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            result(i, j) = m(i, j);
    return result;
}

//! projective transformation of 3D vector v by a 4x4 matrix m:
//! add 1 as 4th component of v, multiply m*v, divide by 4th component
template <typename Scalar>
Vector<Scalar, 3> projective_transform(const Mat4<Scalar>& m,
                                       const Vector<Scalar, 3>& v)
{
    const Scalar x = m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2] + m(0, 3);
    const Scalar y = m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2] + m(1, 3);
    const Scalar z = m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2] + m(2, 3);
    const Scalar w = m(3, 0) * v[0] + m(3, 1) * v[1] + m(3, 2) * v[2] + m(3, 3);
    return Vector<Scalar, 3>(x / w, y / w, z / w);
}

//! affine transformation of 3D vector v by a 4x4 matrix m:
//! add 1 as 4th component of v, multiply m*v, do NOT divide by 4th component
template <typename Scalar>
Vector<Scalar, 3> affine_transform(const Mat4<Scalar>& m,
                                   const Vector<Scalar, 3>& v)
{
    const Scalar x = m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2] + m(0, 3);
    const Scalar y = m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2] + m(1, 3);
    const Scalar z = m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2] + m(2, 3);
    return Vector<Scalar, 3>(x, y, z);
}

//! linear transformation of 3D vector v by a 4x4 matrix m:
//! transform vector by upper-left 3x3 submatrix of m
template <typename Scalar>
Vector<Scalar, 3> linear_transform(const Mat4<Scalar>& m,
                                   const Vector<Scalar, 3>& v)
{
    const Scalar x = m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2];
    const Scalar y = m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2];
    const Scalar z = m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2];
    return Vector<Scalar, 3>(x, y, z);
}

//! return the inverse of a 4x4 matrix
template <typename Scalar>
Mat4<Scalar> inverse(const Mat4<Scalar>& m)
{
    Scalar Coef00 = m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2);
    Scalar Coef02 = m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1);
    Scalar Coef03 = m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1);

    Scalar Coef04 = m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2);
    Scalar Coef06 = m(1, 1) * m(3, 3) - m(1, 3) * m(3, 1);
    Scalar Coef07 = m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1);

    Scalar Coef08 = m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2);
    Scalar Coef10 = m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1);
    Scalar Coef11 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);

    Scalar Coef12 = m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2);
    Scalar Coef14 = m(0, 1) * m(3, 3) - m(0, 3) * m(3, 1);
    Scalar Coef15 = m(0, 1) * m(3, 2) - m(0, 2) * m(3, 1);

    Scalar Coef16 = m(0, 2) * m(2, 3) - m(0, 3) * m(2, 2);
    Scalar Coef18 = m(0, 1) * m(2, 3) - m(0, 3) * m(2, 1);
    Scalar Coef19 = m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1);

    Scalar Coef20 = m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2);
    Scalar Coef22 = m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1);
    Scalar Coef23 = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);

    Vector<Scalar, 4> const SignA(+1, -1, +1, -1);
    Vector<Scalar, 4> const SignB(-1, +1, -1, +1);

    Vector<Scalar, 4> Fac0(Coef00, Coef00, Coef02, Coef03);
    Vector<Scalar, 4> Fac1(Coef04, Coef04, Coef06, Coef07);
    Vector<Scalar, 4> Fac2(Coef08, Coef08, Coef10, Coef11);
    Vector<Scalar, 4> Fac3(Coef12, Coef12, Coef14, Coef15);
    Vector<Scalar, 4> Fac4(Coef16, Coef16, Coef18, Coef19);
    Vector<Scalar, 4> Fac5(Coef20, Coef20, Coef22, Coef23);

    Vector<Scalar, 4> Vec0(m(0, 1), m(0, 0), m(0, 0), m(0, 0));
    Vector<Scalar, 4> Vec1(m(1, 1), m(1, 0), m(1, 0), m(1, 0));
    Vector<Scalar, 4> Vec2(m(2, 1), m(2, 0), m(2, 0), m(2, 0));
    Vector<Scalar, 4> Vec3(m(3, 1), m(3, 0), m(3, 0), m(3, 0));

    // clang-format off
    Vector<Scalar, 4> Inv0 = cmult(SignA, (cmult(Vec1, Fac0) - cmult(Vec2, Fac1) + cmult(Vec3, Fac2)));
    Vector<Scalar, 4> Inv1 = cmult(SignB, (cmult(Vec0, Fac0) - cmult(Vec2, Fac3) + cmult(Vec3, Fac4)));
    Vector<Scalar, 4> Inv2 = cmult(SignA, (cmult(Vec0, Fac1) - cmult(Vec1, Fac3) + cmult(Vec3, Fac5)));
    Vector<Scalar, 4> Inv3 = cmult(SignB, (cmult(Vec0, Fac2) - cmult(Vec1, Fac4) + cmult(Vec2, Fac5)));
    // clang-format on

    Mat4<Scalar> Inverse(Inv0, Inv1, Inv2, Inv3);

    Vector<Scalar, 4> Row0(Inverse(0, 0), Inverse(1, 0), Inverse(2, 0),
                           Inverse(3, 0));
    Vector<Scalar, 4> Col0(m(0, 0), m(0, 1), m(0, 2), m(0, 3));

    Scalar Determinant = Col0.dot(Row0);

    Inverse /= Determinant;

    return Inverse;
}

//! return determinant of 3x3 matrix
template <typename Scalar>
Scalar determinant(const Mat3<Scalar>& m)
{
    return m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(2, 1) +
           m(1, 0) * m(0, 2) * m(2, 1) - m(1, 0) * m(0, 1) * m(2, 2) +
           m(2, 0) * m(0, 1) * m(1, 2) - m(2, 0) * m(0, 2) * m(1, 1);
}

//! return the inverse of a 3x3 matrix
template <typename Scalar>
Mat3<Scalar> inverse(const Mat3<Scalar>& m)
{
    const Scalar det = determinant(m);
    if (det < 1.0e-10 || std::isnan(det))
    {
        throw SolverException("3x3 matrix not invertible");
    }

    Mat3<Scalar> inv;
    inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / det;
    inv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) / det;
    inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / det;
    inv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) / det;
    inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / det;
    inv(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) / det;
    inv(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / det;
    inv(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) / det;
    inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / det;

    return inv;
}

//! compute eigenvector/eigenvalue decomposition of a 3x3 matrix
template <typename Scalar>
bool symmetric_eigendecomposition(const Mat3<Scalar>& m, Scalar& eval1,
                                  Scalar& eval2, Scalar& eval3,
                                  Vector<Scalar, 3>& evec1,
                                  Vector<Scalar, 3>& evec2,
                                  Vector<Scalar, 3>& evec3)
{
    unsigned int i, j;
    Scalar theta, t, c, s;
    Mat3<Scalar> V = Mat3<Scalar>::Identity();
    Mat3<Scalar> R;
    Mat3<Scalar> A = m;
    const Scalar eps = 1e-10; //0.000001;

    int iterations = 100;
    while (iterations--)
    {
        // find largest off-diagonal elem
        if (fabs(A(0, 1)) < fabs(A(0, 2)))
        {
            if (fabs(A(0, 2)) < fabs(A(1, 2)))
            {
                i = 1, j = 2;
            }
            else
            {
                i = 0, j = 2;
            }
        }
        else
        {
            if (fabs(A(0, 1)) < fabs(A(1, 2)))
            {
                i = 1, j = 2;
            }
            else
            {
                i = 0, j = 1;
            }
        }

        // converged?
        if (fabs(A(i, j)) < eps)
            break;

        // compute Jacobi-Rotation
        theta = 0.5 * (A(j, j) - A(i, i)) / A(i, j);
        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
        if (theta < 0.0)
            t = -t;

        c = 1.0 / sqrt(1.0 + t * t);
        s = t * c;

        R = Mat3<Scalar>::Identity();
        R(i, i) = R(j, j) = c;
        R(i, j) = s;
        R(j, i) = -s;

        A = R.transpose() * A * R;
        V = V * R;
    }

    if (iterations > 0)
    {
        // sort and return
        int sorted[3];
        Scalar d[3] = {A(0, 0), A(1, 1), A(2, 2)};

        if (d[0] > d[1])
        {
            if (d[1] > d[2])
            {
                sorted[0] = 0, sorted[1] = 1, sorted[2] = 2;
            }
            else
            {
                if (d[0] > d[2])
                {
                    sorted[0] = 0, sorted[1] = 2, sorted[2] = 1;
                }
                else
                {
                    sorted[0] = 2, sorted[1] = 0, sorted[2] = 1;
                }
            }
        }
        else
        {
            if (d[0] > d[2])
            {
                sorted[0] = 1, sorted[1] = 0, sorted[2] = 2;
            }
            else
            {
                if (d[1] > d[2])
                {
                    sorted[0] = 1, sorted[1] = 2, sorted[2] = 0;
                }
                else
                {
                    sorted[0] = 2, sorted[1] = 1, sorted[2] = 0;
                }
            }
        }

        eval1 = d[sorted[0]];
        eval2 = d[sorted[1]];
        eval3 = d[sorted[2]];

        evec1 = Vector<Scalar, 3>(V(0, sorted[0]), V(1, sorted[0]),
                                  V(2, sorted[0]));
        evec2 = Vector<Scalar, 3>(V(0, sorted[1]), V(1, sorted[1]),
                                  V(2, sorted[1]));
        evec3 = evec1.cross(evec2).normalized();

        return true;
    }

    return false;
}

//! read the space-separated components of a vector from a stream
template <typename Scalar, int N>
inline std::istream& operator>>(std::istream& is, Vector<Scalar, N>& vec)
{
    for (int i = 0; i < N; ++i)
        is >> vec[i];
    return is;
}

//! compute perpendicular vector (rotate vector counter-clockwise by 90 degrees)
template <typename Scalar>
inline Vector<Scalar, 2> perp(const Vector<Scalar, 2>& v)
{
    return Vector<Scalar, 2>(-v[1], v[0]);
}

//!@}

} // namespace pmp
