/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * Quatrnion.hpp
 *
 *  @date 29.08.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

////////////////////////////////////////////////////////////
//
//  Author: Thomas Wiemann
//  Date:   29.08.2008
//
//  Quaternion representation of rotations.
//
//  Based on: http://gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation
//
/////////////////////////////////////////////////////////////


#ifndef __GLQUATERNION_H__
#define __GLQUATERNION_H__

#include "math.h"

#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/Normal.hpp>

#include <iostream>

using namespace std;

namespace lvr2
{

template<typename BaseVecT>
class Quaternion{

  using ValueType = typename BaseVecT::CoordType;

public:
  Quaternion();
  Quaternion(const Quaternion<BaseVecT> &o){ x = o.x; y = o.y; z = o.z; w = o.w;};
  Quaternion(Vector<BaseVecT> vec, ValueType angle);
  Quaternion(ValueType pitch, ValueType yaw, ValueType roll);
  Quaternion(ValueType x, ValueType y, ValueType z, ValueType w);
  Quaternion(ValueType *vec, ValueType w);

  ~Quaternion();

  void normalize();
  void fromAxis(ValueType *vec, ValueType angle);
  void fromAxis(Vector<BaseVecT> axis, ValueType angle);
  void fromEuler(ValueType pitch, ValueType yaw, ValueType roll);

  void getAxisAngle(Vector<BaseVecT> *axis, ValueType *angle);
  void getMatrix(ValueType *m);

  void printMatrix();
  void printDebugInfo();

  Vector<BaseVecT> toEuler();

  ValueType X() const {return x;};
  ValueType Y() const {return y;};
  ValueType Z() const {return z;};
  ValueType W() const {return w;};

  Quaternion<BaseVecT> getConjugate();
  Quaternion<BaseVecT> copy();

  Quaternion<BaseVecT> operator* (Quaternion<BaseVecT> rq);

  Vector<BaseVecT> operator* (Vector<BaseVecT> vec);
  Vector<BaseVecT> operator* (Vector<BaseVecT> *vec);

  Matrix4<BaseVecT> getMatrix();

private:
  ValueType w, x, y, z;

};

} // namespace lvr2

template<typename BaseVecT>
inline ostream& operator<<(ostream& os, const lvr2::Quaternion<BaseVecT> q){

	return os << "Quaternion: " << q.W() << " " << q.X() << " " << q.Y() << " " << q.Z() << endl;

}

#include "Quaternion.tcc"

#endif

