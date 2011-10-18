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



#include "geometry/Vertex.hpp"
#include "geometry/Normal.hpp"
#include "geometry/Matrix4.hpp"

#include <iostream>

using namespace std;

namespace lssr
{

template<typename T>
class Quaternion{

public:
  Quaternion();
  Quaternion(const Quaternion<T> &o){ x = o.x; y = o.y; z = o.z; w = o.w;};
  Quaternion(Vertex<T> vec, T angle);
  Quaternion(T pitch, T yaw, T roll);
  Quaternion(T x, T y, T z, T w);
  Quaternion(T* vec, T w);

  ~Quaternion();

  void normalize();
  void fromAxis(T* vec, T angle);
  void fromAxis(Vertex<T> axis, T angle);
  void fromEuler(T pitch, T yaw, T roll);

  void getAxisAngle(Vertex<T> *axis, T *angle);
  void getMatrix(T* m);

  void printMatrix();
  void printDebugInfo();

  Vertex<T> toEuler();

  T X() const {return x;};
  T Y() const {return y;};
  T Z() const {return z;};
  T W() const {return w;};

  Quaternion<T> getConjugate();
  Quaternion<T> copy();

  Quaternion<T> operator* (Quaternion rq);

  Vertex<T> operator* (Vertex<T> vec);
  Vertex<T> operator* (Vertex<T>* vec);

  Matrix4<T> getMatrix();

private:
  T w, x, y, z;

};

}

template<typename T>
inline ostream& operator<<(ostream& os, const lssr::Quaternion<T> q){

	return os << "Quaternion: " << q.W() << " " << q.X() << " " << q.Y() << " " << q.Z() << endl;

}

#include "Quaternion.tcc"

#endif

