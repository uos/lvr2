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

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/geometry/Normal.hpp"

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
  Quaternion(BaseVecT vec, ValueType angle);
  Quaternion(ValueType pitch, ValueType yaw, ValueType roll);
  Quaternion(ValueType x, ValueType y, ValueType z, ValueType w);
  Quaternion(ValueType *vec, ValueType w);

  ~Quaternion();

  void normalize();
  void fromAxis(ValueType *vec, ValueType angle);
  void fromAxis(BaseVecT axis, ValueType angle);
  void fromEuler(ValueType pitch, ValueType yaw, ValueType roll);

  void getAxisAngle(BaseVecT *axis, ValueType *angle);
  void getMatrix(ValueType *m);

  void printMatrix();
  void printDebugInfo();

  BaseVecT toEuler();

  ValueType X() const {return x;};
  ValueType Y() const {return y;};
  ValueType Z() const {return z;};
  ValueType W() const {return w;};

  Quaternion<BaseVecT> getConjugate();
  Quaternion<BaseVecT> copy();

  Quaternion<BaseVecT> operator* (Quaternion<BaseVecT> rq);

  BaseVecT operator* (BaseVecT vec);
  BaseVecT operator* (BaseVecT *vec);

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

