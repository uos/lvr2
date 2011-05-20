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

#include "Constants.h"
#include "BaseVertex.h"
#include "Normal.h"
#include "Matrix4.h"

#include <iostream>

using namespace std;

class Quaternion{

public:
  Quaternion();
  Quaternion(const Quaternion &o){ x = o.x; y = o.y; z = o.z; w = o.w;};
  Quaternion(Vertex vec, float angle);
  Quaternion(float pitch, float yaw, float roll);
  Quaternion(float x, float y, float z, float w);
  Quaternion(float* vec, float w);

  ~Quaternion();

  void normalize();
  void fromAxis(float* vec, float angle);
  void fromAxis(Vertex axis, float angle);
  void fromEuler(float pitch, float yaw, float roll);

  void getAxisAngle(Vertex *axis, float *angle);
  void getMatrix(float* m);

  void printMatrix();
  void printDebugInfo();

  Vertex toEuler();

  float X() const {return x;};
  float Y() const {return y;};
  float Z() const {return z;};
  float W() const {return w;};

  Quaternion getConjugate();
  Quaternion copy();

  Quaternion operator* (Quaternion rq);

  Vertex operator* (Vertex vec);
  Vertex operator* (Vertex* vec);

  Matrix4 getMatrix();

private:
  float w, x, y, z;

};

inline ostream& operator<<(ostream& os, const Quaternion q){

	return os << "Quaternion: " << q.W() << " " << q.X() << " " << q.Y() << " " << q.Z() << endl;

}

#endif

