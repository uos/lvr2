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
 * Quatrnion.tcc
 *
 *  @date 29.08.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */
 
namespace lvr
{

#ifndef PI
#define PI 3.141592654f
#endif

#ifndef PH
#define PH 1.570796326f
#endif

#ifndef PIOVER180
#define PIOVER180 0.017453292f
#endif

#ifndef TOLERANCE
#define TOLERANCE 0.0000001f
#endif

template<typename T>
Quaternion<T>::Quaternion(){

  x = 1.0;
  y = 0.0;
  z = 0.0;
  w = 0.0;

}

template<typename T>
Quaternion<T>::~Quaternion(){

}

template<typename T>
Quaternion<T>::Quaternion(T pitch, T yaw, T roll){

  fromEuler(pitch, yaw, roll);

}

template<typename T>
Quaternion<T>::Quaternion(Vertex<T> vec, T angle){

  fromAxis(vec, angle);

}

template<typename T>
Quaternion<T>::Quaternion(T _x, T _y, T _z, T _angle){

  x = _x;
  y = _y;
  z = _z;
  w = _angle;

}

template<typename T>
Quaternion<T>::Quaternion(T* vec, T _w){

  x = vec[0];
  y = vec[1];
  z = vec[2];
  w = _w;
}

template<typename T>
void Quaternion<T>::normalize()
{
	// Don't normalize if we don't have to
	T mag2 = w * w + x * x + y * y + z * z;
	if (fabs(mag2 - 1.0f) > TOLERANCE) {
		T mag = sqrt(mag2);
		w /= mag;
		x /= mag;
		y /= mag;
		z /= mag;
	}
}

template<typename T>
Quaternion<T> Quaternion<T>::copy(){

  //return Quaternion<T>(w, x, y, z);
  return Quaternion<T>(0, 0, 0);

}

template<typename T>
void Quaternion<T>::fromAxis(T* vec, T angle){

    T sinAngle;
    angle *= 0.5f;
    Normal<float> vn(vec[0], vec[1], vec[2]);
    sinAngle = sin(angle);

    x = (vn.x * sinAngle);
    y = (vn.y * sinAngle);
    z = (vn.z * sinAngle);
    w = cos(angle);

}

template<typename T>
void Quaternion<T>::fromAxis(Vertex<T> axis, T angle){

  T sinAngle;
  angle *= 0.5f;
  Normal<float> vn(axis.x, axis.y, axis.z);


  sinAngle = sin(angle);

  x = (vn.x * sinAngle);
  y = (vn.y * sinAngle);
  z = (vn.z * sinAngle);
  w = cos(angle);

}

template<typename T>
Quaternion<T> Quaternion<T>::getConjugate(){

  return Quaternion<T>(-x, -y, -z, w);

}

template<typename T>
Quaternion<T> Quaternion<T>::operator* (const Quaternion<T> rq){
	return Quaternion<T>(w * rq.x + x * rq.w + y * rq.z - z * rq.y,
					  w * rq.y + y * rq.w + z * rq.x - x * rq.z,
					  w * rq.z + z * rq.w + x * rq.y - y * rq.x,
					  w * rq.w - x * rq.x - y * rq.y - z * rq.z);
}

template<typename T>
Vertex<T> Quaternion<T>::operator* (Vertex<T> vec){

  Normal<float> vn(vec);

  Quaternion<T> vecQuat, resQuat;
  vecQuat.x = vn.x;
  vecQuat.y = vn.y;
  vecQuat.z = vn.z;
  vecQuat.w = 0.0f;

  resQuat = vecQuat * getConjugate();
  resQuat = *this * resQuat;

  return (Vertex<T>(resQuat.x, resQuat.y, resQuat.z));

}

template<typename T>
void Quaternion<T>::fromEuler(T pitch, T yaw, T roll){


  T p = pitch * PIOVER180 / 2.0f;
  T y = yaw * PIOVER180 / 2.0f;
  T r = roll * PIOVER180 / 2.0f;

  T sinp = sin(p);
  T siny = sin(y);
  T sinr = sin(r);
  T cosp = cos(p);
  T cosy = cos(y);
  T cosr = cos(r);

  x = sinr * cosp * cosy - cosr * sinp * siny;
  y = cosr * sinp * cosy + sinr * cosp * siny;
  z = cosr * cosp * siny - sinr * sinp * cosy;
  w = cosr * cosp * cosy + sinr * sinp * siny;

  normalize();
}

template<typename T>
void Quaternion<T>::getAxisAngle(Vertex<T> *axis, T *angle){

	T scale = sqrt(x * x + y * y + z * z);
	axis->x = x / scale;
	axis->y = y / scale;
	axis->z = z / scale;
	*angle = acos(w) * 2.0f;
}

template<typename T>
Matrix4<T> Quaternion<T>::getMatrix(){

	T matrix[16];
	getMatrix(matrix);
	return Matrix4<T>(matrix);
}

template<typename T>
void Quaternion<T>::getMatrix(T* m){

  T x2 = x * x;
  T y2 = y * y;
  T z2 = z * z;
  T xy = x * y;
  T xz = x * z;
  T yz = y * z;
  T wx = w * x;
  T wy = w * y;
  T wz = w * z;


  // return Matrix4<T>( 1.0f - 2.0f * (y2 + z2), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
// 			2.0f * (xy + wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz - wx), 0.0f,
// 			2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (x2 + y2), 0.0f,
// 			0.0f, 0.0f, 0.0f, 1.0f)

  m[0] = 1.0f - 2.0f * (y2 + z2);
  m[1] = 2.0f * (xy - wz);
  m[2] = 2.0f * (xz + wy);
  m[3] = 0.0f;

  m[4] = 2.0f * (xy + wz);
  m[5] = 1.0f - 2.0f * (x2 + z2);
  m[6] = 2.0f * (yz - wx);
  m[7] = 0.0f;

  m[8] = 2.0f * (xz - wy);
  m[9] = 2.0f * (yz + wx);
  m[10] = 1.0f - 2.0f * (x2 + y2);
  m[11] = 0.0f;

  m[12] = 0.0f;
  m[13] = 0.0f;
  m[14] = 0.0f;
  m[15] = 1.0f;

}

template<typename T>
Vertex<T> Quaternion<T>::toEuler(){

//	double sqw = q1.w*q1.w;
//	double sqx = q1.x*q1.x;
//	double sqy = q1.y*q1.y;
//	double sqz = q1.z*q1.z;
//	double unit = sqx + sqy + sqz + sqw; // if normalised is one, otherwise is correction factor
//	double test = q1.x*q1.y + q1.z*q1.w;
//	if (test > 0.499*unit) { // singularity at north pole
//		heading = 2 * atan2(q1.x,q1.w);
//		attitude = Math.PI/2;
//		bank = 0;
//		return;
//	}
//	if (test < -0.499*unit) { // singularity at south pole
//		heading = -2 * atan2(q1.x,q1.w);
//		attitude = -Math.PI/2;
//		bank = 0;
//		return;
//	}
//	heading = atan2(2*q1.y*q1.w-2*q1.x*q1.z , sqx - sqy - sqz + sqw);
//	attitude = asin(2*test/unit);
//	bank = atan2(2*q1.x*q1.w-2*q1.y*q1.z , -sqx + sqy - sqz + sqw)

	T yaw, pitch, roll;

	T sqw = w * w;
	T sqx = x * x;
	T sqy = y * y;
	T sqz = z * z;

	T unit = sqx + sqy + sqz + sqw;
	T test = x * y + z * w;
	if(test > 0.49999 * unit){    //singularity at nort pole
		yaw = 0;
		pitch = 2.0f * atan2(x, w);
		roll = PH;
		return Vertex<T>(yaw, pitch, roll);
	}
	if(test < -0.49999 * unit){  //singularity at south pole
		yaw = 0;
		pitch = -2 * atan2(x, w);
		roll = PH;
		return Vertex<T>(yaw, pitch, roll);
	}
	yaw = atan2(2 * x * w - 2 * y * z, -sqx + sqy - sqz + sqw );
	pitch = atan2(2 * y * w - 2 * x * z, sqx - sqy - sqz + sqw);
	roll = asin(2 * test / unit);
	return Vertex<T>(yaw, pitch, roll);
}

template<typename T>
void Quaternion<T>::printMatrix(){

  T matrix[16];
  getMatrix(matrix);

  printf("Quaternion in matrix representation: \n");
  for(int i = 0; i < 12; i+= 4){
    printf("%2.3f %2.3f %2.3f %2.3f\n",
		 matrix[i], matrix[i+1], matrix[i+2], matrix[i+3]);
  }


}

} // namespace lvr


