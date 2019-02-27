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
 * Quatrnion.tcc
 *
 *  @date 29.08.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */
 
namespace lvr2
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

template<typename BaseVecT>
Quaternion<BaseVecT>::Quaternion(){

  x = 1.0;
  y = 0.0;
  z = 0.0;
  w = 0.0;

}

template<typename BaseVecT>
Quaternion<BaseVecT>::~Quaternion(){

}

template<typename BaseVecT>
Quaternion<BaseVecT>::Quaternion(ValueType pitch, ValueType yaw, ValueType roll){

  fromEuler(pitch, yaw, roll);

}

template<typename BaseVecT>
Quaternion<BaseVecT>::Quaternion(BaseVecT vec, ValueType angle){

  fromAxis(vec, angle);

}

template<typename BaseVecT>
Quaternion<BaseVecT>::Quaternion(ValueType _x, ValueType _y, ValueType _z, ValueType _angle){

  x = _x;
  y = _y;
  z = _z;
  w = _angle;

}

template<typename BaseVecT>
Quaternion<BaseVecT>::Quaternion(ValueType *vec, ValueType _w){

  x = vec[0];
  y = vec[1];
  z = vec[2];
  w = _w;
}

template<typename BaseVecT>
void Quaternion<BaseVecT>::normalize()
{
	// Don't normalize if we don't have to
	ValueType mag2 = w * w + x * x + y * y + z * z;
	if (fabs(mag2 - 1.0f) > TOLERANCE) {
		ValueType mag = sqrt(mag2);
		w /= mag;
		x /= mag;
		y /= mag;
		z /= mag;
	}
}

template<typename BaseVecT>
Quaternion<BaseVecT> Quaternion<BaseVecT>::copy(){

  //return Quaternion<BaseVecT>(w, x, y, z);
  return Quaternion<BaseVecT>(0, 0, 0);

}

template<typename BaseVecT>
void Quaternion<BaseVecT>::fromAxis(ValueType *vec, ValueType angle){

    ValueType sinAngle;
    angle *= 0.5f;
    Normal<BaseVecT> vn(vec[0], vec[1], vec[2]);
    sinAngle = sin(angle);

    x = (vn.x * sinAngle);
    y = (vn.y * sinAngle);
    z = (vn.z * sinAngle);
    w = cos(angle);

}

template<typename BaseVecT>
void Quaternion<BaseVecT>::fromAxis(BaseVecT axis, ValueType angle){

  ValueType sinAngle;
  angle *= 0.5f;
  Normal<typename BaseVecT::CoordType> vn(axis.x, axis.y, axis.z);


  sinAngle = sin(angle);

  x = (vn.x * sinAngle);
  y = (vn.y * sinAngle);
  z = (vn.z * sinAngle);
  w = cos(angle);

}

template<typename BaseVecT>
Quaternion<BaseVecT> Quaternion<BaseVecT>::getConjugate(){

  return Quaternion<BaseVecT>(-x, -y, -z, w);

}

template<typename BaseVecT>
Quaternion<BaseVecT> Quaternion<BaseVecT>::operator* (const Quaternion<BaseVecT> rq){
	return Quaternion<BaseVecT>(w * rq.x + x * rq.w + y * rq.z - z * rq.y,
					  w * rq.y + y * rq.w + z * rq.x - x * rq.z,
					  w * rq.z + z * rq.w + x * rq.y - y * rq.x,
					  w * rq.w - x * rq.x - y * rq.y - z * rq.z);
}

template<typename BaseVecT>
BaseVecT Quaternion<BaseVecT>::operator* (BaseVecT vec){

  Normal<typename BaseVecT::CoordType> vn(vec);

  Quaternion<BaseVecT> vecQuat, resQuat;
  vecQuat.x = vn.x;
  vecQuat.y = vn.y;
  vecQuat.z = vn.z;
  vecQuat.w = 0.0f;

  resQuat = vecQuat * getConjugate();
  resQuat = *this * resQuat;

  return (BaseVecT(resQuat.x, resQuat.y, resQuat.z));

}

template<typename BaseVecT>
void Quaternion<BaseVecT>::fromEuler(ValueType pitch, ValueType yaw, ValueType roll){


  ValueType p = pitch * PIOVER180 / 2.0f;
  ValueType y = yaw * PIOVER180 / 2.0f;
  ValueType r = roll * PIOVER180 / 2.0f;

  ValueType sinp = sin(p);
  ValueType siny = sin(y);
  ValueType sinr = sin(r);
  ValueType cosp = cos(p);
  ValueType cosy = cos(y);
  ValueType cosr = cos(r);

  x = sinr * cosp * cosy - cosr * sinp * siny;
  y = cosr * sinp * cosy + sinr * cosp * siny;
  z = cosr * cosp * siny - sinr * sinp * cosy;
  w = cosr * cosp * cosy + sinr * sinp * siny;

  normalize();
}

template<typename BaseVecT>
void Quaternion<BaseVecT>::getAxisAngle(BaseVecT *axis, ValueType *angle){

	ValueType scale = sqrt(x * x + y * y + z * z);
	axis->x = x / scale;
	axis->y = y / scale;
	axis->z = z / scale;
	*angle = acos(w) * 2.0f;
}

template<typename BaseVecT>
Matrix4<BaseVecT> Quaternion<BaseVecT>::getMatrix(){

	ValueType matrix[16];
	getMatrix(matrix);
	return Matrix4<BaseVecT>(matrix);
}

template<typename BaseVecT>
void Quaternion<BaseVecT>::getMatrix(ValueType *m){

  ValueType x2 = x * x;
  ValueType y2 = y * y;
  ValueType z2 = z * z;
  ValueType xy = x * y;
  ValueType xz = x * z;
  ValueType yz = y * z;
  ValueType wx = w * x;
  ValueType wy = w * y;
  ValueType wz = w * z;


  // return Matrix4<BaseVecT>( 1.0f - 2.0f * (y2 + z2), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
// 			2.0f * (xy + wzBaseVecT), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz - wx), 0.0f,
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

template<typename BaseVecT>
BaseVecT Quaternion<BaseVecT>::toEuler(){

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

	ValueType yaw, pitch, roll;

	ValueType sqw = w * w;
	ValueType sqx = x * x;
	ValueType sqy = y * y;
	ValueType sqz = z * z;

	ValueType unit = sqx + sqy + sqz + sqw;
	ValueType test = x * y + z * w;
	if(test > 0.49999 * unit){    //singularity at nort pole
		yaw = 0;
		pitch = 2.0f * atan2(x, w);
		roll = PH;
		return BaseVecT(yaw, pitch, roll);
	}
	if(test < -0.49999 * unit){  //singularity at south pole
		yaw = 0;
		pitch = -2 * atan2(x, w);
		roll = PH;
		return BaseVecT(yaw, pitch, roll);
	}
	yaw = atan2(2 * x * w - 2 * y * z, -sqx + sqy - sqz + sqw );
	pitch = atan2(2 * y * w - 2 * x * z, sqx - sqy - sqz + sqw);
	roll = asin(2 * test / unit);
	return BaseVecT(yaw, pitch, roll);
}

template<typename BaseVecT>
void Quaternion<BaseVecT>::printMatrix(){

  ValueType matrix[16];
  getMatrix(matrix);

  printf("Quaternion in matrix representation: \n");
  for(int i = 0; i < 12; i+= 4){
    printf("%2.3f %2.3f %2.3f %2.3f\n",
		 matrix[i], matrix[i+1], matrix[i+2], matrix[i+3]);
  }


}

} // namespace lvr2
