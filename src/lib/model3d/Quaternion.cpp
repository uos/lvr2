#include "Quaternion.h"


Quaternion::Quaternion(){

  x = 1.0;
  y = 0.0;
  z = 0.0;
  w = 0.0;

}

Quaternion::~Quaternion(){

}

Quaternion::Quaternion(float pitch, float yaw, float roll){

  fromEuler(pitch, yaw, roll);

}

Quaternion::Quaternion(Vertex vec, float angle){

  fromAxis(vec, angle);

}

Quaternion::Quaternion(float _x, float _y, float _z, float _angle){

  x = _x;
  y = _y;
  z = _z;
  w = _angle;

}

Quaternion::Quaternion(float* vec, float _w){

  x = vec[0];
  y = vec[1];
  z = vec[2];
  w = _w;
}


void Quaternion::normalize()
{
	// Don't normalize if we don't have to
	float mag2 = w * w + x * x + y * y + z * z;
	if (fabs(mag2 - 1.0f) > TOLERANCE) {
		float mag = sqrt(mag2);
		w /= mag;
		x /= mag;
		y /= mag;
		z /= mag;
	}
}

Quaternion Quaternion::copy(){

  //return Quaternion(w, x, y, z);
  return Quaternion(0, 0, 0);

}

void Quaternion::fromAxis(float* vec, float angle){

    float sinAngle;
    angle *= 0.5f;
    Normal vn(vec[0], vec[1], vec[2]);
    sinAngle = sin(angle);

    x = (vn.x * sinAngle);
    y = (vn.y * sinAngle);
    z = (vn.z * sinAngle);
    w = cos(angle);

}

void Quaternion::fromAxis(Vertex axis, float angle){

  float sinAngle;
  angle *= 0.5f;
  Normal vn(axis.x, axis.y, axis.z);


  sinAngle = sin(angle);

  x = (vn.x * sinAngle);
  y = (vn.y * sinAngle);
  z = (vn.z * sinAngle);
  w = cos(angle);

}

Quaternion Quaternion::getConjugate(){

  return Quaternion(-x, -y, -z, w);

}


Quaternion Quaternion::operator* (const Quaternion rq){
	return Quaternion(w * rq.x + x * rq.w + y * rq.z - z * rq.y,
					  w * rq.y + y * rq.w + z * rq.x - x * rq.z,
					  w * rq.z + z * rq.w + x * rq.y - y * rq.x,
					  w * rq.w - x * rq.x - y * rq.y - z * rq.z);
}

Vertex Quaternion::operator* (Vertex vec){

  Normal vn(vec);

  Quaternion vecQuat, resQuat;
  vecQuat.x = vn.x;
  vecQuat.y = vn.y;
  vecQuat.z = vn.z;
  vecQuat.w = 0.0f;

  resQuat = vecQuat * getConjugate();
  resQuat = *this * resQuat;

  return (Vertex(resQuat.x, resQuat.y, resQuat.z));

}

void Quaternion::fromEuler(float pitch, float yaw, float roll){


  float p = pitch * PIOVER180 / 2.0f;
  float y = yaw * PIOVER180 / 2.0f;
  float r = roll * PIOVER180 / 2.0f;

  float sinp = sin(p);
  float siny = sin(y);
  float sinr = sin(r);
  float cosp = cos(p);
  float cosy = cos(y);
  float cosr = cos(r);

  x = sinr * cosp * cosy - cosr * sinp * siny;
  y = cosr * sinp * cosy + sinr * cosp * siny;
  z = cosr * cosp * siny - sinr * sinp * cosy;
  w = cosr * cosp * cosy + sinr * sinp * siny;

  normalize();
}

void Quaternion::getAxisAngle(Vertex *axis, float *angle){

	float scale = sqrt(x * x + y * y + z * z);
	axis->x = x / scale;
	axis->y = y / scale;
	axis->z = z / scale;
	*angle = acos(w) * 2.0f;
}

Matrix4 Quaternion::getMatrix(){

	float matrix[16];
	getMatrix(matrix);
	return Matrix4(matrix);
}

void Quaternion::getMatrix(float* m){

  float x2 = x * x;
  float y2 = y * y;
  float z2 = z * z;
  float xy = x * y;
  float xz = x * z;
  float yz = y * z;
  float wx = w * x;
  float wy = w * y;
  float wz = w * z;


  // return Matrix4( 1.0f - 2.0f * (y2 + z2), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
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

Vertex Quaternion::toEuler(){

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

	float yaw, pitch, roll;

	float sqw = w * w;
	float sqx = x * x;
	float sqy = y * y;
	float sqz = z * z;

	float unit = sqx + sqy + sqz + sqw;
	float test = x * y + z * w;
	if(test > 0.49999 * unit){    //singularity at nort pole
		yaw = 0;
		pitch = 2.0f * atan2(x, w);
		roll = PH;
		return Vertex(yaw, pitch, roll);
	}
	if(test < -0.49999 * unit){  //singularity at south pole
		yaw = 0;
		pitch = -2 * atan2(x, w);
		roll = PH;
		return Vertex(yaw, pitch, roll);
	}
	yaw = atan2(2 * x * w - 2 * y * z, -sqx + sqy - sqz + sqw );
	pitch = atan2(2 * y * w - 2 * x * z, sqx - sqy - sqz + sqw);
	roll = asin(2 * test / unit);
	return Vertex(yaw, pitch, roll);
}

void Quaternion::printMatrix(){

  float matrix[16];
  getMatrix(matrix);

  printf("Quaternion in matrix representation: \n");
  for(int i = 0; i < 12; i+= 4){
    printf("%2.3f %2.3f %2.3f %2.3f\n",
		 matrix[i], matrix[i+1], matrix[i+2], matrix[i+3]);
  }


}


