/*
 * Matrix4.cpp
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */


#include "Matrix4.h"

Matrix4::Matrix4() {
	for(int i = 0; i < 16; i++) m[i] = 0.0;
	m[0] = m[5] = m[10] = m[15] = 1.0;
}

Matrix4::Matrix4(float* other){
	for(int i = 0; i < 16; i++) m[i] = other[i];
}


Matrix4::Matrix4(const Matrix4& other){
	for(int i = 0; i < 16; i++) m[i] = other[i];
}

Matrix4::Matrix4(string filename){
	loadFromFile(filename);
}

Matrix4::Matrix4(Vertex axis, float angle){

	if(fabs(angle) < 0.0001){

		bool invert_z = axis.z < 0;

		//Angle to yz-plane
		float pitch = atan2(axis.z, axis.x) - PH;
		if(pitch < 0.0f) pitch += 2.0f * PI;

		if(axis.x == 0.0f && axis.z == 0.0) pitch = 0.0f;

		//Transform axis into yz-plane
		axis.x =  axis.x * cos(pitch) + axis.z * sin(pitch);
		axis.z = -axis.x * sin(pitch) + axis.z * cos(pitch);

		//Angle to y-Axis
		float yaw = atan2(axis.y, axis.z);
		if(yaw < 0) yaw += 2 * PI;

		Matrix4 m1, m2, m3;

		if(invert_z) yaw = -yaw;

		cout << "YAW: " << yaw << " PITCH: " << pitch << endl;

		if(fabs(yaw)   > 0.0001){
			m2 = Matrix4(Vertex(1.0f, 0.0f, 0.0f), yaw);
			m3 = m3 * m2;
		}

		if(fabs(pitch) > 0.0001){
			m1 = Matrix4(Vertex(0.0f, 1.0f, 0.0f), pitch);
			m3 = m3 * m1;
		}

		for(int i = 0; i < 16; i++) m[i] = m3[i];

	} else {
		float c = cos(angle);
		float s = sin(angle);
		float t = 1.0f - c;
		float tmp1, tmp2;

		Normal a(axis); //Normalize axis

		m[ 0] = c + a.x * a.x * t;
		m[ 5] = c + a.y * a.y * t;
		m[10] = c + a.z * a.z * t;

		tmp1 = a.x * a.y * t;
		tmp2 = a.z * s;
		m[ 4] = tmp1 + tmp2;
		m[ 1] = tmp1 - tmp2;

		tmp1 = a.x * a.z * t;
		tmp2 = a.y * s;
		m[ 8] = tmp1 - tmp2;
		m[ 2] = tmp1 + tmp2;

		tmp1 = a.y * a.z * t;
		tmp2 = a.x * s;
		m[ 9] = tmp1 + tmp2;
		m[ 6] = tmp1 - tmp2;

		m[ 3] = m[ 7] = m[11] = 0.0;
		m[12] = m[13] = m[14] = 0.0;
		m[15] = 1.0;
	}

}

Matrix4::Matrix4(Vertex position, Vertex angles){
	float sx = sin(angles.x);
	float cx = cos(angles.x);
	float sy = sin(angles.y);
	float cy = cos(angles.y);
	float sz = sin(angles.z);
	float cz = cos(angles.z);

	m[0]  = cy*cz;
	m[1]  = sx*sy*cz + cx*sz;
	m[2]  = -cx*sy*cz + sx*sz;
	m[3]  = 0.0;
	m[4]  = -cy*sz;
	m[5]  = -sx*sy*sz + cx*cz;
	m[6]  = cx*sy*sz + sx*cz;
	m[7]  = 0.0;
	m[8]  = sy;
	m[9]  = -sx*cy;
	m[10] = cx*cy;

	m[11] = 0.0;

	m[12] = position.x;
	m[13] = position.y;
	m[14] = position.z;
	m[15] = 1;
}

void Matrix4::loadFromFile(string filename){

	ifstream in(filename.c_str());

	for(int i = 0; i < 16; i++){
		if(!in.good()){
			cout << "Warning: Matrix::loadFromFile: File not found or corrupted." << endl;
			return;
		}
		in >> m[i];
	}

}

void Matrix4::transpose(){
	float m_tmp[16];
	m_tmp[0]  = m[0];
	m_tmp[4]  = m[1];
	m_tmp[8]  = m[2];
	m_tmp[12] = m[3];
	m_tmp[1]  = m[4];
	m_tmp[5]  = m[5];
	m_tmp[9]  = m[6];
	m_tmp[13] = m[7];
	m_tmp[2]  = m[8];
	m_tmp[6]  = m[9];
	m_tmp[10] = m[10];
	m_tmp[14] = m[11];
	m_tmp[3]  = m[12];
	m_tmp[7]  = m[13];
	m_tmp[11] = m[14];
	m_tmp[15] = m[15];
	for(int i = 0; i < 16; i++) m[i] = m_tmp[i];
}

Matrix4::~Matrix4() {

}


void Matrix4::toPostionAngle(float* pose){

	if(pose != 0){
		float _trX, _trY;
		if(m[0] > 0.0) {
			pose[4] = asin(m[8]);
		} else {
			pose[4] = PI - asin(m[8]);
		}
		// rPosTheta[1] =  asin( m[8]);           // Calculate Y-axis angle

		float  C    =  cos( pose[4] );
		if ( fabs( C ) > 0.005 )  {                 // Gimball lock?
			_trX      =  m[10] / C;             // No, so get X-axis angle
			_trY      =  -m[9] / C;
			pose[3]  = atan2( _trY, _trX );
			_trX      =  m[0] / C;              // Get Z-axis angle
			_trY      = -m[4] / C;
			pose[5]  = atan2( _trY, _trX );
		} else {                                    // Gimball lock has occurred
			pose[3] = 0.0;                       // Set X-axis angle to zero
			_trX      =  m[5];  //1                // And calculate Z-axis angle
			_trY      =  m[1];  //2
			pose[5]  = atan2( _trY, _trX );
		}

		pose[0] = m[12];
		pose[1] = m[13];
		pose[2] = m[14];
	}
}


float Matrix4::operator[](const int i) const{
	if(i < 16)
		return m[i];
	else{
		cout << "Matrix4: Warning: Index out of range:" << i << endl;
		return 0.0;
	}
}



float Matrix4::at(int i) const{
	return m[i];
}
