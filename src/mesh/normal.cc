#include "normal.h"

Normal::Normal() : BaseVertex() {};

Normal::Normal(const Normal& n) : BaseVertex(n.x, n.y, n.z){
  normalize();
}

Normal::Normal(const BaseVertex &v) : BaseVertex(v){
  normalize();
}

Normal::Normal(float x, float y, float z): BaseVertex(x, y, z){
  normalize();  
}


void Normal::normalize(){

  //Don't normalize if we don't have to
  float l_square = x * x + y * y + z * z;
  if( fabs(1 - l_square) > 0.001){
  
    float length = sqrt(l_square);
    if(length != 0){
	 x /= length;
	 y /= length;
	 z /= length;
    }
  }
  
}

Normal Normal::operator+(const Normal n) const{

  return Normal(x + n.x, y + n.y, z + n.z); 
  
}

void Normal::operator+=(const Normal n){
  
  x = x + n.x;
  y = y + n.y;
  z = z + n.z;
  
  normalize();
}

Normal Normal::operator-(const Normal n) const{

  return Normal(x + n.x, y + n.y, z + n.z); 
  
}

void Normal::operator-=(const Normal n){
  x = x + n.x;
  y = y + n.y;
  z = z + n.z;
  normalize();
}
