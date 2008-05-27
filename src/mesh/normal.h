#ifndef __NORMAL_H__
#define __NORMAL_H__

#include "baseVertex.h"

class Normal : public BaseVertex{

public:
  Normal();
  Normal(float x, float y, float z);
  Normal(const BaseVertex &other);
  Normal(const Normal &other);

  virtual inline void render();
  
  void normalize();

  virtual Normal operator+(const Normal n) const;
  virtual Normal operator-(const Normal n) const;

  virtual void operator+=(const Normal n);
  virtual void operator-=(const Normal n);

};

inline ostream& operator<<(ostream& os, const Normal n){
  os << "Normal: " << n.x << " " << n.y << " " << n.z << endl;
  return os;
}

inline void Normal::render(){
  glNormal3f(x, y, z);
}


#endif
