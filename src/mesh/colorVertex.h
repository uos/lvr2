#ifndef __COLOR_VERTEX_H__
#define __COLOR_VERTEX_H__

#include "baseVertex.h"

typedef unsigned char uchar;

class ColorVertex : public BaseVertex{

public:
  ColorVertex();
  ColorVertex(float x, float y, float z,
		    float r, float g, float b);

  ColorVertex(float x, float y, float z,
		    uchar r, uchar g, uchar b);

  ColorVertex(BaseVertex v, float r, float g, float b);
  ColorVertex(BaseVertex v, uchar r, uchar g, uchar b);

  virtual inline void render();
  
  uchar r;
  uchar g;
  uchar b;
  
};

inline ostream& operator<<(ostream& os, const ColorVertex v){
  os << "Color Vertex: " << v.x << " " << v.y << " " << v.z << endl;
  return os;
}

inline void ColorVertex::render(){
  //glColor3f(r, g, b);
  glVertex3f(x, y, -z);
}

#endif
