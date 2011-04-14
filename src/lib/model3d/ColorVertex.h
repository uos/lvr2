#ifndef __COLOR_VERTEX_H__
#define __COLOR_VERTEX_H__

#include "BaseVertex.h"

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
  ColorVertex(float, float, float);

  virtual ~ColorVertex(){};

  virtual inline void render();

  uchar r;
  uchar g;
  uchar b;

};

inline ostream& operator<<(ostream& os, const ColorVertex v){
  int r, g, b;
  r = (int)v.r;
  g = (int)v.g;
  b = (int)v.b;

  os << "Color Vertex: " << v.x << " " << v.y << " " << v.z << " "
	<< r << " " << g << " " << b << endl;
  return os;
}

inline void ColorVertex::render(){
  //glColor3f(r / 255.0, g / 255.0, b / 255.0);
  glVertex3f(x, y, z);
}

#endif
