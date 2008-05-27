#include "colorVertex.h"

ColorVertex::ColorVertex() : BaseVertex(){
  r = g = b = 0;
}

ColorVertex::ColorVertex(float x, float y, float z,
					float _r, float _g, float _b) : BaseVertex(x, y, z){
  r = (uchar)(_r * 255);
  g = (uchar)(_g * 255);
  b = (uchar)(_b * 255);
}

ColorVertex::ColorVertex(float x, float y, float z,
					uchar _r, uchar _g, uchar _b) : BaseVertex(x, y, z){
  r = _r;
  g = _g;
  b = _b;
}

ColorVertex::ColorVertex(BaseVertex v,
					float _r, float _g, float _b) : BaseVertex(v){
  r = (uchar)(_r * 255);
  g = (uchar)(_g * 255);
  b = (uchar)(_b * 255);
}

ColorVertex::ColorVertex(BaseVertex v,
					uchar _r, uchar _g, uchar _b) : BaseVertex(v){
  r = _r;
  g = _g;
  b = _b;
}
