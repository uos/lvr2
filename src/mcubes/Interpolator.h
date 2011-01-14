#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

#include "../model3d/ColorVertex.h"

class Interpolator{

public:
  virtual float distance(ColorVertex v)= 0;
  virtual float** getNormals(size_t &n) {return 0;};
};

#endif
