#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

#include <lib3d/ColorVertex.h>

class Interpolator{

public:
  virtual float distance(ColorVertex v)= 0;

};

#endif
