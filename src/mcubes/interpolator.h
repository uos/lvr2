#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

#include "../mesh/colorVertex.h"

class Interpolator{

public:
  virtual float distance(ColorVertex v)= 0;
  
};

#endif
