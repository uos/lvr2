#ifndef __TANGENT_PLANE_H__
#define __TANGENT_PLANE_H__

#include <ANN/ANN.h>
#include <iostream>
#include <vector>
using namespace std;

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

#include "../newmat/newmat.h"
#include "../mesh/baseVertex.h"
#include "../mesh/normal.h"

class TangentPlane{

public:
  TangentPlane(){};
  
  TangentPlane(const TangentPlane& o){
    center = o.center;
    normal = o.normal;
  }

  TangentPlane(BaseVertex v,
			ANNpointArray points,
			ANNkd_tree* tree,
			int n);

  BaseVertex getCenter() { return center;};
  Normal getNormal(){ return normal;};

  float getDistance(BaseVertex v);
  
private:

  BaseVertex center;
  Normal normal;
  
};

#endif
