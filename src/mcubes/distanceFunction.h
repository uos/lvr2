#ifndef __DISTANCE_FUNCTION_H__
#define __DISTANCE_FUNCTION_H__

#include <iostream>
#include <fstream>
#include <vector>

#include <ANN/ANN.h>

#include "../mesh/baseVertex.h"
#include "../mesh/colorVertex.h"
#include "../newmat/newmat.h"

#include "tangentPlane.h"

using namespace std;

typedef enum direction{X, Y, Z};

class DistanceFunction{

public:
  DistanceFunction(ANNpointArray points, int n, int k, bool use_tp = true);
  ~DistanceFunction();
  
  float distance(const BaseVertex v, Normal &n, int k, float epsilon, direction dir, bool& ok) const;
  void  distance(ColorVertex vertices[], float distances[], int k, float epsilon);
  
  int getSign(BaseVertex v, int n); 

private:
  
  void createTangentPlanes();

  ANNkd_tree* point_tree;
  ANNkd_tree* tp_tree;

  ANNpointArray tp_centers;
  ANNpointArray points;

  int number_of_points;

  bool create_tangent_planes;

  vector<TangentPlane> tangentPlanes;
  vector<Normal> normals;
  
};

#endif
