#ifndef __DISTANCE_FUNCTION_H__
#define __DISTANCE_FUNCTION_H__

#include <iostream>
#include <vector>

#include <ANN/ANN.h>

#include "../mesh/baseVertex.h"

#include "tangentPlane.h"

using namespace std;


class DistanceFunction{

public:
  DistanceFunction(ANNpointArray points, int n, int k, bool use_tp = true);
  ~DistanceFunction();
  
  float distance(const BaseVertex v) const;

private:

  void createTangentPlanes();

  ANNkd_tree* point_tree;
  ANNkd_tree* tp_tree;

  ANNpointArray tp_centers;
  ANNpointArray points;

  int number_of_points;

  bool create_tangent_planes;

  vector<TangentPlane> tangentPlanes;
};

#endif
