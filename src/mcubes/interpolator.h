#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

#include <iostream>
#include <fstream>
#include <omp.h>

#include <ANN/ANN.h>

#include <vector>
using namespace std;

#include "../newmat/newmat.h"
#include "../mesh/colorVertex.h"
#include "../mesh/normal.h"

class Interpolator{

public:
  Interpolator(ANNpointArray points, int n, int k1, int k2, int epsilon);
  ~Interpolator(){};

  void write(string filename);
  float distance(ColorVertex v);
  
private:

  ANNkd_tree* point_tree;
  ANNpointArray points;

  Normal* normals;

  int number_of_points;
  
};

#endif
