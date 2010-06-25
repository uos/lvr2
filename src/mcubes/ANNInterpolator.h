#ifndef __ANN_INTERPOLATOR_H__
#define __ANN_INTERPOLATOR_H__

#include <iostream>
#include <fstream>
#include <omp.h>
#include <ANN/ANN.h>

//#include <kdtree++/kdtree.hpp>

#include <vector>
using namespace std;

#include "../newmat/newmat.h"

#include <lib3d/ColorVertex.h>
#include <lib3d/Normal.h>

#include "Interpolator.h"

class ANNInterpolator: public Interpolator{

public:
  ANNInterpolator(ANNpointArray points, int n, float voxelsize, int k_max, float epsilon);


  ~ANNInterpolator(){};

  void write(string filename);
  float distance(ColorVertex v);

  //private:

  ANNkd_tree* point_tree;
  ANNpointArray points;
  float voxelsize;
  float vs_sq;
  float epsilon;

  Normal* normals;

  int number_of_points;
  int k_max;

};



#endif
