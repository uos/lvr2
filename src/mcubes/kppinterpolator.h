#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

#include <iostream>
#include <fstream>
#include <omp.h>

#include <ANN/ANN.h>
#include <kdtree++/kdtree.hpp>

#include <vector>
using namespace std;

#include "../newmat/newmat.h"
#include "../mesh/colorVertex.h"
#include "../mesh/normal.h"

struct IdPoint{

  IdPoint(){};
  IdPoint(ANNpoint _p, int i){p = _p; id = i;};
  IdPoint(const IdPoint &o){ p = o.p; id = o.id;};

  float operator[](const int i){return p[i];};
  
  ANNpoint p;
  int id;
};

inline float tac(IdPoint p, int k){ return p.p[k];};

typedef KDTree::KDTree<3, IdPoint, std::pointer_to_binary_function<IdPoint, int, float> > tree_type;

class Interpolator{

public:
  Interpolator(ANNpointArray points, int n, int k1, int k2, int epsilon);
  
  
  ~Interpolator(){};

  void write(string filename);
  float distance(ColorVertex v);
  
  //private:

  ANNkd_tree* point_tree;
  ANNpointArray points;

  Normal* normals;

  int number_of_points;

  tree_type* t;
  
};



#endif
