#ifndef __HASH_GRID_H__
#define __HASH_GRID_H__

#include "../io/plyWriter.h"
#include "../mesh/staticMesh.h"
#include "tables.h"
#include "distanceFunction.h"
#include "interpolator.h"
#include "box.h"


#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <ext/hash_map>

#include <ANN/ANN.h>


using __gnu_cxx::hash_map;
using namespace std;

class HashGrid{

public:
  HashGrid(string filename, float voxelsize, float scale = 1.0);
  ~HashGrid();
  
  void writeMesh();
  void writeGrid();
  void writeBorders();
  
private:

  void createGrid();
  void createMesh();
  
  
  int readPoints(string filename, float scale);
  int getFieldsPerLine(string filename);
  
  inline int hashValue(int j, int j, int k);
  inline int calcIndex(float f);

  hash_map<int, Box*> cells;
  
  //The voxelsize
  float voxelsize;
  
  //The Mesh
  StaticMesh mesh;

  //Point array 
  ANNpointArray points;

  //Number of data points
  int number_of_points;

  //Bounding box representation
  float xmin;
  float ymin;
  float zmin;

  float xmax;
  float ymax;
  float zmax;
  
  //Maximum indices
  int max_index;
  int max_index_square;
  int max_index_x;
  int max_index_y;
  int max_index_z;

  //Distance function
  //DistanceFunction distance_function
  Interpolator* interpolator;
  
};

inline int HashGrid::hashValue(int i, int j, int k){
  return i * max_index_square + j * max_index + k;
}

inline int HashGrid::calcIndex(float f){
  return f < 0 ? f-.5:f+.5;
}

#endif
