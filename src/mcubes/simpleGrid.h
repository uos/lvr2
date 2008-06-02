#ifndef __SIMPLE_GRID_H__
#define __SIMPLE_GRID_H__

#include <ANN/ANN.h>
#include <fstream>

#include "../mesh/staticMesh.h"
#include "../io/plyWriter.h"

#include "distanceFunction.h"
#include "tables.h"

class SimpleGrid{

public:
  SimpleGrid(string filename, float voxelsize, float scale = 1.0);
  ~SimpleGrid();

  int readPoints(string filename, float scale);
  int getFieldsPerLine(string filename);

  void writeMesh();
  
private:

  void createMesh();
  
  StaticMesh mesh;
  
  float voxelsize;

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

  int number_of_points;
  
  ANNpointArray points;
  
  //Distance function
  DistanceFunction* distance_function;
};

#endif
