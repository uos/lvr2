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
  void createCorners(ColorVertex*, BaseVertex);
  void createIntersections(ColorVertex corners[],
					  DistanceFunction* df,
					  ColorVertex intersections[]);

  int calcIndex() const;

  float calcIntersection(float x1, float x2,
					float d1, float d2, bool interpolate);
  
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


  ColorVertex corners[8];
  ColorVertex intersections[12];
  float distance[8];
  bool configuration[8];
  
  ANNpointArray points;
  
  //Distance function
  DistanceFunction* distance_function;
};

#endif
