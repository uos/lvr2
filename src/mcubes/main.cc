#include "hashGrid.h"

#include <iostream>

int main(int argc, char** argv){

  string filename(argv[1]);

  float voxelsize;
  sscanf(argv[2], "%f", &voxelsize);

  HashGrid hashGrid(filename, voxelsize);
  hashGrid.writeMesh();
 
  return 0;
  
}
