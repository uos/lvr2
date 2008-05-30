#include "hashGrid.h"

#include <iostream>

int main(int argc, char** argv){

  string filename(argv[1]);

  float voxelsize;
  sscanf(argv[2], "%f", &voxelsize);

  if(argc == 3){
    HashGrid hashGrid(filename, voxelsize);
    hashGrid.writeMesh();
  } else if(argc == 4){

    float scale;
    sscanf(argv[3], "%f", &scale);
    
    HashGrid hashGrid(filename, voxelsize * scale, scale);
    hashGrid.writeMesh();
  }
 
  return 0;
  
}
