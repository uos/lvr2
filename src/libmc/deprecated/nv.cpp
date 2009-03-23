/*
 * nv.cpp
 *
 *  Created on: 05.03.2009
 *      Author: twiemann
 */

#include "NormalVoting.h"

int main(int argc, char** argv){

  string filename(argv[1]);

  float voxelsize;
  sscanf(argv[2], "%f", &voxelsize);

  system("clear");


  NormalVoting nv(filename, voxelsize);

  if(argc == 3){

  }

  return 0;

}
