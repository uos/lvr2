//#include "HashGrid.h"
#include "FastGrid.h"
#include "Options.h"

#include <iostream>

int main(int argc, char** argv){

	Options options(argc, argv);
	if(options.printUsage()) exit(0);

	if(!options.filenameSet())
	{
		cout << "No output files set." << endl;
		exit(0);
	}

	int result = system("clear");
	if(result != 0) {} // do nothing, just prevent compiler warning!

	cout << options << endl;

	//  if(argc == 3){
	//    HashGrid hashGrid(filename, voxelsize);
	//    hashGrid.writeMesh();
	//    hashGrid.writeGrid();
	//    //SimpleGrid simpleGrid(filename, voxelsize);
	//    //simpleGrid.writeMesh();
	//
	//  } else if(argc == 4){
	//
	//    float scale;
	//    sscanf(argv[3], "%f", &scale);
	//
	//    HashGrid hashGrid(filename, voxelsize * scale, scale);
	//    hashGrid.writeMesh();
	//    hashGrid.writeGrid();
	//  }

	FastGrid fastGrid(&options);


	return 0;

}
