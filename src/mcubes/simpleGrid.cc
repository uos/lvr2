#include "simpleGrid.h"

SimpleGrid::SimpleGrid(string filename, float v, float scale){

  //Initalize global variables
  voxelsize = v;

  xmin = 1.0e38;
  ymin = 1.0e38;
  zmin = 1.0e38;

  xmax = -xmin;
  ymax = -ymin;
  zmax = -zmin;

  //Read Data
  cout << "##### Reading data..." << endl;
  number_of_points = readPoints(filename, scale);

  //Create Distance Function
  distance_function = new DistanceFunction(points, number_of_points, 10, false);

  //Create Mesh
  createMesh();
  
}

int SimpleGrid::readPoints(string filename, float scale){

  ifstream in(filename.c_str());

  //Vector to tmp-store points in file
  vector<BaseVertex> pts;
  
  //Read all points. Save maximum and minimum dimensions and
  //calculate maximum indices.
  int c = 0;

  //Get number of data fields to ignore
  int number_of_dummys = getFieldsPerLine(filename) - 3;

  //Point coordinates
  float x, y, z, dummy;
 
  //Read file
  while(in.good()){
    in >> x >> y >> z;
    for(int i = 0; i < number_of_dummys; i++){
	 in >> dummy;
    }

    x *= scale;
    y *= scale;
    z *= scale;
    
    xmin = min(xmin, x);
    ymin = min(ymin, y);
    zmin = min(zmin, z);

    xmax = max(xmax, x);
    ymax = max(ymax, y);
    zmax = max(zmax, z);

    pts.push_back(BaseVertex(x,y,z));

    c++;

    if(c % 10000 == 0) cout << "##### Reading Points... " << c << endl;
  }

  cout << "##### Finished Reading. Number of Data Points: " << pts.size() << endl;


  //Calculate bounding box
  float x_size = fabs(xmax - xmin);
  float y_size = fabs(ymax - ymin);
  float z_size = fabs(zmax - zmin);

  float max_size = max(max(x_size, y_size), z_size);

  //Save needed grid parameters 
  max_index = (int)ceil( max_size / voxelsize) + 4;
  max_index_square = max_index * max_index;

  max_index_x = (int)ceil(x_size / voxelsize) + 4;
  max_index_y = (int)ceil(y_size / voxelsize) + 4;
  max_index_z = (int)ceil(z_size / voxelsize) + 4;

  //Create ANNPointArray
  cout << "##### Creating ANN Points " << endl; 
  points = annAllocPts(c, 3);

  for(size_t i = 0; i < pts.size(); i++){
    points[i][0] = pts[i].x;
    points[i][1] = pts[i].y;
    points[i][2] = pts[i].z;
  }

  pts.clear();
  
  return c;
}

void SimpleGrid::createMesh(){

  
  
}


int SimpleGrid::getFieldsPerLine(string filename){
  
  ifstream in(filename.c_str());

  //Get first line from file
  char first_line[1024];
  in.getline(first_line, 1024);
  in.close();

  //Get number of blanks
  int c = 0;
  char* pch = strtok(first_line, " ");
  while(pch != NULL){
    c++;
    pch = strtok(NULL, " ");
  }

  in.close();

  return c;
}


void SimpleGrid::writeMesh(){

  cout << "##### Writing 'mesh.ply'" << endl;
  
  PLYWriter w("mesh.ply");

  w.writeHeader();
  w.addMesh(mesh);
  w.writeFooter();
  
}
