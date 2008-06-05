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

SimpleGrid::~SimpleGrid(){
  annDeallocPts(points);
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

  int max_number = (max_index_x + 4) * (max_index_y + 4) * (max_index_z + 4);
  int counter = 0, vertex_counter = 0;
  int edge_index = 0;
  
  BaseVertex diff1, diff2;
  Normal normal;
  
  float x, y, z;


  for(int i = -2; i < max_index_x + 2; i++){
    for(int j = -2; j < max_index_y + 2; j++){
	 for(int k = -2; k < max_index_z + 2; k++){
	   if(counter % 10000 == 0) cout << counter << " / " << max_number << ": " << mesh.vertices.size() << endl;

	   x = i * voxelsize + xmin;
	   y = j * voxelsize + ymin;
	   z = k * voxelsize + zmin;

	   //cout << x << " " << y << " " << z << endl;

	   createCorners(corners, BaseVertex(x, y, z));
	   createIntersections(corners, distance_function, intersections);

	   for(int a = 0; a < 8; a++){
		if(distance[a] > 0)
		  configuration[a] = true;
		else
		  configuration[a] = false;				  
	   }
	   
	   edge_index = calcIndex();

	
	   for(int a = 0; MCTable[edge_index][a] != -1; a+= 3){

		for(int b = 0; b < 3; b++){
		  int vertex_index = MCTable[edge_index][a+b];
		  mesh.addVertex(intersections[vertex_index]);
		  mesh.addNormal(Normal(0.0, 0.0, 0.0));
		  mesh.addIndex(vertex_counter);
		  vertex_counter++;
		}

		diff1 = mesh.vertices[vertex_counter - 3] - mesh.vertices[vertex_counter - 2];
		diff2 = mesh.vertices[vertex_counter - 3] - mesh.vertices[vertex_counter - 1];

		normal = diff1.cross(diff2);
		
		for(int b = 1; b <= 3; b++){
		  mesh.interpolateNormal(vertex_counter - b, normal);
		}
		
	   }

	   counter++;
	 }
    }
  }
  
}

int SimpleGrid::calcIndex() const{
  int index = 0;
  for(int i = 0; i < 8; i++){
    if(configuration[i] > 0) index |= (1 << i);
  }
  return index;
}

void SimpleGrid::createCorners(ColorVertex corners[], BaseVertex baseVertex){

  uchar r = 0;
  uchar g = 200;
  uchar b = 0;
  
  corners[0] = ColorVertex(baseVertex, r, g, b);

  corners[1] = ColorVertex(baseVertex.x + voxelsize,
					  baseVertex.y,
					  baseVertex.z,
					  r, g, b);

  corners[2] = ColorVertex(baseVertex.x + voxelsize,
					  baseVertex.y + voxelsize,
					  baseVertex.z,
					  r, g, b);

  corners[3] = ColorVertex(baseVertex.x,
					  baseVertex.y + voxelsize,
					  baseVertex.z,
					  r, g, b);

  corners[4] = ColorVertex(baseVertex.x,
					  baseVertex.y,
					  baseVertex.z + voxelsize,
					  r, g, b);

  corners[5] = ColorVertex(baseVertex.x + voxelsize,
					  baseVertex.y,
					  baseVertex.z + voxelsize,
					  r, g, b);

  corners[6] = ColorVertex(baseVertex.x + voxelsize,
					  baseVertex.y + voxelsize,
					  baseVertex.z + voxelsize,
					  r, g, b);

  corners[7] = ColorVertex(baseVertex.x,
					  baseVertex.y + voxelsize,
					  baseVertex.z + voxelsize,
					  r, g, b);
  
}

void SimpleGrid::createIntersections(ColorVertex corners[],
							  DistanceFunction* df,
							  ColorVertex intersections[]){

  //bool interpolate = (df != 0);
  bool interpolate = true;
  float d1, d2;
  d1 = d2 = 0.0;

  uchar current_color[] = {0, 200, 0};

  float intersection;

  Normal normal;
  
 //Calc distances;
  for(int i = 0; i < 8; i++){
    configuration[i] = false;
    bool ok = true;
    distance[i] = df->distance(corners[i], normal, 500, 1000.0, Z, ok);
  }
  
  //Front Quad
  intersection = calcIntersection(corners[0].x, corners[1].x, distance[0], distance[1], interpolate);
  intersections[0] = ColorVertex(intersection, corners[0].y, corners[0].z,
						   current_color[0], current_color[1], current_color[2]);
 


  intersection = calcIntersection(corners[1].y, corners[2].y, distance[1], distance[2], interpolate);
  intersections[1] = ColorVertex(corners[1].x, intersection, corners[1].z,
						   current_color[0], current_color[1], current_color[2]);

  
  
  intersection = calcIntersection(corners[3].x, corners[2].x, distance[3], distance[2], interpolate);
  intersections[2] = ColorVertex(intersection, corners[2].y, corners[2].z,
						   current_color[0], current_color[1], current_color[2]);

    
  intersection = calcIntersection(corners[0].y, corners[3].y, distance[0], distance[3], interpolate);
  intersections[3] = ColorVertex(corners[3].x, intersection, corners[3].z,
						   current_color[0], current_color[1], current_color[2]);
 
  //Back Quad
  intersection = calcIntersection(corners[4].x, corners[5].x, distance[4], distance[5], interpolate);
  intersections[4] = ColorVertex(intersection, corners[4].y, corners[4].z,
						   current_color[0], current_color[1], current_color[2]);
  
  intersection = calcIntersection(corners[5].y, corners[6].y, distance[5], distance[6], interpolate);
  intersections[5] = ColorVertex(corners[5].x, intersection, corners[5].z,
						   current_color[0], current_color[1], current_color[2]);

    
  intersection = calcIntersection(corners[7].x, corners[6].x, distance[7], distance[6], interpolate);
  intersections[6] = ColorVertex(intersection, corners[6].y, corners[6].z,
						   current_color[0], current_color[1], current_color[2]);
  
 
  intersection = calcIntersection(corners[4].y, corners[7].y, distance[4], distance[7], interpolate);
  intersections[7] = ColorVertex(corners[7].x, intersection, corners[7].z,
						   current_color[0], current_color[1], current_color[2]);
 
  
  //Sides
  intersection = calcIntersection(corners[0].z, corners[4].z, distance[0], distance[4], interpolate);
  intersections[8] = ColorVertex(corners[0].x, corners[0].y, intersection,
						   current_color[0], current_color[1], current_color[2]); 

  intersection = calcIntersection(corners[1].z, corners[5].z, distance[1], distance[5], interpolate);
  intersections[9] = ColorVertex(corners[1].x, corners[1].y, intersection,
						   current_color[0], current_color[1], current_color[2]); 

  intersection = calcIntersection(corners[3].z, corners[7].z, distance[3], distance[7], interpolate);
  intersections[10] = ColorVertex(corners[3].x, corners[3].y, intersection,
						    current_color[0], current_color[1], current_color[2]); 

  intersection = calcIntersection(corners[2].z, corners[6].z, distance[2], distance[6], interpolate);
  intersections[11] = ColorVertex(corners[2].x, corners[2].y, intersection,
						    current_color[0], current_color[1], current_color[2]);

 
}

float SimpleGrid::calcIntersection(float x1, float x2,
					   float d1, float d2, bool interpolate){

  return x2 - d2 * (x1 - x2) / (d1 - d2);
  
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
