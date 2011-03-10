/*
 * FastGrid.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#include "FastGrid.h"
#include "Timestamp.h"
#include "Progress.h"

#include "../rewrite/ObjIO.hpp"

#include <boost/progress.hpp>
#include <boost/filesystem.hpp>

#include "../model3d/PLYWriter.h"


//Each box corner in the grid is shared with 7 other boxes.
//To find an already existing corner, these boxes have to
//be checked. The following table holds the information where
//to look for a given corner. The coding is as follows:
//
//Table row = query vertex
//
//Each row consists of 7 quadruples. The first three numbers
//indicate, how the indices in x- y- and z-direction have to
//be modified. The fourth entry is the vertex of the box
//correspondig to the modified indices.
//
//Example: index_x = 10, index_y = 7, index_z = 5
//
//Query vertex = 5
//
//First quadruple: {+1, 0, +1, 0}
//
//Indices pointing to the nb-box: 10 + 1, 7 + 0, 5 + 1.
//--> The first shared vertex is vertex number 0 of the box in position
//(11, 7, 6) of the grid.
//
//Simple isn't it?

const static int shared_vertex_table[8][28] = {
	{-1, 0, 0, 1, -1, -1, 0, 2,  0, -1, 0, 3, -1,  0, -1, 5, -1, -1, -1, 6,  0, -1, -1, 7,  0,  0, -1, 4},
	{ 1, 0, 0, 0,  1, -1, 0, 3,  0, -1, 0, 2,  0,  0, -1, 5,  1,  0, -1, 4,  1, -1, -1, 7,  0, -1, -1, 6},
	{ 1, 1, 0, 0,  0,  1, 0, 1,  1,  0, 0, 3,  1,  1, -1, 4,  0,  1, -1, 5,  0,  0, -1, 6,  1,  0, -1, 7},
	{ 0, 1, 0, 0, -1,  1, 0, 1, -1,  0, 0, 2,  0,  1, -1, 4, -1,  1, -1, 5, -1,  0, -1, 6,  0,  0, -1, 7},
{ 0, 0, 1, 0, -1,  0, 1, 1, -1, -1, 1, 2,  0, -1,  1, 3, -1,  0,  0, 5, -1, -1,  0, 6,  0, -1,  0, 7},
	{ 1, 0, 1, 0,  0,  0, 1, 1,  0, -1, 1, 2,  1, -1,  1, 3,  1,  0,  0, 4,  0, -1,  0, 6,  1, -1,  0, 7},
	{ 1, 1, 1, 0,  0,  1, 1, 1,  0,  0, 1, 2,  1,  0,  1, 3,  1,  1,  0, 4,  0,  1,  0, 5,  1,  0,  0, 7},
	{ 0, 1, 1, 0, -1,  1, 1, 1, -1,  0, 1, 2,  0,  0,  1, 3,  0,  1,  0, 4, -1,  1,  0, 5, -1,  0,  0, 6}
};


//This table states where each coordinate of a box vertex is relatively
//to the box center
const static int box_creation_table[8][3] = {
	{-1, -1, -1},
	{ 1, -1, -1},
	{ 1,  1, -1},
	{-1,  1, -1},
	{-1, -1,  1},
	{ 1, -1,  1},
	{ 1,  1,  1},
	{-1,  1,  1}
};

FastGrid::FastGrid(Options *options) {

	m_options = options;

	// Parse options
	voxelsize = options->getVoxelsize();
	number_of_points = 0;
	string filename = options->getOutputFileName();

	readPoints(filename);

	cout << timestamp << "Finished Reading. Number of used points: " << number_of_points << endl;

	calcIndices();
	createGrid();
	calcQueryPointValues();
	createMesh();
}

FastGrid::~FastGrid() {

//	for(size_t i = 0; i < number_of_points; i++) delete[] points[i];
//	delete[] points;

	hash_map<int, FastBox*>::iterator it;
	for(it = cells.begin(); it != cells.end(); it++) delete it->second;
}


int  FastGrid::findQueryPoint(int position, int x, int y, int z){

	int n_x, n_y, n_z, q_v, offset;

	for(int i = 0; i < 7; i++){
		offset = i * 4;
		n_x = x + shared_vertex_table[position][offset];
		n_y = y + shared_vertex_table[position][offset + 1];
		n_z = z + shared_vertex_table[position][offset + 2];
		q_v = shared_vertex_table[position][offset + 3];

		int hash = hashValue(n_x, n_y, n_z);
		hash_map<int, FastBox*>::iterator it;
		it = cells.find(hash);
		if(it != cells.end()){
			FastBox* b = it->second;
			if(b->vertices[q_v] != -1) return b->vertices[q_v];
		}
	}

	return -1;

}

void FastGrid::createGrid(){

	//Create Grid
	cout << timestamp << "Creating Grid..." << endl;

	//Current indices
	int index_x, index_y, index_z;
	int hash_value;

	float vsh = voxelsize / 2.0;

	//Iterators
	hash_map<int, FastBox*>::iterator it;
	hash_map<int, FastBox*>::iterator neighbor_it;

	int global_index = 0;
	int current_index = 0;

	int dx, dy, dz;

	for(size_t i = 0; i < number_of_points; i++){
		index_x = calcIndex((points[i][0] - bounding_box.v_min.x) / voxelsize);
		index_y = calcIndex((points[i][1] - bounding_box.v_min.y) / voxelsize);
		index_z = calcIndex((points[i][2] - bounding_box.v_min.z) / voxelsize);


		for(int j = 0; j < 8; j++){

			dx = HGCreateTable[j][0];
			dy = HGCreateTable[j][1];
			dz = HGCreateTable[j][2];

			hash_value = hashValue(index_x + dx, index_y + dy, index_z +dz);
			it = cells.find(hash_value);
			if(it == cells.end()){
				//Calculate box center
				Vertex box_center = Vertex((index_x + dx) * voxelsize + bounding_box.v_min.x,
						                   (index_y + dy) * voxelsize + bounding_box.v_min.y,
						                   (index_z + dz) * voxelsize + bounding_box.v_min.z);

				//Create new box
				FastBox* box = new FastBox;

				//Setup the box itself
				for(int k = 0; k < 8; k++){

					//Find point in Grid
					current_index = findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);

					//If point exist, save index in box
					if(current_index != -1) box->vertices[k] = current_index;
					else{
					    //Otherwise create new grid point and associate it with the current box
						Vertex position(box_center.x + box_creation_table[k][0] * vsh,
								        box_center.y + box_creation_table[k][1] * vsh,
								        box_center.z + box_creation_table[k][2] * vsh);

						query_points.push_back(QueryPoint(position));

						box->vertices[k] = global_index;
						global_index++;

					}
				}

				//Set pointers to the neighbors of the current box
				int neighbor_index = 0;
				int neighbor_hash = 0;

				for(int a = -1; a < 2; a++){
					for(int b = -1; b < 2; b++){
						for(int c = -1; c < 2; c++){

							//Calculate hash value for current neighbor cell
							neighbor_hash = hashValue(index_x + dx + a,
									                  index_y + dy + b,
									                  index_z + dz + c);

							//Try to find this cell in the grid
							neighbor_it = cells.find(neighbor_hash);

							//If it exists, save pointer in box
							if(neighbor_it != cells.end()){
								box->neighbors[neighbor_index] = (*neighbor_it).second;
							}

							neighbor_index++;
						}
					}
				}

				cells[hash_value] = box;
			}
		}
	}
	cout << timestamp << "Finished Grid Creation. Number of generated cells:        " << cells.size() << endl;
	cout << timestamp << "Finished Grid Creation. Number of generated query points: " << query_points.size() << endl;
}

void FastGrid::calcQueryPointValues(){

    omp_set_num_threads(m_options->getNumThreads());

    string comment = timestamp.getElapsedTime() + "Calculating distance values ";
    ProgressBar progress((int)query_points.size(), comment);

    Timestamp ts;
    #pragma omp parallel for
	for(int i = 0; i < (int)query_points.size(); i++){
		//if(i % 1000 == 0) cout << "##### Calculating distance values: " << i << " / " << query_points.size() << endl;
		QueryPoint p = query_points[i];
		ColorVertex v = ColorVertex(p.position, 0.0f, 1.0f, 0.0f);
		p.distance = interpolator->distance(v);
		query_points[i] = p;
		++progress;
	}
	unsigned long end_time = GetCurrentTimeInMilliSec();

	cout << endl;
	cout << timestamp << "Elapsed time: " << ts << endl;
}

void FastGrid::createMesh(){

	string comment = timestamp.getElapsedTime() + "Creating Mesh ";

	hash_map<int, FastBox*>::iterator it;
	FastBox* b;
	int global_index = 0;

	omp_set_num_threads(m_options->getNumThreads());

	mesh = new HalfEdgeMesh();

	ProgressBar progress(cells.size(), comment);
	for(it = cells.begin(); it != cells.end(); it++){
		//if(c % 1000 == 0) cout << "##### Iterating Cells... " << c << " / " << cells.size() << endl;;
		b = it->second;
		global_index = b->calcApproximation(query_points, *mesh, global_index);
		++progress;
	}

	cout << endl;

//	mesh->extract_borders();
//	mesh->write_polygons("border.bor");

	if(m_options->saveFaceNormals())
	{
		mesh->write_face_normals("face_normals.nor");
	}

	if(m_options->createClusters())
	{
		vector<planarCluster> planes;
		mesh->cluster(planes);
		if(m_options->optimizeClusters()) mesh->optimizeClusters(planes);
		mesh->finalize(planes);
		mesh->save("planes.ply");

		list<list<planarCluster> > objects;
		mesh->classifyCluster(planes, objects);
		cout << "FOUND OBJECTS: " << objects.size() << endl;
		mesh->finalize(objects);
	    mesh->save("clusters.ply");
	}

	mesh->finalize();
	mesh->save("mesh.ply");

//	mesh->finalize(planes);
//	mesh->save("planes.ply");

//	cout << "##### Creating Progressive Mesh..." << endl;
//
//	mesh->pmesh();
//	mesh->finalize();
//	mesh->save("reduced.ply");

//	mesh->finalize();
//	mesh->save("mesh.ply");

//	mesh.printStats();
//	mesh.finalize();
//	mesh.save("mesh.ply");
	//he_mesh.analize();
	//he_mesh.extract_borders();
	//he_mesh.write_polygons("borders.bor");

	if(m_options->savePointsAndNormals())
	{
		cout << timestamp << "Saving points and normals..." << endl;
		savePointsAndNormals();
	}

	if(m_options->saveNormals())
	{
	    cout << timestamp << "Saving interpolated normals..." << endl;
	    static_cast<StannInterpolator*>(interpolator)->write_normals();
 	}

	// Test hack for obj support
	cout << timestamp << "Saving mesh.obj..." << endl;
	lssr::ObjIO<float, unsigned int> io;
	io.setVertexArray(mesh->getVertices(), mesh->getNumberOfVertices());
	io.setNormalArray(mesh->getNormals(),  mesh->getNumberOfVertices());
	io.setIndexArray(mesh->getIndices(), mesh->getNumberOfFaces());
	io.write("mesh.obj");

}

void FastGrid::calcIndices(){

	float max_size = max(max(bounding_box.x_size, bounding_box.y_size), bounding_box.z_size);

	//Save needed grid parameters
	max_index = (int)ceil( (max_size + 5 * voxelsize) / voxelsize);
	max_index_square = max_index * max_index;

	max_index_x = (int)ceil(bounding_box.x_size / voxelsize) + 1;
	max_index_y = (int)ceil(bounding_box.y_size / voxelsize) + 2;
	max_index_z = (int)ceil(bounding_box.z_size / voxelsize) + 3;

}


void FastGrid::readPoints(string filename)
{

	// Get file extension
	boost::filesystem::path selectedFile(filename);
	string extension = selectedFile.extension();

	if(extension == ".pts" || extension == ".3d" || extension == ".xyz")
	{
		readPlainASCII(filename);
	}
	else if(extension == ".ply")
	{
		readPLY(filename);
	}

}

void FastGrid::readPLY(string filename)
{
	cout << timestamp << "Reading " << filename << endl;
	PLYIO io;
	io.read(filename);

	// Get point cloud data
	size_t n = 0;
	points = io.getIndexedVertexArray(n);

	// Read normals if present
	float** normals = 0;
	if(io.containsElement("normal"))
	{
		normals = io.getIndexedNormalArray(n);
	}
	number_of_points = n;

	// Calc bounding box
	for(size_t i = 0; i < number_of_points; i++)
	{
		bounding_box.expand(points[i][0], points[i][1], points[i][2]);
	}

	interpolator = new StannInterpolator(points, normals, n, voxelsize, 100, 100.0);
	interpolator->setKd(m_options->getKd());
	interpolator->setKi(m_options->getKi());
	interpolator->setKn(m_options->getKn());
	interpolator->init();
}

void FastGrid::readPlainASCII(string filename)
{
	ifstream in(filename.c_str());

	//Vector to tmp-store points in file
	vector<BaseVertex> pts;

	//Read all points. Save maximum and minimum dimensions and
	//calculate maximum indices.
	int c = 0;

	//Get number of data fields to ignore
	int number_of_dummys = getFieldsPerLine(filename) - 3;

	//Point coordinates
	float dummy;

	//if(in.good()) in >> dummy;
	char line[1024];
	// Count points in file
	while(in.good())
	{
		in.getline(line, 1024);
		c++;
	}

	cout << timestamp << "Creating point array..." << endl;

	// Alloc memory fpr point cloud data
	points = new float*[c];
	for(int i = 0; i < c; i++) points[i] = new float[3];

	number_of_points = c;

	// Reopen
	in.close();
	in.open(filename.c_str());

	c = 0;
	//Read file
	string prefix = timestamp.getElapsedTime() + "Reading points...";
	ProgressCounter p_count(10000, prefix);

	while(in.good() ){
		in >> points[c][0] >> points[c][1] >> points[c][2];

		for(int i = 0; i < number_of_dummys; i++){
			in >> dummy;
		}


		bounding_box.expand(points[c][0], points[c][1], points[c][2]);
		if(c % 1 == 0) pts.push_back(BaseVertex(points[c][0],points[c][1],points[c][2]));
		c++;

		++p_count;
		//if(c % 100000 == 0) cout << timestamp << "Reading Points... " << c << endl;
	}
	cout << endl;
	cout << timestamp << "Finished Reading. " << endl;
	cout << timestamp << "Number of Data Points: " << pts.size() << endl;

	interpolator = new StannInterpolator(points, 0, number_of_points, 10.0, 100, 100.0);
    interpolator->setKd(m_options->getKd());
    interpolator->setKi(m_options->getKi());
    interpolator->setKn(m_options->getKn());
    interpolator->init();
}

int FastGrid::getFieldsPerLine(string filename){

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

void FastGrid::writeGrid(){
	cout << timestamp << "Writing 'grid.hg'" << endl;

	ofstream out("grid.hg");

	out << number_of_points << endl;

	for(size_t i = 0; i < number_of_points; i++){
		out << points[i][0] << " " <<  points[i][1] << " " << points[i][2] << endl;
	}

	hash_map<int, FastBox*>::iterator it;
	FastBox* box;

	for(it = cells.begin(); it != cells.end(); it++){
		box = it->second;

		for(int i= 0; i < 8; i++){
			QueryPoint qp = query_points[box->vertices[i]];
			Vertex v = qp.position;
			out << v.x << " " << v.y << " " << v.z << " " << 0.0 << " " << 1.0 << " " << 00. << endl;
		}
	}
}

void FastGrid::savePointsAndNormals()
{
	size_t n = 0;
	float** normals = interpolator->getNormals(n);

	PLYIO writer;

	// Create vertex element description
	PLYElement* vertex_element = new PLYElement("vertex", number_of_points);
	vertex_element->addProperty("x", "float");
	vertex_element->addProperty("y", "float");
	vertex_element->addProperty("z", "float");

	// Crate normal element description
	PLYElement* normal_element = new PLYElement("normal", n);
	normal_element->addProperty("x", "float");
	normal_element->addProperty("y", "float");
	normal_element->addProperty("z", "float");

	// Add description
	writer.addElement(vertex_element);
	writer.addElement(normal_element);

	// Add data
	writer.setIndexedVertexArray(points, number_of_points);
	writer.setIndexedNormalArray(normals, n);

	writer.save("points_and_normals.ply");
}
