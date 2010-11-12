#include <iostream>
#include <fstream>

#include <cstring>

#include <lib3d/PointCloud.h>
#include <lib3d/PLYWriter.h>

using namespace std;

int getFieldsPerLine(string filename)
{
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

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout << "Usage: ascii2ply <infile> <outfile>" << endl;
		return 0;
	}

	PLYIO ply_writer;

	string inFileName = string(argv[1]);
	string outFileName = string(argv[2]);

	// Get number of data entrys per line in input file
	int number_of_dummys = getFieldsPerLine(inFileName) - 3;

	// Count points
	size_t max_line_size = 1024;
	char line[1024];
	size_t number_of_points = 0;
	ifstream in;
	in.open(inFileName.c_str());
	while(in.good())
	{
		in.getline(line, max_line_size);
		number_of_points++;
		if(number_of_points % 10000000 == 0)
		{
			cout << "Counting points: " << number_of_points << endl;
		}
	}
	cout << "Number of points in file: " << number_of_points << endl;
	in.close();

	// Allocate memory for point data and read points
	cout << "Allocating point array" << endl;
	float* points = new float[3 * number_of_points];
	in.open(inFileName.c_str());
	int c = 0;
	char junk[1024];

	float x, y, z;


	while(in.good())
	{
		int index = c * 3;
		in.getline(line, max_line_size);
		sscanf(line, "%f %f %f %s", &x, &y, &z, junk);
		points[c] = x;
		points[c + 1] = y;
		points[c + 2] = z;
		if(c % 10000000 == 0) cout << "Reading points: " << c << endl;
		c++;
	}
	cout << "Read " << c << " data points" << endl;

	// Create PLY file
	PLYElement* vertex_element = new PLYElement("vertex", number_of_points);
	vertex_element->addProperty("x", "float");
	vertex_element->addProperty("y", "float");
	vertex_element->addProperty("z", "float");

	ply_writer.addElement(vertex_element);
	ply_writer.setVertexArray(points, number_of_points);
	ply_writer.save(outFileName);

	return 0;
}
