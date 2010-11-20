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
	int mod = 1;
	int reduction = 0;

	if(argc < 3)
	{
		cout << "Usage: ascii2ply <infile> <outfile> (reduction)" << endl;
		return 0;
	}
	if(argc == 4)
	{
	    reduction = atoi(argv[3]);
		cout << "Reducing data to " << reduction << " points." << endl;
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
	cout << "Counting points..." << endl;
	while(in.good())
	{
		in.getline(line, max_line_size);
		number_of_points++;
	}
	cout << "Number of points in file: " << number_of_points << endl;
	in.close();

	// Calculate reduction factor
	if(reduction != 0)
	{
		mod = number_of_points / reduction + 1;
		cout << "Reducing data using every " << mod << "th point." << endl;
	}

	// Calculate number of points to read
	number_of_points /= mod;
	cout << "Reading a total of " << number_of_points << " points." << endl;

	// Allocate memory for point data and read points
	cout << "Allocating point array" << endl;
	float* points = new float[3 * number_of_points];
	in.open(inFileName.c_str());
	int c = 0, pos = 0;
	char junk[1024];

	float x, y, z;


	while(in.good())
	{
		in.getline(line, max_line_size);
		// Parse points
		if(c % mod == 0 && pos < number_of_points && in.good())
		{
			int index = pos * 3;
			sscanf(line, "%f %f %f %s", &x, &y, &z, junk);
			points[index] = x;
			points[index + 1] = y;
			points[index + 2] = z;
			//cout << x << " " << y << " " << z << endl;
			pos++;
		}
		if(c % 10000000 == 0) cout << "Reading points: " << c << " / " << pos << endl;
		c++;
	}
	cout << "Read " << c << " data points (" << number_of_points << ") estimated." <<  endl;

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
