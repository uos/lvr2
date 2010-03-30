#include "PLYWriter.h"
#include <cstring>
#include <sstream>
#include <cassert>

using std::stringstream;

#include <boost/algorithm/string.hpp>


using boost::algorithm::to_lower;
using boost::algorithm::is_equal;

PLYIO::PLYIO()
{
	m_vertices 		= 0;
	m_normals		= 0;
	m_colors		= 0;
	m_indices		= 0;
	m_binary		= false;;

 }


PLYIO::~PLYIO()
{
	cout << "PLYIO::Dtor" << endl;
}
float* PLYIO::getVertexArray(size_t &n)
{
	if(m_vertices)
	{
		n = m_numberOfVertices;
	}
	else
	{
		n = 0;
	}
	return m_vertices;
}

float* PLYIO::getNormalArray(size_t &n)
{
	if(m_normals)
	{
		n = m_numberOfVertices;
	}
	else
	{
		n = 0;
	}
	return m_normals;
}

float* PLYIO::getColorArray(size_t &n)
{
	if(m_colors)
	{
		n = m_numberOfVertices;
	}
	else
	{
		n = 0;
	}
	return m_colors;
}

unsigned int* PLYIO::getIndexArray(size_t &n)
{
	if(m_indices)
	{
		n = m_numberOfFaces;
	}
	else
	{
		n = 0;
	}
	return m_indices;
}

void PLYIO::setVertexArray(float* array, size_t n)
{
	m_vertices = array;
	m_numberOfVertices = n;
}

void PLYIO::setIndexArray(unsigned int* array, size_t n)
{
	m_indices = array;
	m_numberOfFaces = n;
}

void PLYIO::save(string filename, bool binary)
{
	// Save file mode
	m_binary = binary;

	// Open stream to write file
	ofstream out(filename.c_str());

	if(out.good())
	{
		writeHeader(out);
		writeElements(out);
	}
	else
	{
		cout << "PLYIO::save(): Unable to open file '"
			 << filename << "'." << endl;
	}

	// Close stream
	out.close();
}

void PLYIO::read(string filename)
{
	ifstream in(filename.c_str());

	if(in.good())
	{
		deleteBuffers();
		readHeader(in);
		loadElements(in);
	}
	else
	{
		cout << "PLYIO::read(): Unable to open file '"
			 << filename << "'." << endl;
	}

}

void PLYIO::readHeader(ifstream& in)
{
	// Char buffer to read lines from file
	char line[1024];

	// Check for magic number (read first word in file)
	in.getline(line, 1024);
	string ply_tag(line);


	if(ply_tag.compare("ply") != 0)
	{
		cout << "PLYIO::readHeader(): No PLY tag in first line." << endl;
		return;
	}

	// Read until the "end_header" tag was found
	do
	{
		in.getline(line, 1024);
	}
	while(parseHeaderLine(line));
}

bool PLYIO::parseHeaderLine(const char* line)
{
	// Convert to std::string for convenience
	string line_str(line);

	// Convert to lower characters to avoid parsing problems
	to_lower(line_str);

	// Check for end header
	if(line_str.compare("end_header") == 0)
	{
		return false;
	}
	else
	{
		// Create a stream for line parsing
		stringstream ss(line_str);

		// Get element / property description
		string element_dscr;
		ss >> element_dscr;

		// Create element / property descriptions according to line
		if(element_dscr.compare("comment") == 0)
		{
			// Ignore comments
			return true;
		}
		else if(element_dscr.compare("format") == 0)
		{
			// Check format
			string format;
			ss >> format;
			if(format.compare("ascii") == 0)
			{
				m_binary = false;
				return true;
			}
			else if(format.compare("binary_little_endian") == 0)
			{
				m_binary = true;
				return true;
			}
			else if(format.compare("binary_big_endian") == 0)
			{
				m_binary = true;
				cout << "PLYIO::parseHeaderLine(): Error: Big endianess is not yet supported." << endl;
				// Cancel further reading
				return false;
			}
			else
			{
				cout << "PLYIO::parseHeaderLine(): Error: Unknown format'"
				     << format << "'." << endl;
				return false;
			}

 		}
		else if(element_dscr.compare("element") == 0)
		{
			// Read element name
			size_t element_count;
			string element_name;
			ss >> element_name >> element_count;

			// Add new element
			m_elements.push_back(new PLYElement(element_name, element_count));

			// Continue parsing
			return true;
		}
		else if(element_dscr.compare("property") == 0)
		{
			assert(m_elements.size() > 0);

			string type;
			ss >> type;
			if(type.compare("list") == 0)
			{
				string count_type, value_type, property_name;
				ss >> count_type >> value_type >> property_name;

				(m_elements.back())->addProperty(property_name, value_type, count_type);
			}
			else
			{
				string property_name;
				ss >> property_name;
				(m_elements.back())->addProperty(property_name, type);
			}

			return true;
		}
		else
		{
			// Something went wrong...
			cout << "PLYIO::parseHeaderLine(): Error: Invalid header line: '"
				 << line_str << endl;
			return false;
		}
	}

}

void PLYIO::loadElements(ifstream &in)
{

	vector<PLYElement*>::iterator it;
	PLYElement* current_element;

	for(it = m_elements.begin(); it != m_elements.end(); it++)
	{
		current_element = *it;
		string element_name = current_element->getName();
		if(isSupported(element_name))
		{
			// Here we have to decide if an given element is
			// currently supported. Furthermore we have to call
			// the appropriate loading procedure
			if(element_name == "vertex")
			{

				// Load vertex elements
				if(m_binary)
				{
					readVerticesBinary(in, current_element);
				}
				else
				{
					readVerticesASCII(in, current_element);
				}
			}
			else if (element_name == "face")
			{
				if(m_binary)
				{
					readFacesBinary(in, current_element);
				}
				else
				{
					readFacesASCII(in, current_element);
				}
			}
			else
			{
				cout << "PLYIO::loadElements(): Warning: Unknown element '"
					 << element_name << "'." << endl;
			}
		}
		else
		{
			// Skip unsupported elements
			if(m_binary)
			{
				cout << "ERROR: SKIPPING OF UNSUPPORTED ELEMENTS NOT YET IMPLEMENTED. DO IT NOW!!!" << endl;
			}
			else
			{
				// Skip the lines defining the unsupported
				// elements
				char buffer[1024];
				for(size_t i = 0; i < current_element->getCount(); i++)
				{
					in.getline(buffer, 1024);
				}
			}
		}
	}
}

void PLYIO::printElementsInHeader()
{
	cout << "------ Elements in PLY Header ------" << endl << endl;
	for(size_t i = 0; i < m_elements.size(); i++)
	{
		PLYElement* e = m_elements[i];
		cout << e->getName() << " " << e->getCount() << endl;
		vector<Property*>::iterator start, end;
		start = e->getFirstProperty();
		end = e->getLastProperty();
		while(start != end)
		{
			cout << (*start)->getElementTypeStr() << " "
				 << (*start)->getName() << " "
			     << (*start)->getCountTypeStr() << endl;
			start++;
		}
		cout << endl;
	}
	cout << "------------------------------------" << endl;
}

void PLYIO::writeHeader(ofstream& out)
{
	// Write file recognition tag
	out << "ply" << endl;

	// Write format information
	if(m_binary)
	{
		out << "format binary_little_endian 1.0" << endl;
	}
	else
	{
		out << "format ascii 1.0" << endl;
	}

	// Write element descriptions
	for(size_t i = 0; i < m_elements.size(); i++)
	{
		PLYElement* element = m_elements[i];
		out << "element "
			<< element->getName() 	<< " "
		    << element->getCount() 	<< endl;
		element->printProperties(out);
	}

	// Write end header mark
	out << "end_header" << endl;
}

void PLYIO::writeElements(ofstream &out)
{
	// Check for supported elements and write them in
	// the declared order
	for(size_t i = 0; i < m_elements.size(); i++)
	{
		cout << m_elements[i]->getName() << endl;
		if(m_elements[i]->getName() == "vertex")
		{
			m_binary ? writeVerticesBinary(out, m_elements[i])
					 : writeVerticesASCII(out, m_elements[i]);
		}

		if(m_elements[i]->getName() == "face")
		{

			m_binary ? writeFacesBinary(out, m_elements[i]) : writeFacesASCII(out, m_elements[i]);
		}
	}
}

void PLYIO::writeVerticesASCII(ofstream &out, PLYElement *e)
{
	assert(m_vertices);

	vector<Property*>::iterator current, last;
	string property_name;

	for(size_t i = 0; i < m_numberOfVertices; i++)
	{
		size_t vertex_pointer = i * 3;

		// Since we don't know the order in which the properties are
		// written, we have to determine the order. Right know i don't
		// know how to solve this nicely, so I just iterate over
		// the property vector for each vertex and write the corresponding
		// information
		current = e->getFirstProperty();
		last = e->getLastProperty();

		while(current != last)
		{
			property_name = (*current)->getName();
			if(property_name == "x")
			{
				out << m_vertices[vertex_pointer] << " ";
			}
			else if (property_name == "y")
			{
				out << m_vertices[vertex_pointer + 1] << " ";
			}
			else if (property_name == "z")
			{
				out << m_vertices[vertex_pointer + 2] << " ";
			}
			current++;
		}
		out << endl;
	}
}

void PLYIO::writeFacesASCII(ofstream &out, PLYElement *e)
{
	assert(m_indices);

	for(size_t i = 0; i < m_numberOfFaces; i++)
	{
		int index_pointer = i * 3;
		// Write number of vertices per face (currently 3)
		out << "3 ";
		out << m_indices[index_pointer]     << " "
			<< m_indices[index_pointer + 1] << " "
			<< m_indices[index_pointer + 2] << endl;
	}
}

void PLYIO::writeVerticesBinary(ofstream &out, PLYElement* e)
{
	assert(m_vertices);

	// Iterators for property traversal
	vector<Property*>::iterator current, first, last;
	first = e->getFirstProperty();
	last = e->getLastProperty();

	// Determine the number of bytes we need to write all
	// properties. Assume that only supported properties
	// are given.
	size_t buffer_size = 0;
	for(current = first; current != last; current++)
	{
		buffer_size += (*current)->getValueSize();
	}

	// Allocate buffer memory
	char buffer[buffer_size];
	char *pos;

	// Iterate through all vertices and properties (same problem
	// as with writing in ASCII) and copy the corresponding data
	// into the buffer
	for(size_t i = 0; i < m_numberOfVertices; i++)
	{
		size_t vertex_pointer = i * 3;

		// Reset buffer
		memset(buffer, 0, buffer_size);
		pos = buffer;

		// Parse properties and write into buffer
		for(current = first; current != last; current++)
		{
			Property* p = (*current);
			string property_name = p->getName();
			string property_type = p->getElementTypeStr();
			if(property_name == "x")
			{
				pos = putElementInBuffer(pos, property_type, m_vertices[vertex_pointer]);
			}
			else if (property_name == "y")
			{
				pos = putElementInBuffer(pos, property_type, m_vertices[vertex_pointer + 1]);
			}
			else if (property_name == "z")
			{
				pos = putElementInBuffer(pos, property_type, m_vertices[vertex_pointer + 2]);
			}
			else
			{
				pos = putElementInBuffer(pos, property_type, 0);
			}
		}

		// Write buffer to stream
		out.write(buffer, buffer_size);
	}

}

char* PLYIO::putElementInBuffer(char* buffer, string value_type, float value)
{

	if(value_type == "char")
	{
		char c = (char)value;
		memcpy(buffer, &c, sizeof(char));
		buffer ++;
	}
	else if (value_type == "uchar")
	{
		unsigned char c = (unsigned char)value;
		memcpy(buffer, &c, sizeof(unsigned char));
		buffer ++;
	}
	else if (value_type == "short")
	{
		short s = (short)value;
		memcpy(buffer, &s, sizeof(short));
		buffer += 2;
	}
	else if (value_type == "ushort")
	{
		unsigned short s = (unsigned short)value;
		memcpy(buffer, &s, sizeof(unsigned short));
		buffer += 2;
	}
	else if (value_type == "int")
	{
		int i = (int)value;
		memcpy(buffer, &i, sizeof(int));
		buffer += 4;
	}
	else if (value_type == "uint")
	{
		unsigned int i = (unsigned int)value;
		memcpy(buffer, &i, sizeof(unsigned int));
		buffer += 4;
	}
	else if (value_type == "float")
	{
		memcpy(buffer, &value, sizeof(float));
		buffer += 4;
	}
	else if (value_type == "double")
	{
		double d = (double)value;
		memcpy(buffer, &d, sizeof(double));
		buffer += 8;
	}

	return buffer;
}

void PLYIO::allocVertexBuffers(PLYElement* descr)
{

	// Allocate memory for vertex positions (always required)
	m_vertices = new float[3 * m_numberOfVertices];
	memset(m_vertices, 0, 3 * m_numberOfVertices * sizeof(float));

	// Check if we have to allocate memory for colors: Iterate
	// through all properties and check if we find a color
	// attribute and allocate buffer if found
	vector<Property*>::iterator it;
	for(it = descr->getFirstProperty(); it != descr->getLastProperty(); it++)
	{
		string property_name = (*it)->getName();
		if(property_name == "r" || property_name == "g" || property_name == "b")
		{
			// Leave loop if a color property was found
			m_colors = new float[3 * m_numberOfVertices];
			memset(m_colors, 0, 3 * sizeof(float));
			break;
		}
	}
}

void PLYIO::deleteBuffers()
{

	if(m_vertices)
	{
		delete[] m_vertices;
		m_vertices = 0;
	}

	if(m_colors)
	{
		delete[] m_colors;
		m_colors = 0;
	}

	if(m_indices)
	{
		delete[] m_indices;
		m_indices = 0;
	}
}

void PLYIO::readVerticesASCII(ifstream &in, PLYElement* descr)
{

	// Get number of vertices
	m_numberOfVertices = descr->getCount();

	// Allocate memory for needed buffers
	allocVertexBuffers(descr);

	cout << m_numberOfVertices << endl;

	// Read all vertices
	vector<Property*>::iterator it;
	for(size_t i = 0; i < m_numberOfVertices; i++)
	{
		// Parse through all properties and load the
		// supported ones
		it = descr->getFirstProperty();
		while(it != descr->getLastProperty())
		{
			Property* p = *it;
			string property_name = p->getName();
			if(property_name == "x")
			{
				in >> m_vertices[i * 3];
			}
			else if(property_name == "y")
			{
				in >> m_vertices[i * 3 + 1];
			}
			else if(property_name == "z")
			{
				in >> m_vertices[i * 3 + 2];
			}
			else if(property_name == "r")
			{
				in >> m_colors[i * 3];
			}
			else if(property_name == "g")
			{
				in >> m_colors[i * 3 + 1];
			}
			else if(property_name == "b")
			{
				in >> m_colors[i * 3 + 2];
			}
			it++;
		}
//		cout << m_vertices[3 * i    ] << " "
//			 << m_vertices[3 * i + 1] << " "
//			 << m_vertices[3 * 1 + 2] << endl;
	}
}

void PLYIO::readVerticesBinary(ifstream &in, PLYElement* descr)
{
	cout << "READ VERTICES" << endl;
	// Get number of vertices
	m_numberOfVertices = descr->getCount();

	// Allocate memory for needed buffers
	allocVertexBuffers(descr);

	// Read all vertices
	vector<Property*>::iterator it;
	for(size_t i = 0; i < m_numberOfVertices; i++)
	{
		for(it = descr->getFirstProperty(); it != descr->getLastProperty(); it++)
		{
			// TODO: Calculate buffer position only once.
			Property* p = *it;
			if(p->getName() == "x")
			{
				copyElementToVertexBuffer(in, p, m_vertices,  i * 3);
			}
			else if(p->getName() == "y")
			{
				copyElementToVertexBuffer(in, p, m_vertices, i * 3 + 1);
			}
			else if(p->getName() == "z")
			{
				copyElementToVertexBuffer(in, p, m_vertices, i * 3 + 2);
			}
			else if(p->getName() == "r")
			{
				copyElementToVertexBuffer(in, p, m_colors, i * 3);
			}
			else if(p->getName() == "g")
			{
				copyElementToVertexBuffer(in, p, m_colors, i * 3 + 1);
			}
			else if(p->getName() == "b")
			{
				copyElementToVertexBuffer(in, p, m_colors, i * 3 + 2);
			}
		}
//		cout << m_vertices[i    ] << " "
//			 << m_vertices[i + 1] << " "
//			 << m_vertices[i + 2] << endl;
	}
}

// TODO: Maybe we can write some kind of template implementation???
void PLYIO::copyElementToVertexBuffer(ifstream &in, Property* p, float* buffer, size_t position)
{
	if(p->getElementTypeStr() == "char")
	{
		char tmp = 0;
		in.read(&tmp, sizeof(char));
		buffer[position] = (float)tmp;
	}
	else if(p->getElementTypeStr() == "uchar")
	{
		unsigned char tmp = 0;
		in.read((char*)&tmp, sizeof(unsigned char));
		buffer[position] = (float)tmp;
	}
	else if(p->getElementTypeStr() == "short")
	{
		short tmp = 0;
		in.read((char*)&tmp, sizeof(short));
		buffer[position] = (float)tmp;
	}
	else if(p->getElementTypeStr() == "ushort")
	{
		unsigned short tmp = 0;
		in.read((char*)&tmp, sizeof(unsigned short));
		buffer[position] = (float)tmp;
	}
	else if(p->getElementTypeStr() == "int")
	{
		int tmp = 0;
		in.read((char*)&tmp, sizeof(int));
		buffer[position] = (float)tmp;
	}
	else if(p->getElementTypeStr() == "uint")
	{
		unsigned int tmp = 0;
		in.read((char*)&tmp, sizeof(unsigned int));
		buffer[position] = (float)tmp;
	}
	else if(p->getElementTypeStr() == "float")
	{
		float tmp;
		in.read((char*)&tmp, sizeof(float));
		buffer[position] = tmp;
	}
	else if(p->getElementTypeStr() == "double")
	{
		double tmp = 0;
		in.read((char*)&tmp, sizeof(double));
		buffer[position] = (float)tmp;
	}
}

void PLYIO::readFacesASCII(ifstream &in, PLYElement* descr)
{

	// Save number of vertices
	m_numberOfFaces = descr->getCount();

	// Allocate memory
	m_indices = new unsigned int[3 * m_numberOfFaces];

	// Read indices
	int count, a, b, c, position;
	for(size_t i = 0; i < m_numberOfFaces; i++)
	{
		position = i * 3;
		in >> count >> a >> b >> c;
		//cout << count << " " << a << " " << b << " " << c << endl;
		if(count != 3) cout << "PLYIO::readFacesASCII(): Warning: Indexed face is not a triangle." << endl;
		m_indices[position    ] = a;
		m_indices[position + 1] = b;
		m_indices[position + 2] = c;
	}

}

void PLYIO::readFacesBinary(ifstream &in, PLYElement* descr)
{
	// Save number of vertices
	m_numberOfFaces = descr->getCount();

	// Allocate memory
	m_indices = new unsigned int[3 * m_numberOfFaces];

	// Parse propertys: Search for 'vertex_indices' and
	// save number of bytes for counter and vertex indices
	size_t count_size, value_size;

	Property* list_property = 0;
	vector<Property*>::iterator it;
	for(it = descr->getFirstProperty(); it != descr->getLastProperty(); it++)
	{
		Property* p = *it;
		if(p->getName() == "vertex_indices" || p->getName() == "vertex_index")
		{
			count_size = p->getCountSize();
			value_size = p->getValueSize();
			list_property = p;
			break;
		}
	}

	// Be sure that we found a vertex_list property
	if(list_property == 0)
	{
		cout << "PLYIO::readFacesBinary() : Warning 'vertex_indices' property not found." << endl;
		// TO DO: Fix possible leak here.
		return;
	}

	// Get count and value type names
	string count_name = list_property->getCountTypeStr();
	string value_name = list_property->getElementTypeStr();

	// A pointer to save the current position while parsing the
	// read chunks
	char* chunk_position;

	// Allocate memory for index information
	char* chunk_buffer = new char[count_size + 3 * value_size];

	for(size_t i = 0; i < m_numberOfFaces; i++)
	{
		chunk_position = chunk_buffer;

		// Read index info from file
		in.read(chunk_buffer, count_size + 3 * value_size);

		// Parse buffer -> Get vertex count as unsigned int
		unsigned int vertex_count;
		if(count_name == "char")
		{
			char tmp;
			memcpy(&tmp, chunk_buffer, sizeof(char));
			chunk_position += sizeof(char);
			vertex_count = (unsigned int)tmp;
		}
		else if(count_name == "uchar")
		{
			unsigned char tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(unsigned char));
			chunk_position += sizeof(unsigned char);
			vertex_count = (unsigned int)tmp;
			cout << vertex_count << endl;
		}
		else if(count_name == "short")
		{
			short tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(short));
			chunk_position += sizeof(short);
			vertex_count = (unsigned int)tmp;
		}
		else if(count_name == "ushort")
		{
			unsigned short tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(unsigned short));
			chunk_position += sizeof(unsigned short);
			vertex_count = (unsigned int)tmp;
		}
		else if(count_name == "int")
		{
			int tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(int));
			chunk_position += sizeof(int);
			vertex_count = (unsigned int)tmp;
		}
		else if(count_name == "uint")
		{
			unsigned int tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(unsigned int));
			chunk_position += sizeof(unsigned int);
			vertex_count = (unsigned int)tmp;
		}
		else if(count_name == "float")
		{
			float tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(float));
			chunk_position += sizeof(float);
			vertex_count = tmp;
		}
		else if(count_name == "double")
		{
			double tmp;
			memcpy( (char*)&tmp, chunk_buffer, sizeof(double));
			chunk_position += sizeof(double);
			vertex_count = tmp;
		}

		cout << count_name << " " << vertex_count << endl;

		// Check if the face defintion is a triangle
		if(vertex_count != 3)
		{
			cout << "PLYIO::readFacesBinary(): Warning: Face defintion is not a triangle." << endl;
			// TO DO: Fix possible leak here.
			return;
		}

		// Now we have to copy the vertex indices into the index buffer
		if(value_name == "char")
		{
			char indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( &indices[j], chunk_position + j * sizeof(char), sizeof(char));
				m_indices[i * 3 + j] = (unsigned int) indices[j];
			}
		}
		if(value_name == "uchar")
		{
			unsigned char indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(unsigned char), sizeof(unsigned char));
				m_indices[i * 3 + j] = (unsigned int) indices[j];
			}
		}
		if(value_name == "short")
		{
			short indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(short), sizeof(short));
				m_indices[i * 3 + j] = (unsigned int) indices[j];
			}
		}
		if(value_name == "ushort")
		{
			unsigned short indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(unsigned short), sizeof(unsigned short));
				m_indices[i * 3 + j] = (unsigned int) indices[j];
			}
		}
		if(value_name == "int")
		{
			int indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(int), sizeof(int));
				m_indices[i * 3 + j] = (unsigned int) indices[j];
			}
		}
		if(value_name == "uint")
		{
			unsigned int indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(unsigned int), sizeof(unsigned int));
				m_indices[i * 3 + j] = indices[j];
				cout << m_indices[i * 3 + j] << endl;
			}
		}
		if(value_name == "float")
		{
			float indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(float), sizeof(float));
				m_indices[i * 3 + j] = indices[j];
			}
		}
		if(value_name == "double")
		{
			double indices[3];
			for(int j = 0; j < 3; j++)
			{
				memcpy( (char*) &indices[j], chunk_position + j * sizeof(double), sizeof(double));
				m_indices[i * 3 + j] = (unsigned int)indices[j];
			}
		}
	}

}

bool PLYIO::isSupported(string element_name)
{
	return element_name == "vertex" || element_name == "face";
}

void PLYIO::writeFacesBinary(ofstream &out, PLYElement* e)
{
	// TODO: Remember to change this value for different kind of
	// list properties.
	int N_VERTICES_PER_FACE = 3;

	vector<Property*>::iterator it, end, start;
	start = e->getFirstProperty();
	end = e->getLastProperty();

	// Determine buffer size
	size_t buffer_size = 0;
	for(it = start; it != end; it++)
	{

		size_t count_size = (*it)->getCountSize();
		cout << "Count SIze: " << (*it)->getCountSize() << endl;
		buffer_size += count_size;

		// Check for list properties.
		if(count_size != 0)
		{
			buffer_size += N_VERTICES_PER_FACE * (*it)->getValueSize();
			cout << "VALUE_SIZE: " << (*it)->getValueSize() << endl;
		}
		else
		{
			cout << "ELSE" << endl;
			buffer_size += (*it)->getValueSize();
		}
	}

	// Create buffer
	buffer_size = 13;
	char buffer[buffer_size];

	cout << buffer_size << endl;

	char* pos;

	// Write facee
	for(size_t i = 0; i < m_numberOfFaces; i++)
	{
		// Reset buffer
		memset(buffer, 0, buffer_size);
		pos = buffer;

		size_t index_pointer = i * 3;
		for(it = start; it != end; it++)
		{
			Property* p = (*it);
			string property_name = p->getName();
			string property_type = p->getElementTypeStr();

			if(property_name == "vertex_index" || property_name == "vertex_indices")
			{
				string count_type = p->getCountTypeStr();
				pos = putElementInBuffer(pos, count_type, N_VERTICES_PER_FACE);
				pos = putElementInBuffer(pos, property_type, m_indices[index_pointer]);
				pos = putElementInBuffer(pos, property_type, m_indices[index_pointer + 1]);
				pos = putElementInBuffer(pos, property_type, m_indices[index_pointer + 2]);
			}

		}


		// Write to output stream
		out.write(buffer, buffer_size);
	}

}

void PLYIO::addElement(PLYElement* e)
{
	m_elements.push_back(e);
}
