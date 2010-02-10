#include "PLYWriter.h"
#include <cstring>

PLYIO::PLYIO()
{
	m_vertices 	= 0;
	m_normals	= 0;
	m_colors	= 0;
	m_indices	= 0;
	m_binary	= 0;
}


PLYIO::~PLYIO()
{

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
		memcpy(buffer, &s, sizeof(s));
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

bool PLYIO::isSupported(string element_name)
{
	return true;
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
		buffer_size += count_size;

		// Check for list properties.
		if(count_size != 0)
		{
			buffer_size += N_VERTICES_PER_FACE * (*it)->getValueSize();
		}
		else
		{
			buffer_size += (*it)->getValueSize();
		}
	}

	// Create buffer
	char buffer[buffer_size];

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

			if(property_name == "vertex_index")
			{
				cout << property_type << endl;
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
