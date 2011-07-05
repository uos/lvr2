#include "BaseMesh.hpp"

namespace lssr
{

template<typename VertexT, typename IndexType>
void BaseMesh<VertexT, IndexType>::save(string filename)
{
	PLYIO ply_writer;

	// Create element descriptions
	PLYElement* vertex_element = new PLYElement("vertex", m_nVertices);
	vertex_element->addProperty("x", "float");
	vertex_element->addProperty("y", "float");
	vertex_element->addProperty("z", "float");
	if(this->m_colorBuffer != 0)
	{
		vertex_element->addProperty("r", "float");
		vertex_element->addProperty("g", "float");
		vertex_element->addProperty("b", "float");
	}


	PLYElement* face_element = new PLYElement("face",m_nFaces);
	face_element->addProperty("vertex_indices", "uint", "uchar");


	// Add elements descriptions to header
	ply_writer.addElement(vertex_element);
	ply_writer.addElement(face_element);

	// Set data arrays
	ply_writer.setVertexArray(this->m_vertexBuffer, m_nVertices);
	ply_writer.setIndexArray(this->m_indexBuffer, m_nFaces);
	if(this->m_colorBuffer != 0)
	{
		ply_writer.setColorArray(this->m_colorBuffer, this->m_nVertices);
	}

	// Save
	ply_writer.save(filename, true);
}

}
