/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/**
 * LVRMeshBufferBridge.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRMeshBufferBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkTriangle.h>
#include <vtkProperty.h>
#include <vtkImageData.h>
#include <vtkTexture.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

namespace lvr
{

LVRMeshBufferBridge::LVRMeshBufferBridge(MeshBufferPtr meshBuffer) :
        m_meshBuffer(meshBuffer)
{
    if(meshBuffer)
    {
        computeMeshActor(meshBuffer);
        meshBuffer->getVertexArray(m_numVertices);
        meshBuffer->getFaceArray(m_numFaces);
    }
    else
    {
        m_numFaces = 0;
        m_numVertices = 0;
    }

    m_numColoredFaces 	= 0;
    m_numTexturedFaces 	= 0;
    m_numTextures		= 0;
}

void LVRMeshBufferBridge::setBaseColor(float r, float g, float b)
{
	vtkSmartPointer<vtkProperty> p = m_meshActor->GetProperty();
    p->SetColor(r, g, b);
    m_meshActor->SetProperty(p);

    p = m_wireframeActor->GetProperty();
    float inv_r = (float)1 - r;
    float inv_g = (float)1 - g;
    float inv_b = (float)1 - b;
    p->SetColor(inv_r, inv_g, inv_b);
    m_wireframeActor->SetProperty(p);
}

LVRMeshBufferBridge::LVRMeshBufferBridge(const LVRMeshBufferBridge& b)
{
    m_numVertices   = b.m_numVertices;
    m_numFaces      = b.m_numFaces;
    m_meshActor     = b.m_meshActor;
}

size_t	LVRMeshBufferBridge::getNumColoredFaces()
{
	return m_numColoredFaces;
}

size_t	LVRMeshBufferBridge::getNumTexturedFaces()
{
	return m_numTexturedFaces;
}
size_t	LVRMeshBufferBridge::getNumTextures()
{
	return m_numTextures;
}

size_t LVRMeshBufferBridge::getNumTriangles()
{
    return m_numFaces;
}

size_t LVRMeshBufferBridge::getNumVertices()
{
    return m_numVertices;
}

MeshBufferPtr  LVRMeshBufferBridge::getMeshBuffer()
{
    return m_meshBuffer;
}

LVRMeshBufferBridge::~LVRMeshBufferBridge()
{
    // TODO Auto-generated destructor stub
}

void LVRMeshBufferBridge::computeMeshActor(MeshBufferPtr meshbuffer)
{
    if(meshbuffer)
    {
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();

        // Parse vertex and index buffer
        size_t n_v, n_i;
        floatArr vertices = meshbuffer->getVertexArray(n_v);
        uintArr indices = meshbuffer->getFaceArray(n_i);

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

        for(size_t i = 0; i < n_v; i++){
            size_t index = 3 * i;
            points->InsertNextPoint(
                    vertices[index    ],
                    vertices[index + 1],
                    vertices[index + 2]);
        }

        for(size_t i = 0; i < n_i; i++)
        {
            size_t index = 3 * i;
            vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
            t->GetPointIds()->SetId(0, indices[index]);
            t->GetPointIds()->SetId(1, indices[index + 1]);
            t->GetPointIds()->SetId(2, indices[index + 2]);
            triangles->InsertNextCell(t);
        }

        mesh->SetPoints(points);
        mesh->SetPolys(triangles);

        vtkSmartPointer<vtkPolyDataMapper> mesh_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mesh_mapper->SetInput(mesh);
        m_meshActor = vtkSmartPointer<vtkActor>::New();
        m_meshActor->SetMapper(mesh_mapper);

        vtkSmartPointer<vtkPolyDataMapper> wireframe_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        wireframe_mapper->SetInput(mesh);
        m_wireframeActor = vtkSmartPointer<vtkActor>::New();
        m_wireframeActor->ShallowCopy(m_meshActor);
        m_wireframeActor->SetMapper(wireframe_mapper);
        vtkSmartPointer<vtkProperty> p = vtkSmartPointer<vtkProperty>::New();
        p->DeepCopy(m_meshActor->GetProperty());
        p->SetRepresentationToWireframe();
        m_wireframeActor->SetProperty(p);

        setBaseColor(0.9, 0.9, 0.9);
    }
}

void LVRMeshBufferBridge::setOpacity(float opacityValue)
{
	vtkSmartPointer<vtkProperty> p = m_meshActor->GetProperty();
    p->SetOpacity(opacityValue);
    m_meshActor->SetProperty(p);
}

void LVRMeshBufferBridge::setVisibility(bool visible)
{
    if(visible)
    {
        m_meshActor->VisibilityOn();
        m_wireframeActor->VisibilityOn();
    }
    else
    {
        m_meshActor->VisibilityOff();
        m_wireframeActor->VisibilityOff();
    }
}

void LVRMeshBufferBridge::setShading(int shader)
{
    vtkSmartPointer<vtkProperty> p = m_meshActor->GetProperty();
    p->SetShading(shader);
    m_meshActor->SetProperty(p);
}

vtkSmartPointer<vtkActor> LVRMeshBufferBridge::getWireframeActor()
{
    return m_wireframeActor;
}

vtkSmartPointer<vtkActor> LVRMeshBufferBridge::getMeshActor()
{
    return m_meshActor;
}

void LVRMeshBufferBridge::computeMaterialGroups(vector<MaterialGroup*>& textureMaterials, vector<MaterialGroup*>& colorMaterials)
{
	map<int, MaterialGroup* > texMatMap;
	map<Vertex<unsigned char>, MaterialGroup* > colorMatMap;

	// Get buffers
	materialArr	materials;
	floatArr	textureCoords;
	textureArr	textures;
	uintArr		faceMaterials;

	size_t numMaterials;
	materials = m_meshBuffer->getMaterialArray(numMaterials);

	size_t numFaceMaterials;
	faceMaterials = m_meshBuffer->getFaceMaterialIndexArray(numFaceMaterials);

	// Iterate over face material buffer and
	// sort faces by their material
	for(size_t i = 0; i < m_numFaces; i++)
	{
		map<int, MaterialGroup*>::iterator texIt;
		map<Vertex<unsigned char>, MaterialGroup* >::iterator colIt;

		// Get material by index and lookup in map. If present
		// add face index to the corresponding group. Create a new
		// group if none was found. For efficient rendering we have to
		// create groups by color and texture index,
		Material* m = materials[faceMaterials[i]];

		if(m->texture_index != -1)
		{

			texIt = texMatMap.find(m->texture_index);
			if(texIt == texMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = m->texture_index;
				g->color = Vertex<float>(1.0, 1.0, 1.0);
				g->faceBuffer.push_back(i);
				textureMaterials.push_back(g);
				texMatMap[m->texture_index] = g;
			}
			else
			{
				texIt->second->faceBuffer.push_back(i);
			}
		}
		else
		{
			colIt = colorMatMap.find(Vertex<unsigned char>(m->r, m->g, m->b));
			if(colIt == colorMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = m->texture_index;
				g->faceBuffer.push_back(i);
				g->color = Vertex<float>(m->r / 255.0f, m->g / 255.0f, m->b / 255.0f);
				colorMaterials.push_back(g);
			}
			else
			{
				colIt->second->faceBuffer.push_back(i);
			}
		}
	}
}



void LVRMeshBufferBridge::remapTexturedIndices(
		MaterialGroup* g,
		vector<Vertex<float> >& vertices,
		vector<Vertex<float> >& texCoords,
		vector<int>& indices)
{
	// Mapping stuff
	map<int, int> indexMap;
	map<int, int>::iterator it;
	int globalIndex = 0;

	// Get vertex buffer
	size_t n;
	size_t nc;
	floatArr vertexBuffer = m_meshBuffer->getVertexArray(n);
	uintArr	 faceBuffer = m_meshBuffer->getFaceArray(n);
	floatArr textures = m_meshBuffer->getVertexTextureCoordinateArray(nc);

	// Iterate through the index buffer that references global indices
	// and create new vertex and index buffer using local indices
	for(size_t i = 0; i < g->faceBuffer.size(); i++)
	{
		int a = faceBuffer[g->faceBuffer[i] * 3    ];
		int b = faceBuffer[g->faceBuffer[i] * 3 + 1];
		int c = faceBuffer[g->faceBuffer[i] * 3 + 2];


		// Lookup a *******************************************************************************
		it = indexMap.find(a);
		if(it == indexMap.end())
		{
			indexMap[a] = globalIndex;
			indices.push_back(globalIndex);
			Vertex<float> v(vertexBuffer[3 * a], vertexBuffer[3 * a + 1], vertexBuffer[3 * a + 2]);
			Vertex<float> t(textures[3 * a], textures[3 * a + 1], textures[3 * a + 2]);
			texCoords.push_back(t);
			vertices.push_back(v);
			globalIndex++;
		}
		else
		{
			indices.push_back(indexMap[a]);
		}

		// Lookup b *******************************************************************************
		it = indexMap.find(b);
		if(it == indexMap.end())
		{
			indexMap[b] = globalIndex;
			indices.push_back(globalIndex);
			Vertex<float> v(vertexBuffer[3 * b], vertexBuffer[3 * b + 1], vertexBuffer[3 * b + 2]);
			Vertex<float> t(textures[3 * b], textures[3 * b + 1], textures[3 * b + 2]);
			texCoords.push_back(t);
			vertices.push_back(v);
			globalIndex++;
		}
		else
		{
			indices.push_back(indexMap[b]);
		}

		// Lookup c *******************************************************************************
		it = indexMap.find(c);
		if(it == indexMap.end())
		{
			indexMap[c] = globalIndex;
			indices.push_back(globalIndex);
			Vertex<float> v(vertexBuffer[3 * c], vertexBuffer[3 * c + 1], vertexBuffer[3 * c + 2]);
			Vertex<float> t(textures[3 * c], textures[3 * c + 1], textures[3 * c + 2]);
			texCoords.push_back(t);
			vertices.push_back(v);
			globalIndex++;
		}
		else
		{
			indices.push_back(indexMap[c]);
		}

	}
}

void LVRMeshBufferBridge::remapIndices(
		vector<MaterialGroup*> groups,
		vector<Vertex<float> >& vertices,
		vector<Vertex<unsigned char> >& colors,
		vector<int>& indices)
{
	int globalIndex = 0;
	map<int, int> indexMap;
	map<int, int>::iterator it;

	size_t n;
	floatArr vertexBuffer = m_meshBuffer->getVertexArray(n);
	uintArr	 faceBuffer = m_meshBuffer->getFaceArray(n);

	for(size_t a = 0; a < groups.size(); a++)
	{
		MaterialGroup* g = groups[a];

		for(size_t i = 0; i < g->faceBuffer.size(); i++)
		{
			int a = faceBuffer[g->faceBuffer[i] * 3    ];
			int b = faceBuffer[g->faceBuffer[i] * 3 + 1];
			int c = faceBuffer[g->faceBuffer[i] * 3 + 2];

			colors.push_back(
					Vertex<unsigned char>(
							(unsigned char) (255*g->color[0]),
							(unsigned char) (255*g->color[1]),
							(unsigned char) (255*g->color[2])
					));

			// Lookup a *******************************************************************************
			it = indexMap.find(a);
			if(it == indexMap.end())
			{
				indexMap[a] = globalIndex;
				indices.push_back(globalIndex);
				Vertex<float> v(vertexBuffer[3 * a], vertexBuffer[3 * a + 1], vertexBuffer[3 * a + 2]);
				vertices.push_back(v);
				globalIndex++;
			}
			else
			{
				indices.push_back(indexMap[a]);
			}

			// Lookup b *******************************************************************************
			it = indexMap.find(b);
			if(it == indexMap.end())
			{
				indexMap[b] = globalIndex;
				indices.push_back(globalIndex);
				Vertex<float> v(vertexBuffer[3 * b], vertexBuffer[3 * b + 1], vertexBuffer[3 * b + 2]);
				vertices.push_back(v);
				globalIndex++;
			}
			else
			{
				indices.push_back(indexMap[b]);
			}

			// Lookup c *******************************************************************************
			it = indexMap.find(c);
			if(it == indexMap.end())
			{
				indexMap[c] = globalIndex;
				indices.push_back(globalIndex);
				Vertex<float> v(vertexBuffer[3 * c], vertexBuffer[3 * c + 1], vertexBuffer[3 * c + 2]);
				vertices.push_back(v);
				globalIndex++;
			}
			else
			{
				indices.push_back(indexMap[c]);
			}

		}

	}
}


vtkSmartPointer<vtkActor> LVRMeshBufferBridge::getTexturedActor(MaterialGroup* g)
{
	vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();

	// Remap global indices for new actor
	vector<Vertex<float> > vertices;
	vector<Vertex<float> > texCoords;
	vector<int> indices;
	remapTexturedIndices(g, vertices, texCoords, indices);

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkFloatArray> tc = vtkSmartPointer<vtkFloatArray>::New();
	tc->SetNumberOfComponents(3);
	tc->SetName("TextureCoordinates");

	// Insert used vertices and texture coordinates into new arrays
    for(size_t i = 0; i < vertices.size(); i++){
        points->InsertNextPoint(
                vertices[i][0],
                vertices[i][1],
                vertices[i][2]);
         tc->InsertNextTuple3(texCoords[i][1], texCoords[i][0], 0.0);

    }

    // Add triangles
    for(size_t i = 0; i < indices.size() / 3; i++)
    {
        size_t index = 3 * i;
        vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
        t->GetPointIds()->SetId(0, indices[index]);
        t->GetPointIds()->SetId(1, indices[index + 1]);
        t->GetPointIds()->SetId(2, indices[index + 2]);
        triangles->InsertNextCell(t);
        m_numTexturedFaces++;
    }

    // Build polydata
	mesh->SetPoints(points);
	mesh->SetPolys(triangles);
	mesh->GetPointData()->SetTCoords(tc);


	// Generate actor
	vtkSmartPointer<vtkPolyDataMapper> mesh_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mesh_mapper->SetInput(mesh);
	actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mesh_mapper);
	actor->SetTexture(getTexture(g->textureIndex));

	// Set color properties
	vtkSmartPointer<vtkProperty> property = vtkSmartPointer<vtkProperty>::New();
	property->SetColor(g->color[0], g->color[1], g->color[2]);
	actor->SetProperty(property);

	return actor;
}

vtkSmartPointer<vtkTexture> LVRMeshBufferBridge::getTexture(int index)
{
	size_t n;
	textureArr textures = m_meshBuffer->getTextureArray(n);
	GlTexture* tex = textures[index];
	int w = tex->m_width;
	int h = tex->m_height;
	unsigned char* texData = tex->m_pixels;

	vtkSmartPointer<vtkImageData> data = vtkSmartPointer<vtkImageData>::New();
	data->SetDimensions(h, w, 1);
	data->SetNumberOfScalarComponents(3);
	data->SetScalarTypeToUnsignedChar();
	data->AllocateScalars();
	int c = 0;
	for(int i = 0; i < h; i++)
	{
		for(int j = 0; j < w; j++)
		{
			unsigned char* pixel = static_cast<unsigned char*>(data->GetScalarPointer(i,j,0));
			pixel[0] = texData[3 * c];
			pixel[1] = texData[3 * c + 1];
			pixel[2] = texData[3 * c + 2];
			c++;
		}
	}
	data->Modified();

	vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
	texture->SetInputConnection(data->GetProducerPort());

	return texture;
}


vtkSmartPointer<vtkActorCollection> LVRMeshBufferBridge::getTexturedActors()
{
	size_t numTextures;
	size_t numFaceMaterials;

	vector<MaterialGroup*> textureGroups;
	vector<MaterialGroup*> colorGroups;
	computeMaterialGroups(textureGroups, colorGroups);

	m_numTextures = textureGroups.size();

	vtkSmartPointer<vtkActorCollection> collection = vtkSmartPointer<vtkActorCollection>::New();
	for(size_t i = 0; i < textureGroups.size(); i++)
	{
		//cout << i <<  " / " << textureGroups.size() << endl;
		vtkSmartPointer<vtkActor> a = getTexturedActor(textureGroups[i]);
		collection->AddItem(a);
	}

	vtkSmartPointer<vtkActor> a = getColorMeshActor(colorGroups);
	collection->AddItem(a);

	return collection;
}

vtkSmartPointer<vtkActor> LVRMeshBufferBridge::getColorMeshActor(vector<MaterialGroup*> groups)
{
	vector<Vertex<float> > vertices;
	vector<int> indices;
	vector<Vertex<unsigned char> > colors;

	vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();

	remapIndices(groups, vertices, colors, indices);

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

	vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
	scalars->SetNumberOfComponents(3);
	scalars->SetName("Colors");

	// Insert used vertices and texture coordinates into new arrays
    for(size_t i = 0; i < vertices.size(); i++){
        points->InsertNextPoint(
                vertices[i][0],
                vertices[i][1],
                vertices[i][2]);
    }

    // Add triangles
    for(size_t i = 0; i < indices.size() / 3; i++)
    {
    	unsigned char color[3];
    	color[0] = colors[i][0];
    	color[1] = colors[i][1];
    	color[2] = colors[i][2];

    	scalars->InsertNextTupleValue(color);

        size_t index = 3 * i;
        vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
        t->GetPointIds()->SetId(0, indices[index]);
        t->GetPointIds()->SetId(1, indices[index + 1]);
        t->GetPointIds()->SetId(2, indices[index + 2]);
        triangles->InsertNextCell(t);
        m_numColoredFaces++;
    }


    // Build polydata
	mesh->SetPoints(points);
	mesh->SetPolys(triangles);
	mesh->GetCellData()->SetScalars(scalars);


	// Generate actor
	vtkSmartPointer<vtkPolyDataMapper> mesh_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mesh_mapper->SetInput(mesh);
	actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mesh_mapper);

	return actor;
}

bool LVRMeshBufferBridge::hasTextures()
{
	size_t num;
	m_meshBuffer->getTextureArray(num);
	return (num > 0);
}


} /* namespace lvr */
