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

#include <lvr2/util/Util.hpp>

namespace lvr2
{

LVRMeshBufferBridge::LVRMeshBufferBridge(MeshBuffer2Ptr meshBuffer) :
        m_meshBuffer(meshBuffer)
{
    if(meshBuffer)
    {
        computeMeshActor(meshBuffer);
        m_numVertices = meshBuffer->numVertices();
        m_numFaces = meshBuffer->numFaces();
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

MeshBuffer2Ptr  LVRMeshBufferBridge::getMeshBuffer()
{
    return m_meshBuffer;
}

LVRMeshBufferBridge::~LVRMeshBufferBridge()
{
    // TODO Auto-generated destructor stub
}

void LVRMeshBufferBridge::computeMeshActor(MeshBuffer2Ptr meshbuffer)
{
    if(meshbuffer)
    {
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();

        // Parse vertex and index buffer
        size_t n_v, n_i, n_c;
        unsigned w_color;
        n_v = meshbuffer->numVertices();
        floatArr vertices = meshbuffer->getVertices();
        n_i = meshbuffer->numFaces();
        indexArray indices = meshbuffer->getFaceIndices();
        n_c = n_v;
        ucharArr colors = meshbuffer->getVertexColors(w_color);

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");

        for(size_t i = 0; i < n_v; i++){
            size_t index = 3 * i;
            points->InsertNextPoint(
                    vertices[index    ],
                    vertices[index + 1],
                    vertices[index + 2]);

            if(colors)
        	{
                size_t color_index = w_color * i;
        		unsigned char color[3];
                color[0] = colors[color_index    ];
                color[1] = colors[color_index + 1];
                color[2] = colors[color_index + 2];
#if VTK_MAJOR_VERSION < 7      
        		scalars->InsertNextTupleValue(color);
#else
			scalars->InsertNextTypedTuple(color);
#endif
        	}
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

        if(colors)
        {
        	mesh->GetPointData()->SetScalars(scalars);
        }

        vtkSmartPointer<vtkPolyDataMapper> mesh_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#ifdef LVR_USE_VTK5
        mesh_mapper->SetInput(mesh);
#else
        mesh_mapper->SetInputData(mesh);
#endif
        m_meshActor = vtkSmartPointer<vtkActor>::New();
        m_meshActor->SetMapper(mesh_mapper);

        vtkSmartPointer<vtkPolyDataMapper> wireframe_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#ifdef LVR_USE_VTK5
        wireframe_mapper->SetInput(mesh);
#else
        wireframe_mapper->SetInputData(mesh);
#endif
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
	map<int, MaterialGroup*> texMatMap;
	map<Vector<VecUChar>, MaterialGroup*, Util::ColorVecCompare> colorMatMap;

	// Get buffers
	vector<Material>  &materials = m_meshBuffer->getMaterials();
	vector<Texture>   &textures = m_meshBuffer->getTextures();
	floatArr	      textureCoords;
	indexArray	      faceMaterials;

	size_t numMaterials = materials.size();

	size_t numFaceMaterials = m_meshBuffer->numFaces();
	faceMaterials = m_meshBuffer->getFaceMaterialIndices();

	// Iterate over face material buffer and
	// sort faces by their material
	for(size_t i = 0; i < m_numFaces; i++)
	{
		map<int, MaterialGroup*>::iterator texIt;
		map<Vector<VecUChar>, MaterialGroup*, Util::ColorVecCompare>::iterator colIt;

		// Get material by index and lookup in map. If present
		// add face index to the corresponding group. Create a new
		// group if none was found. For efficient rendering we have to
		// create groups by color and texture index,
        const Material &m = materials[faceMaterials[i]];

		if(m.m_texture)
		{

			texIt = texMatMap.find(m.m_texture->idx());
			if(texIt == texMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = m.m_texture->idx();
				g->color = Vector<Vec>(1.0, 1.0, 1.0);
				g->faceBuffer.push_back(i);
				textureMaterials.push_back(g);
				texMatMap[m.m_texture->idx()] = g;
			}
			else
			{
				texIt->second->faceBuffer.push_back(i);
			}
		}
		else
		{
			colIt = colorMatMap.find(Vector<VecUChar>(m.m_color->at(0),m.m_color->at(1), m.m_color->at(2)));
			if(colIt == colorMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = -1;
				g->faceBuffer.push_back(i);
				g->color = Vector<Vec>(m.m_color->at(0) / 255.0f, m.m_color->at(1) / 255.0f, m.m_color->at(2) / 255.0f);
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
		vector<Vector<Vec> >& vertices,
		vector<Vector<Vec> >& texCoords,
		vector<int>& indices)
{
	// Mapping stuff
	map<int, int> indexMap;
	map<int, int>::iterator it;
	int globalIndex = 0;

	// Get vertex buffer
	size_t n = m_meshBuffer->numVertices();
    size_t n_i = m_meshBuffer->numFaces();
	size_t nc = n;
	floatArr vertexBuffer = m_meshBuffer->getVertices();
	indexArray faceBuffer = m_meshBuffer->getFaceIndices();
	floatArr textures     = m_meshBuffer->getTextureCoordinates();

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
			Vector<Vec> v(vertexBuffer[3 * a], vertexBuffer[3 * a + 1], vertexBuffer[3 * a + 2]);
			Vector<Vec> t(textures[2 * a], textures[2 * a + 1], 0.0);
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
			Vector<Vec> v(vertexBuffer[3 * b], vertexBuffer[3 * b + 1], vertexBuffer[3 * b + 2]);
			Vector<Vec> t(textures[2 * b], textures[2 * b + 1], 0.0);
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
			Vector<Vec> v(vertexBuffer[3 * c], vertexBuffer[3 * c + 1], vertexBuffer[3 * c + 2]);
			Vector<Vec> t(textures[2 * c], textures[2 * c + 1], 0.0);
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
		vector<Vector<Vec> >& vertices,
		vector<Vector<VecUChar> >& colors,
		vector<int>& indices)
{
	int globalIndex = 0;
	map<int, int> indexMap;
	map<int, int>::iterator it;

	size_t n   = m_meshBuffer->numVertices();
    size_t n_i = m_meshBuffer->numFaces();
	floatArr vertexBuffer = m_meshBuffer->getVertices();
	uintArr	 faceBuffer   = m_meshBuffer->getFaceIndices();

	for(size_t a = 0; a < groups.size(); a++)
	{
		MaterialGroup* g = groups[a];

		for(size_t i = 0; i < g->faceBuffer.size(); i++)
		{
			int a = faceBuffer[g->faceBuffer[i] * 3    ];
			int b = faceBuffer[g->faceBuffer[i] * 3 + 1];
			int c = faceBuffer[g->faceBuffer[i] * 3 + 2];

			colors.push_back(
					Vector<VecUChar>(
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
				Vector<Vec> v(vertexBuffer[3 * a], vertexBuffer[3 * a + 1], vertexBuffer[3 * a + 2]);
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
				Vector<Vec> v(vertexBuffer[3 * b], vertexBuffer[3 * b + 1], vertexBuffer[3 * b + 2]);
				vertices.push_back(v);
				globalIndex++;
			}
			else
			{
				indices.push_back(indexMap[b]); }

			// Lookup c *******************************************************************************
			it = indexMap.find(c);
			if(it == indexMap.end())
			{
				indexMap[c] = globalIndex;
				indices.push_back(globalIndex);
				Vector<Vec> v(vertexBuffer[3 * c], vertexBuffer[3 * c + 1], vertexBuffer[3 * c + 2]);
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
	vector<Vector<Vec> > vertices;
	vector<Vector<Vec> > texCoords;
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
#ifdef LVR_USE_VTK5
	mesh_mapper->SetInput(mesh);
#else
	mesh_mapper->SetInputData(mesh);
#endif
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
	vector<Texture> &textures = m_meshBuffer->getTextures();
	size_t n = textures.size();
	Texture &tex = textures[index];
	int w = tex.m_width;
	int h = tex.m_height;
    unsigned char numChannels = tex.m_numChannels;
    unsigned char numBytesPerChan = tex.m_numBytesPerChan;
	unsigned char* texData = tex.m_data;

	vtkSmartPointer<vtkImageData> data = vtkSmartPointer<vtkImageData>::New();
	data->SetDimensions(h, w, 1);
#ifdef LVR_USE_VTK5
	data->SetNumberOfScalarComponents(3);
	data->SetScalarTypeToUnsignedChar();
	data->AllocateScalars();
#else
	data->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
#endif
	int c = 0;
	for(int i = 0; i < h; i++)
	{
		for(int j = 0; j < w; j++)
		{
			unsigned char* pixel = static_cast<unsigned char*>(data->GetScalarPointer(i,j,0));
            size_t index = numChannels * numBytesPerChan * c;
			pixel[0] = texData[index                      ];
			pixel[1] = texData[index + 1 * numBytesPerChan];
			pixel[2] = texData[index + 2 * numBytesPerChan];
			c++;
		}
	}
	data->Modified();

	vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
#ifdef LVR_USE_VTK5
	texture->SetInput(data);
#else
	texture->SetInputData(data);
#endif

	return texture;
}


vtkSmartPointer<vtkActorCollection> LVRMeshBufferBridge::getTexturedActors()
{
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

    for (auto m : textureGroups)
    {
        delete m;
    }

    for (auto m : colorGroups)
    {
        delete m;
    }

	return collection;
}

vtkSmartPointer<vtkActor> LVRMeshBufferBridge::getColorMeshActor(vector<MaterialGroup*> groups)
{
	vector<Vector<Vec> > vertices;
	vector<int> indices;
	vector<Vector<VecUChar> > colors;

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
#if VTK_MAJOR_VERSION < 7
    	scalars->InsertNextTupleValue(color);
#else
	scalars->InsertNextTypedTuple(color);
#endif

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
#ifdef LVR_USE_VTK5
	mesh_mapper->SetInput(mesh);
#else
	mesh_mapper->SetInputData(mesh);
#endif
	actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mesh_mapper);

	return actor;
}

bool LVRMeshBufferBridge::hasTextures()
{
	return m_meshBuffer->getTextures().size() > 0;
}

} /* namespace lvr2 */
