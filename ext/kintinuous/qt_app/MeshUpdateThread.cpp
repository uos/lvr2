/*
 * MeshUpdateThread.cpp
 *
 *  Created on: Jan 29, 2016
 *      Author: twiemann
 */

#include "MeshUpdateThread.hpp"

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

#include <QApplication>

MeshUpdateThread::MeshUpdateThread(kfusion::KinFu::Ptr kinfu)
{
	moveToThread(QApplication::instance()->thread());
	cout << "CREATE" << endl;
	m_kinfu = kinfu;
}

void MeshUpdateThread::computeMeshActor(HMesh* meshbuffer)
{
	static size_t verts_size = 0;
	static size_t faces_size = 0;
	cout << "0" << endl;

	m_vertices.clear();
	m_faces.clear();

    if(meshbuffer)
    {
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();

        // Parse vertex and index buffer

        size_t slice_size = meshbuffer->getVertices().size() - verts_size;
        size_t slice_face_size = meshbuffer->getFaces().size() - faces_size;

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");


        for(size_t k = 0; k < slice_size; k++)
        {
        	auto vertex = meshbuffer->getVertices()[k + verts_size];
        	m_indexMap[vertex] = k + verts_size;
        	points->InsertNextPoint(
        			vertex->m_position[0],
					vertex->m_position[1],
					vertex->m_position[2]);

        	//cout << vertex->m_position[0] << " " << vertex->m_position[1] << " " << vertex->m_position[2] << endl;

        	m_vertices.push_back(vertex->m_position[0]);
        	m_vertices.push_back(vertex->m_position[1]);
        	m_vertices.push_back(vertex->m_position[2]);

        	unsigned char color[3] = {0, 255, 0};
        	scalars->InsertNextTupleValue(color);

        }




        mesh->GetPointData()->SetScalars(scalars);
        cout << "2 " << slice_face_size << endl;
        for(size_t k = 0; k < slice_face_size; k++)
        {
        	auto face = meshbuffer->getFaces()[k + faces_size];

        	vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
        	t->GetPointIds()->SetId(0, m_indexMap[face->m_edge->end()]);
        	t->GetPointIds()->SetId(1, m_indexMap[face->m_edge->next()->end()]);
        	t->GetPointIds()->SetId(2, m_indexMap[face->m_edge->next()->next()->end()]);
        	triangles->InsertNextCell(t);

        	m_faces.push_back(m_indexMap[face->m_edge->end()]);
        	m_faces.push_back(m_indexMap[face->m_edge->next()->end()]);
        	m_faces.push_back(m_indexMap[face->m_edge->next()->next()->end()]);


        	int a = m_indexMap[face->m_edge->end()];
        	int b = m_indexMap[face->m_edge->next()->end()];
        	int c = m_indexMap[face->m_edge->next()->next()->end()];

        	cout << a << " " << b << " " << c << " " << m_indexMap.size() << " " << m_vertices.size() / 3 << endl;
//
//        	if(a >= m_indexMap.size()) cout << "A: " << a << " / " << m_indexMap.size() << endl;
//        	if(b >= m_indexMap.size()) cout << "B: " << b << " / " << m_indexMap.size() << endl;
//        	if(c >= m_indexMap.size()) cout << "C: " << c << " / " << m_indexMap.size() << endl;
      }

		verts_size = meshbuffer->getVertices().size();
		faces_size = meshbuffer->getFaces().size();

        mesh->SetPoints(points);
        mesh->SetPolys(triangles);


        vtkSmartPointer<vtkPolyDataMapper> mesh_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

#ifdef LVR2_USE_VTK5
        mesh_mapper->SetInput(mesh);
#else
        mesh_mapper->SetInputData(mesh);
#endif

        m_meshActor = vtkActor::New();
        m_meshActor->SetMapper(mesh_mapper);


        vtkSmartPointer<vtkProperty> p = vtkSmartPointer<vtkProperty>::New();
        p->SetColor(1.0, 1.0, 1.0);
        m_meshActor->SetProperty(p);
        m_meshActor->VisibilityOn();


//        vtkSmartPointer<vtkPolyDataMapper> wireframe_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//        // wireframe_mapper->SetInputData(mesh); VTK 6
//        wireframe_mapper->SetInput(mesh);
//        m_wireframeActor = vtkSmartPointer<vtkActor>::New();
//        m_wireframeActor->ShallowCopy(m_meshActor);
//        m_wireframeActor->SetMapper(wireframe_mapper);
//
//        vtkSmartPointer<vtkProperty> p = vtkSmartPointer<vtkProperty>::New();
//        p->DeepCopy(m_meshActor->GetProperty());
//        p->SetRepresentationToWireframe();
//        m_wireframeActor->SetProperty(p);
//
//        float r = 0.0;
//        float g = 0.9;
//        float b = 0.0;
//
//        p->DeepCopy(m_meshActor->GetProperty());
//        p->SetRepresentationToWireframe();
//        p->SetColor(r, g, b);
//        m_meshActor->SetProperty(p);
//
//        p = m_wireframeActor->GetProperty();
//        float inv_r = (float)1 - r;
//        float inv_g = (float)1 - g;
//        float inv_b = (float)1 - b;
//        p->SetColor(inv_r, inv_g, inv_b);
//        m_wireframeActor->SetProperty(p);
    }
}

void MeshUpdateThread::run()
{
	while(true)
	{
	  //auto b = m_kinfu->cyclical().getMesh();
		//computeMeshActor(b);
		//Q_EMIT(meshUpdate(m_meshActor));
	}
}

MeshUpdateThread::~MeshUpdateThread()
{
	// TODO Auto-generated destructor stub
}


