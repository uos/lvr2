/*
 * Model3dDataCollector.cpp
 *
 *  Created on: 14.10.2010
 *      Author: Thomas Wiemann
 */

#include "Model3dDataCollector.h"

#include <iostream>
#include <lib3d/PolygonMesh.h>

using std::cout;
using std::endl;

Model3dDataCollector::Model3dDataCollector(ClientProxy* proxy, DataManager* manager)
	: DataCollector(proxy, manager)
{
	float scale = 1.0f;
	Model3dProxy* model_proxy = static_cast<Model3dProxy*>(m_proxy);
	if(model_proxy != 0)
	{
		// Request data and get number of polygons
		model_proxy->RequestData();
		int num_polygons = model_proxy->GetGeomCount();

		// Create a new polygon mesh
		PolygonMesh* mesh = new PolygonMesh;

		for(int i = 0; i < num_polygons; i++)
		{
	        Polygon p;
	        player_geom_entity_t geom = model_proxy->GetGeom(i);
	        p.color_r = 0.0;//((double)geom.color.red)/256.0;
	        p.color_g = 0.0;//((double)geom.color.green)/256.0,
	        p.color_b = 1.0;//((double)geom.color.blue)/256.0,
	        p.color_alpha = 0.5;//((double)geom.color.alpha)/256.0;

	        // Here we have to convert from Player's coordinate system
	        // the the OpenGL coodinate system y = z, z = -y,
	        for (size_t j = 0; j < geom.points_count; ++j) {
	            p.addVertex(geom.points[j].px * scale, geom.points[j].pz * scale, -geom.points[j].py * scale);
	        }
	        mesh->addPolygon(p);
		}
		mesh->sort();
		m_renderable = mesh;
	}
	else
	{
		cout << "Model3dDataCollector: Zero pointer to proxy." << endl;
	}
}

Model3dDataCollector::~Model3dDataCollector()
{
	// TODO Auto-generated destructor stub
}

ViewerType Model3dDataCollector::supportedViewerType()
{
	return PERSPECTIVE_VIEWER;
}
