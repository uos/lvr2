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

/*
 * FusionMesh.tcc
 *
 *  @date 13.11.2008
 *	@author Henning Deeken (hdeeken@uos.de)
 *	@author Ann-Katrin Häuser (ahaeuser@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

namespace lvr
{

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
	
///
/// Mesh Construction Methods
///

template<typename VertexT, typename NormalT> FusionMesh<VertexT, NormalT>::FusionMesh()
{
   clearLocalBuffer();
   clearGlobalBuffer();
}

template<typename VertexT, typename NormalT> FusionMesh<VertexT, NormalT>::FusionMesh(MeshBufferPtr mesh)
{
   clearLocalBuffer();
   clearGlobalBuffer();
   addMesh(mesh);
   integrate();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addVertex(VertexT v)
{
    // Create new FusionMeshVertex and increase vertex counter
    m_local_vertices.push_back(new FusionVertex<VertexT, NormalT>(v));    
    m_local_index++;
    //cout << "Adding Vertex - " << v << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addNormal(NormalT n)
{
    // Is a vertex exists at globalIndex, save normal
    assert(m_local_index == m_local_vertices.size());
    m_local_vertices[m_local_index - 1]->m_normal = n;
    
    //cout << "Adding Normal - " << n << endl;  
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c, FFace* &face)
{		
    // Create a new face
    face = new FFace;

	face->m_index[0] = a;
	face->m_index[1] = b;
	face->m_index[2] = c;
	
	FVertex* v0 = m_local_vertices[a];
	FVertex* v1 = m_local_vertices[b];
	FVertex* v2 = m_local_vertices[c];
	
	face->vertices[0] = v0;
	face->vertices[1] = v1;
	face->vertices[2] = v2;
   
    m_local_faces.push_back(face);
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c)
{
	  //cout << "Adding Triangle..." << a << " " << b << " " << c << endl;
      FFace* face;
      addTriangle(a, b, c, face);
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMesh(MeshBufferPtr mesh)
{
    size_t num_verts, num_norms, num_faces;
    floatArr vertices = mesh->getVertexArray(num_verts);
    floatArr normals = mesh->getVertexNormalArray(num_norms);
    
    //if(num_norms != num_verts)
	//	cout << "Unequal number of vertices and normals" << endl;
    
    // Add all vertices
    for(size_t i = 0; i < num_verts; i++)
    {
        addVertex(VertexT(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]));
		
    }
    
    // Add all faces
    uintArr faces = mesh->getFaceArray(num_faces);
    for(size_t i = 0; i < num_faces; i++)
    {
        addTriangle(faces[3 * i], faces[3 * i + 1], faces[3 * i + 2]);
    }
    
      // Add all vertex normals, in case we need that.
    /*
    
    for(size_t i = 0; i < num_norms; i++)
    {
         addNormal(NormalT(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]));
		
    } 
    
    */ 

}

///
/// Integration Methods
///

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::clearLocalBuffer()
{
	m_local_index = 0;
	m_local_vertices.clear();	
	m_local_faces.clear();

}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::clearGlobalBuffer()
{
	m_global_index = 0;
	m_global_vertices.clear();	
	m_global_faces.clear();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printLocalBufferStatus()
{
	cout << timestamp << "Local Buffer" << endl;
	cout << timestamp << "#Index         :" << m_local_index <<  endl;
	cout << timestamp << "#Vertices      :" << m_local_vertices.size() <<  endl;
	cout << timestamp << "#Faces         :" << m_local_faces.size() << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printGlobalBufferStatus()
{
	cout << timestamp << "Global Buffer" << endl;
	cout << timestamp << "#Index         :" << m_global_index <<  endl;
	cout << timestamp << "#Vertices      :" << m_global_vertices.size() <<  endl;
	cout << timestamp << "#Faces         :" << m_global_faces.size() << endl; 
}

// sloppy variant with possibly redundant vertices, to add properly make set checkup
template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobalVertex(FVertex *v)
{
    // Create new FusionMeshVertex and increase vertex counter
    //v->m_self_index = m_global_index;
    //cout << "at insertion " << v->m_self_index << "-" << v->m_position << endl;
    
    m_global_vertices.push_back(v);
    
   // HIER wird auch der m_self_index des bereits vorhandenen Vertex im Global Buffer ersetzt
    m_global_vertices[m_global_index]->m_self_index = m_global_index;
   // cout << "after insertion " << v->m_self_index << "-" << v->m_position << endl;
  
   // cout << "in globale" <<  m_global_vertices[m_global_index]->m_self_index << "-" <<  m_global_vertices[m_global_index]->m_position << endl << endl;
  
    
    m_global_index++; // = m_global_vertices.size() - 1;   
	
	
   // cout << "Adding Global Vertex at global buffer position " << m_global_index <<  endl;
   // cout << "m_Self_index " << v->m_self_index <<  endl;

}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobalTriangle(FFace *face, int increment)
{		
	face->m_index[0] = face->m_index[0] + increment;
	face->m_index[1] = face->m_index[1] + increment;
	face->m_index[2] = face->m_index[2] + increment;
	
	m_global_faces.push_back(face);   
   // cout << "Adding Global Face - " << face->m_index[0] << " " << face->m_index[1] << " " << face->m_index[2] << " " << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::lazyIntegrate()
{
	cout <<endl << timestamp << "Start Lazy Integration..." << endl << endl;
	
	//printLocalBufferStatus();
	//printGlobalBufferStatus();
	
	size_t num_current_local_vertices  = m_local_vertices.size();
	size_t num_current_local_faces  = m_local_faces.size();
		
	size_t num_current_global_vertices = m_global_vertices.size();
	size_t num_current_global_faces = m_global_faces.size();

    // Add all vertices
    for(size_t i = 0; i < num_current_local_vertices; i++)
    {
        addGlobalVertex(m_local_vertices[i]);
    }
    
    // Add all faces
    for(size_t i = 0; i < num_current_local_faces; i++)
    {
		addGlobalTriangle(m_local_faces[i], num_current_global_vertices);
    }
    
    clearLocalBuffer();
    
    cout << endl << timestamp << "Finished Lazy Integration..." << endl << endl;
	
	//printLocalBufferStatus();
	//printGlobalBufferStatus();
	
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::remoteIntegrate(vector<FFace*>& faces)
{	
	/*
	MapIterator it;
	it =  global_vertices_map.find(m_global_vertices[0]->m_position);
	cout << "Found it_index: " << it->second << endl;
	*/
	
	cout << "Start Remote Integrate..." << endl;
	
	MapIterator it;
	
	for(size_t i = 0; i < faces.size(); i++)
    {
		FFace* face = faces[i];
		
		for(int j = 0; j < 3; j++)
		{
			FVertex* v =  m_local_vertices[face->m_index[j]]; // vertices[j];
			
			//cout << "Find in map: " << v->m_position << endl;
			
			//it = global_vertices_map.find(v->m_position);
			
			//cout << "Found it_index: " << it->second << endl;
			
			std::pair<MapIterator,bool> const& r=global_vertices_map.insert(std::pair<VertexT, size_t>(v->m_position, m_global_index));
			
				if (r.second) { 
					cout << "added vertex" << endl;
					addGlobalVertex(v);
					if (m_global_index == 35507 || m_global_index == 35508) {
						cout << "Index[" <<  "35460" << "] " << m_global_vertices[35460]->m_self_index << " vertex: " << face->m_index[j] << endl;
					}
					face->m_index[j] = v->m_self_index;
					//cout << "m_self " << global_vertices_map[v->m_position] << endl;
				} else {
					
					// value wasn't inserted because my_map[foo_obj] already existed.
					// note: the old value is available through r.first->second
					// and may not be "some value"
					
					cout << "already have vertex " << endl;
					
					face->m_index[j] = r.first->second;
					if(face->m_index[j] >= m_global_vertices.size())
					{
					    cout << r.first->first << " " << r.first->second << endl;
						cout << "error: " <<  face->m_index[j] << " >=  " << m_global_vertices.size() << endl;
					}
					else
						face->vertices[j] = m_global_vertices[face->m_index[j]];
				}
				//trying to find error
				/* int ind = m_global_index;
				if(ind != m_global_vertices[ind]->m_self_index) 
				cout << "Index[" <<  ind << "] " << m_global_vertices[i]->m_self_index << endl; 
				*/
			
			/*
			if(it == global_vertices_map.end())
			{
				cout << "Found it_index: " << it->second << endl;
				
				cout << "size " << global_vertices_map.size() <<endl;
				cout << "addVertex" << endl;
				
				//cout << "end_index: " << it->second << endl;
				
				//cout << "before insertion " << v->m_self_index << "-" << v->m_position << endl;
				
				//addGlobalVertex(v);
				//cout << "Insert new vertex with self_index: " << v->m_self_index << " current size: " << m_global_index <<endl;
				global_vertices_map.insert(std::pair<VertexT, size_t>(v->m_position, v->m_self_index));
				cout << "size " << global_vertices_map.size() <<endl <<endl;
				
				/*
				face->m_index[j] = v->m_self_index;
				if(	m_global_vertices[v->m_self_index]->m_self_index != v->m_self_index)
				{
					cout << "inconsistency during addglobal vertex" << endl;
					cout << "global buffer " << 	m_global_vertices[v->m_self_index] << endl;
					cout << "global buffer " << 	v->m_self_index << endl;
				}
				 * /
			}
			else
			{	
				cout << "already existent" << endl;
				/*if (it->second != m_global_vertices[it->second]->m_self_index)
				{	
					//cout << "map position " << it->first << " local pos: " << v->m_position << "global pos:" << m_global_vertices[it->second]->m_position << endl;
					cout << "Vertex already in buffer, map_int: " << it->second << " buffer id: " << m_global_vertices[it->second]->m_self_index << endl;
				}
				//cout << "Vertex already in global buffer at " << it->second << " should be equal to " << face->vertices[j]->m_self_index << endl;
				//cout << "Vertex is bla" << it->first;
				
				face->m_index[j] = it->second;
				face->vertices[j] = m_global_vertices[face->m_index[j]];
				//face->vertices[j]->m_self_index = it->second;
				//temp = it->second;
				//cout << "Vertex already in global buffer at " << it->second << " should be equal to " << m_global_vertices[temp]->m_self_index << endl;
				//cout << "Vertex is bla" << it->first;	
				* /
			}
			*/
		}
    }
	
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::buildTree()
{
	size_t num_current_global_vertices = m_global_vertices.size();
	size_t num_current_global_faces = m_global_faces.size();

	std::list<Triangle> triangles;	

	if(num_current_global_faces > 0)
	{
		for(size_t i = 0; i < num_current_global_faces; i++)
		{
			FVertex* v0 = m_global_vertices[m_global_faces[i]->m_index[0]];
			FVertex* v1 = m_global_vertices[m_global_faces[i]->m_index[1]];
			FVertex* v2 = m_global_vertices[m_global_faces[i]->m_index[2]];
			
			Point a(v0->m_position.x, v0->m_position.y, v0->m_position.z);
			Point b(v1->m_position.x, v1->m_position.y, v1->m_position.z);
			Point c(v2->m_position.x, v2->m_position.y, v2->m_position.z);
			
			triangles.push_back(Triangle(a,b,c));
		}
		
		tree.clear();
		tree.insert(triangles.begin(), triangles.end());
	}	
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::buildVertexMap()
{
	size_t num_current_global_vertices = m_global_vertices.size();
	
	if(num_current_global_vertices > 0)
	{
		global_vertices_map.clear();
		for(size_t i = 0; i < num_current_global_vertices; i++)
		{
			global_vertices_map.insert(std::pair<VertexT, size_t>(m_global_vertices[i]->m_position, i));
		}
	}
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::sortFaces(vector<FFace*>& remote_faces, vector<FFace*>& integration_faces)
{	
	cout << timestamp << "Start Sorting Faces... " << endl <<endl;
	
	cout << "Distance Threshold: " << sqrt(threshold) << endl;
	cout << "Squared Distance Threshold: " << threshold << endl;
	
	redundant_faces = 0;
	special_case_faces = 0;
		
	for(size_t i = 0; i < m_local_faces.size(); i++)
	{	
		FFace* face = m_local_faces[i];
		
		FVertex* v0 = m_local_vertices[face->m_index[0]];
		FVertex* v1 = m_local_vertices[face->m_index[1]];
		FVertex* v2 = m_local_vertices[face->m_index[2]];
		
		Point a(v0->m_position.x, v0->m_position.y, v0->m_position.z);
		Point b(v1->m_position.x, v1->m_position.y, v1->m_position.z);
		Point c(v2->m_position.x, v2->m_position.y, v2->m_position.z);
		
		FT dist_a = tree.squared_distance(a);
		FT dist_b = tree.squared_distance(b);
		FT dist_c = tree.squared_distance(c);
		
		Triangle temp = Triangle(a,b,c);
		
		if (dist_a > threshold && dist_b > threshold && dist_c > threshold)
		{
			
			bool result = true;
			try {
				result = tree.do_intersect(temp);
			} catch (...)
		    {
				//cout << "i: " << i << " hier werf ich nen fehler" << endl;
			}
			if (result)
			{
				// unhandled exceptional situation
				//cout << "found intersection out of distance" << endl;
				//find solution
				face->r = 200;
				face->g = 200;
				face->b = 200;	
				special_case_faces++;
				// lassen wir erstmal ganz weg
			}
			//Best Case: detect non overlapping local face
			else
			{
				face->r = 0;
				face->g = 0;
				face->b = 200;
				
				remote_faces.push_back(face);	
			}
		}
		else if(dist_a < threshold && dist_b < threshold && dist_c < threshold)
		{
			// Delete Case: redundant faces
			face->r = 200;
			face->g = 200;
			face->b = 0;
			redundant_faces++;
			// lassen wir ganz weg

		}
		else
		{
				//partial overlaping, gaps etc. case
				//cout << "found within distance" << endl;
				//ggf. hier intersection erzwingen ?! (wall method s. paper)
				
			  //  face->r = 100;
			  //	face->g = 100;
			  //	face->b = 0;
			
			/*if (tree.do_intersect(temp))
			{
				* 
				*/
			
				//cout << "found intersection within distance" << endl;
				//find solution
				face->r = 200;
				face->g = 0;
				face->b = 0;
				
				integration_faces.push_back(face);
				
				// integrieren wir erstmal einfach so
			//}
			
		}
	}
	
	printFaceSortingStatus();
    cout << timestamp << "Finished Sorting Faces..." << endl;
 	
	/*face->m_index[0] = face->m_index[0] + increment;
	face->m_index[1] += increment;
	face->m_index[2] += increment;
	
	m_global_faces.push_back(face);*/
    
   // cout << "Adding Tree Face - " << face->m_index[0] << " " << face->m_index[1] << " " << face->m_index[2] << " " << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printFaceSortingStatus()
{
	size_t num_current_local_faces  = m_local_faces.size();
	
		double remote_ratio = ((double) remote_faces.size() / (double) num_current_local_faces) * 100;
		double integration_ratio = ((double) integration_faces.size() / (double) num_current_local_faces) * 100;
		double redundant_ratio = ((double) redundant_faces / (double) num_current_local_faces) * 100;
		double special_case_ratio = ((double) special_case_faces  / (double) num_current_local_faces) * 100;
		
		cout << "Found # " <<  remote_faces.size() << " Remote Faces... " << remote_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  integration_faces.size()  << " Integration Faces... " << integration_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  redundant_faces  << " Redundant Faces... " << redundant_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  special_case_faces  << " Special Case Faces... " <<  special_case_ratio << "% of all incoming" << endl;		
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::integrate()
{
	cout <<endl << timestamp << "Start Integrating... " << endl;
	
	printLocalBufferStatus();
	printGlobalBufferStatus(); 
    
    if (m_global_vertices.size() == 0)
    {
		lazyIntegrate(); //shorten
	}
    else
    {
		buildTree();
		buildVertexMap();
		// for all faces in local buffer
		// check whether to add face to tree/global buffer
		
		sortFaces(remote_faces, integration_faces);
		
		//lazyIntegrate();
		cout << "Start Integration" << endl;
		remoteIntegrate(remote_faces);
		//remoteIntegrate(integration_faces);
	}
	
    // for all faces in local buffer
    // add face to globalbuffer()
    /* {
		 * int status = checkOverlappingStatus(face)
		 * 
		 * if(0 = completely redundant) break;
		 * if(1 = border overlap) --> clip(boarder_faces, face)
		 * if(2 = interior overlap) --> später 
		 * 
	   }
    */
     
    clearLocalBuffer();
   
    cout << endl << "Errors" << endl;
    for(unsigned int i = 0; i < m_global_vertices.size(); i++)
    {
		if(i != m_global_vertices[i]->m_self_index) 
		cout << "Index[" <<  i << "] " << m_global_vertices[i]->m_self_index << endl; 
	}
	
    cout << endl << timestamp << "Finished Integrating..." << endl << endl;
	
	printLocalBufferStatus();
	printGlobalBufferStatus();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMeshAndIntegrate(MeshBufferPtr mesh)
{
	addMesh(mesh);
	integrate();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMeshAndLazyIntegrate(MeshBufferPtr mesh)
{
	addMesh(mesh);
	lazyIntegrate();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::finalize()
{
	
    cout << endl << timestamp << "Finalizing mesh..." << endl;

    boost::unordered_map<FusionVertex<VertexT, NormalT>*, int> index_map;

    int numVertices = m_global_vertices.size();
    int numFaces 	= m_global_faces.size();
    
    // Default Color values. Used if regions should not be colored.
    float r=0, g=255, b=0;
    std::vector<uchar> faceColorBuffer;

    floatArr vertexBuffer( new float[3 * numVertices] );
    floatArr normalBuffer( new float[3 * numVertices] );
    ucharArr colorBuffer(  new uchar[3 * numVertices] );
    uintArr  indexBuffer(  new unsigned int[3 * numFaces] );

    // Set the Vertex and Normal Buffer for every Vertex.
    typename vector<FVertex*>::iterator vertices_iter = m_global_vertices.begin();
    typename vector<FVertex*>::iterator vertices_end  = m_global_vertices.end();
    
    for(size_t i = 0; vertices_iter != vertices_end; ++i, ++vertices_iter)
    {
        vertexBuffer[3 * i] =     (*vertices_iter)->m_position[0];
        vertexBuffer[3 * i + 1] = (*vertices_iter)->m_position[1];
        vertexBuffer[3 * i + 2] = (*vertices_iter)->m_position[2];

        normalBuffer [3 * i] =     -(*vertices_iter)->m_normal[0];
        normalBuffer [3 * i + 1] = -(*vertices_iter)->m_normal[1];
        normalBuffer [3 * i + 2] = -(*vertices_iter)->m_normal[2];

        // Map the vertices to a position in the buffer.
        // This is necessary since the old indices might have been compromised.
        index_map[*vertices_iter] = i;
    }

    typename vector<FusionFace<VertexT, NormalT>*>::iterator face_iter = m_global_faces.begin();
    typename vector<FusionFace<VertexT, NormalT>*>::iterator face_end  = m_global_faces.end();

    for(size_t i = 0; face_iter != face_end; ++i, ++face_iter)
    {
		
		r=(float) (*face_iter)->r;
		g=(float) (*face_iter)->g;
		b=(float) (*face_iter)->b;
		
        indexBuffer[3 * i]      = index_map[m_global_vertices[(*face_iter)->m_index[0]]];
        indexBuffer[3 * i + 1]  = index_map[m_global_vertices[(*face_iter)->m_index[1]]];
        indexBuffer[3 * i + 2]  = index_map[m_global_vertices[(*face_iter)->m_index[2]]];

/*
		indexBuffer[3 * i]      = (*face_iter)->m_index[0];
        indexBuffer[3 * i + 1]  = (*face_iter)->m_index[1];
        indexBuffer[3 * i + 2]  = (*face_iter)->m_index[2];
*/
	
        colorBuffer[indexBuffer[3 * i]  * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i]  * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i]  * 3 + 2] = b;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 2] = b;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 2] = b;
        
        /// TODO: Implement materials
        // faceColorBuffer.push_back( r );
        // faceColorBuffer.push_back( g );
        // faceColorBuffer.push_back( b );
    
    }
    // Hand the buffers over to the Model class for IO operations.

    if ( !this->m_meshBuffer )
    {
        this->m_meshBuffer = MeshBufferPtr( new MeshBuffer );
    }   
    
    this->m_meshBuffer->setVertexArray( vertexBuffer, numVertices );
    this->m_meshBuffer->setVertexColorArray( colorBuffer, numVertices );
    this->m_meshBuffer->setVertexNormalArray( normalBuffer, numVertices  );
    this->m_meshBuffer->setFaceArray( indexBuffer, numFaces );
    //this->m_meshBuffer->setFaceColorArray( faceColorBuffer );
    this->m_finalized = true;
   
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::flipEdge(uint v1, uint v2)
{
	cout << "No Edge no Flip!" << endl;
	cout << "But these are two nice uints" << v1 << ", " << v2 << endl;
}

} // namespace lvr
