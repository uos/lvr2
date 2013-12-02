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
 *  @author Sebastian Puetz (spuetz@uos.de)
 */

namespace lvr
{

///
/// Constructors
///

template<typename VertexT, typename NormalT> FusionMesh<VertexT, NormalT>::FusionMesh()
{

	verbose = false;
   m_local_index = 0;
   m_global_index = 0;
}

template<typename VertexT, typename NormalT> FusionMesh<VertexT, NormalT>::FusionMesh(MeshBufferPtr mesh)
{
	verbose = false;
   FusionMesh();
   addMesh(mesh);
   integrate();
}

///
/// Methods of BaseMesh
///

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addVertex(VertexT v)
{
	m_local_vertices.push_back(new FusionVertex<VertexT, NormalT>(v));	
	m_local_index++;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addNormal(NormalT n)
{
    assert(m_local_index == m_local_vertices.size());
    m_local_vertices[m_local_index - 1]->m_normal = n;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c/*, FFace* &face*/)
{		
    // Create a new face
    FFace* face = new FFace;

	face->m_index[0] = a;
	face->m_index[1] = b;
	face->m_index[2] = c;
	
	FVertex* v0 = m_local_vertices[a];
	FVertex* v1 = m_local_vertices[b];
	FVertex* v2 = m_local_vertices[c];
	
	/*if (v0->m_position == v1->m_position || v1->m_position == v2->m_position || v0->m_position == v2->m_position) {    
		cout << "This face is degenerated from beginning" << endl;
		cout << v0->m_position << ", " << v1->m_position << ", " << v2->m_position << endl;
	}*/
	
	face->vertices[0] = v0;
	face->vertices[1] = v1;
	face->vertices[2] = v2;

	m_local_faces.push_back(face);
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::finalize()
{
    cout << endl << timestamp << "Finalizing mesh..." << endl;

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
    }

    typename vector<FusionFace<VertexT, NormalT>*>::iterator face_iter = m_global_faces.begin();
    typename vector<FusionFace<VertexT, NormalT>*>::iterator face_end  = m_global_faces.end();

    for(size_t i = 0; face_iter != face_end; ++i, ++face_iter)
    {
		if ((*face_iter)->is_valid) { //NUR ZUM TESTEN
		
		r=(float) (*face_iter)->r;
		g=(float) (*face_iter)->g;
		b=(float) (*face_iter)->b;

		indexBuffer[3 * i]      = (*face_iter)->m_index[0];
        indexBuffer[3 * i + 1]  = (*face_iter)->m_index[1];
        indexBuffer[3 * i + 2]  = (*face_iter)->m_index[2];

	
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

///
/// Fusion Specific Methods
///

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMesh(MeshBufferPtr mesh)
{	
    size_t num_verts, num_norms, num_faces;
    floatArr vertices = mesh->getVertexArray(num_verts);
    floatArr normals = mesh->getVertexNormalArray(num_norms);
    
    if(num_norms != num_verts)
		cout << "Unequal number of vertices and normals" << endl;
    
    clearLocalBuffer();
    
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
    /*for(size_t i = 0; i < num_norms; i++)
    {
         addNormal(NormalT(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]));
		
    }*/
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::integrate()
{
	cout <<endl << timestamp << "Start Integrating... " << endl;
	
	printLocalBufferStatus();
	printGlobalBufferStatus(); 
    
    if (m_global_vertices.size() == 0)
    {
		addGlobal(m_local_faces);
	}
    else
    {
		buildTree();
		buildVertexMap();
		
		sortFaces();
		
		cout << "Start Integration" << endl;
		addGlobal(remote_faces);
		intersectIntegrate(intersection_faces);
	}
	 
    clearLocalBuffer();
   
    cout << endl << "Face Errors" << endl;
    for(unsigned int i = 0; i < m_global_faces.size(); i++)
    {
		if (m_global_faces[i]->m_index[0] >= m_global_index || 
				m_global_faces[i]->m_index[1] >= m_global_index || 
				m_global_faces[i]->m_index[2] >= m_global_index) 
		{
			cout << "Vertex Indices for Face[" << i << "]: " << 
				m_global_faces[i]->m_index[0] << ", " << 
				m_global_faces[i]->m_index[1] << ", " << 
				m_global_faces[i]->m_index[0] << endl;
			cout << "m_global_index: " << m_global_index << endl;
		}
	}
    
    cout << endl << "Vertice Errors" << endl;
    for(unsigned int i = 0; i < m_global_vertices.size(); i++)
    {
		if(i != m_global_vertices[i]->m_self_index) 
		cout << "Index[" <<  i << "] " << m_global_vertices[i]->m_self_index << endl; 
	}
	
    cout << endl << timestamp << "Finished Integrating..." << endl << endl;
	
	printLocalBufferStatus();
	printGlobalBufferStatus();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::lazyIntegrate()
{
	cout <<endl << timestamp << "Start Lazy Integration..." << endl << endl;
	
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
		FFace* face = m_local_faces[i]; 
		face->m_index[0] = face->m_index[0] + num_current_global_vertices;
		face->m_index[1] = face->m_index[1] + num_current_global_vertices;
		face->m_index[2] = face->m_index[2] + num_current_global_vertices;
		addGlobalFace(face);
    }
    
    clearLocalBuffer();
    
    cout << endl << timestamp << "Finished Lazy Integration..." << endl << endl;
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

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMeshAndRemoteIntegrateOnly(MeshBufferPtr mesh)
{
	addMesh(mesh);
	
	cout <<endl << timestamp << "Start Integrating... " << endl;
	
	printLocalBufferStatus();
	printGlobalBufferStatus(); 
    
    if (m_global_vertices.size() == 0)
    {
		addGlobal(m_local_faces);
	}
    else
    {
		buildTree();
		buildVertexMap();
		
		sortFaces();
		
		addGlobal(remote_faces);
	}
	 
    clearLocalBuffer();
	
    cout << endl << timestamp << "Finished Integrating..." << endl << endl;
	
	printLocalBufferStatus();
	printGlobalBufferStatus();
}

///
/// Clear Methods (internal)
///

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::clearLocalBuffer()
{	
	for (int i = 0; i < m_local_vertices.size(); i++) {
		if (!m_local_vertices[i]->is_valid) {
			delete m_local_vertices[i];
		}
	}
	m_local_vertices.clear();	
	for (int i = 0; i < m_local_faces.size(); i++) {
		if (!m_local_faces[i]->is_valid) {
			delete m_local_faces[i];
		}
	}
	m_local_faces.clear();
	m_local_index = 0;
}

/*template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::clearGlobalBuffer()
{
	m_global_index = 0;
	m_global_vertices.clear();
	m_global_faces.clear();
}*/

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::reset()
{
	clearLocalBuffer();
	clearGlobalBuffer();
}

///
/// Print Methods (internal)
///

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printLocalBufferStatus()
{
	cout << timestamp << "Local Buffer" << endl;
	cout << timestamp << "#Index      :" << m_local_index <<  endl;
	cout << timestamp << "#Vertices   :" << m_local_vertices.size() <<  endl;
	cout << timestamp << "#Faces      :" << m_local_faces.size() << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printGlobalBufferStatus()
{
	cout << timestamp << "Global Buffer" << endl;
	cout << timestamp << "#Index       :" << m_global_index <<  endl;
	cout << timestamp << "#Vertices    :" << m_global_vertices.size() <<  endl;
	cout << timestamp << "#Faces       :" << m_global_faces.size() << endl; 
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printFaceSortingStatus()
{
	size_t num_current_local_faces  = m_local_faces.size();
	
		double remote_ratio = ((double) remote_faces.size() / (double) num_current_local_faces) * 100;
		double intersection_ratio = ((double) intersection_faces.size() / (double) num_current_local_faces) * 100;
		double closeby_ratio = ((double) closeby_faces.size() / (double) num_current_local_faces) * 100;
		double redundant_ratio = ((double) redundant_faces / (double) num_current_local_faces) * 100;
		double special_case_ratio = ((double) special_case_faces  / (double) num_current_local_faces) * 100;
		cout << endl;
		cout << "Found # " <<  remote_faces.size() << " Remote Faces... " << remote_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  intersection_faces.size()  << " Intersection Faces... " << intersection_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  closeby_faces.size() << " Closeby but not intersecting Faces... " << closeby_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  redundant_faces  << " Redundant Faces... " << redundant_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  special_case_faces  << " Special Case Faces... " <<  special_case_ratio << "% of all incoming" << endl;		
		cout << endl;
}

///
/// Integration Methods (internal)
///

template<typename VertexT, typename NormalT> ETriangle FusionMesh<VertexT, NormalT>::faceToETriangle(FFace *face)
{
	FVertex* v0 = face->vertices[0];
	FVertex* v1 = face->vertices[1];
	FVertex* v2 = face->vertices[2];
	
	Point a(v0->m_position[0], v0->m_position[1], v0->m_position[2]);
	Point b(v1->m_position[0], v1->m_position[1], v1->m_position[2]);
	Point c(v2->m_position[0], v2->m_position[1], v2->m_position[2]);
	
	return ETriangle(a,b,c, face->m_self_index);
}

template<typename VertexT, typename NormalT> Plane FusionMesh<VertexT, NormalT>::faceToPlane(FFace *face)
{
	FVertex* v0 = face->vertices[0];
	FVertex* v1 = face->vertices[1];
	FVertex* v2 = face->vertices[2];
	
	Point a = Point(v0->m_position[0], v0->m_position[1], v0->m_position[2]);
	Point b = Point(v1->m_position[0], v1->m_position[1], v1->m_position[2]);
	Point c = Point(v2->m_position[0], v2->m_position[1], v2->m_position[2]);
	
	return Plane(a,b,c);
}

template<typename VertexT, typename NormalT> AffineTF FusionMesh<VertexT, NormalT>::calcTransform(FFace *face)
{
	ETriangle tri = faceToETriangle(face);
	
	Point a = tri.vertex(1);
	Point b = tri.vertex(2);
	Point c = tri.vertex(3);
	
	// calculate the plane-vectors
	CGAL::Vector_3<K> vec_AB = b - a;
	CGAL::Vector_3<K> vec_AC = c - a;
	
	// calculate the required angles for the rotations 
	double alpha 	= atan2(vec_AB.z(), vec_AB.x());
	double beta 	= -atan2(vec_AB.y(), cos(alpha)*vec_AB.x() + sin(alpha)*vec_AB.z());
	double gamma 	= -atan2(-sin(alpha)*vec_AC.x()+cos(alpha)*vec_AC.z(),
				sin(beta)*(cos(alpha)*vec_AC.x()+sin(alpha)*vec_AC.z()) + cos(beta)*vec_AC.y()); 
	
	AffineTF trans ( 1, 0, 0, -a.x(),
					 0, 1, 0, -a.y(),
					 0, 0, 1, -a.z(),
					 1);
	
	AffineTF roty ( cos(alpha), 0, sin(alpha), 0,
					0, 1, 0, 0,
					-sin(alpha), 0, cos(alpha), 0,
					1);
	
	AffineTF rotz ( cos(beta), -sin(beta), 0, 0,
					sin(beta), cos(beta), 0, 0,
					0, 0, 1, 0,
					1);
	
	AffineTF rotx (	1, 0, 0, 0,
					0, cos(gamma), -sin(gamma), 0,
					0, sin(gamma), cos(gamma), 0,
					1);

	/*
	 * transformation to the xy-plane:
	 * first translate till the point a is in the origin of ordinates,
	 * then rotate around the the y-axis till the z-value of the point b is zero,
	 * then rotate around the z-axis till the y-value of the point b is zero,
	 * then rotate around the x-axis till all z-values are zero
	 */

	return rotx * rotz * roty * trans;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobalVertex(FVertex *v)
{
    m_global_vertices.push_back(v);
    m_global_vertices[m_global_index]->m_self_index = m_global_index;
    m_global_index++;
    v->is_valid = true;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobalFace(FFace *face)
{	
	face->is_valid = true;
	face->m_self_index = m_global_faces.size();
	m_global_faces.push_back(face);   
   // cout << "Adding Global Face - " << face->m_index[0] << " " << face->m_index[1] << " " << face->m_index[2] << " " << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::buildTree()
{
	tree.clear();
	tree_triangles.clear();
	size_t num_current_global_vertices = m_global_vertices.size();
	size_t num_current_global_faces = m_global_faces.size();
	
	if(num_current_global_faces > 0)
	{
		for(size_t i = 0; i < num_current_global_faces; i++)
		{
			FFace* face = m_global_faces[i];
			ETriangle tri = faceToETriangle(face);
			tree_triangles.push_back(tri);
		}
		tree.insert(tree_triangles.begin(), tree_triangles.end());
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

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::sortFaces()
{	
	cout << timestamp << "Start Sorting Faces... " << endl <<endl;
	
	cout << "Distance Threshold: " << sqrt(threshold) << endl;
	cout << "Squared Distance Threshold: " << threshold << endl;
	
	remote_faces.clear();
	intersection_faces.clear();
	closeby_faces.clear();
	redundant_faces = 0;
	special_case_faces = 0;
	int far_tree_intersect_fails = 0;
	int close_tree_intersect_fails = 0;
	int squared_distance_fails = 0;
	
	bool result = false;
	try {
		result = tree.accelerate_distance_queries();
	} catch (...)
	{
		cout << "function sortFaces: tree.accelerate_distance_queries() failed" << endl;
	}
	if (result) {
		cout << "successfully accelerated_distance_queries" << endl; 
	}
	
	FFace* face;
	FVertex* v0;
	FVertex* v1;
	FVertex* v2;
	Triangle temp;

	for(size_t i = 0; i < m_local_faces.size(); i++)
	{
		face = m_local_faces[i];
		v0 = m_local_vertices[face->m_index[0]];
		v1 = m_local_vertices[face->m_index[1]];
		v2 = m_local_vertices[face->m_index[2]];
		Point a(v0->m_position.x, v0->m_position.y, v0->m_position.z);
		Point b(v1->m_position.x, v1->m_position.y, v1->m_position.z);
		Point c(v2->m_position.x, v2->m_position.y, v2->m_position.z);
		try {
			v0->m_tree_dist = tree.squared_distance(a);
			v1->m_tree_dist = tree.squared_distance(b);
			v2->m_tree_dist = tree.squared_distance(c);
		} catch (...)
		{
			//cout << "WARNING: tree.squared_distance call failed, face is skipped during sortFaces()" << endl;
			squared_distance_fails++;
			continue;
		}
		temp = Triangle(a,b,c);
		
		//check wether distance to all vertices is above threshold
		if (v0->m_tree_dist > threshold && v1->m_tree_dist > threshold && v2->m_tree_dist > threshold)
		{
			bool result = true;
			try {
				result = tree.do_intersect(temp);
			} catch (...)
			{
				far_tree_intersect_fails++;
			}
			if (result)
			{
				// unhandled exceptional situation
				//find solution
				special_case_faces++;
			}
			//detected non overlapping local face
			else {
				remote_faces.push_back(face);
			}
		}
		else if(v0->m_tree_dist <= threshold && v1->m_tree_dist <= threshold && v2->m_tree_dist <= threshold)
		{	
			// Delete Case: redundant faces
			redundant_faces++;
		}
		else
		{	
			bool result = false;
			try {
				result = tree.do_intersect(temp);
			} catch (...)
			{
				close_tree_intersect_fails++;
			}
			if(result) {
				// Intersection Case:
				intersection_faces.push_back(face);
			}
			else {
				//partial overlaping, gaps etc. case
				//ggf. hier intersection erzwingen ?! (wall method s. paper)
				closeby_faces.push_back(face);
			}
		}
	}	
	printFaceSortingStatus();
	cout << "WARNING: For " << squared_distance_fails << " of all Faces tree.squared_distance() failed for at least one point" << endl;
	cout << "WARNING: For " << far_tree_intersect_fails << " of Special Case Faces call to tree.intersect() failed" << endl;
	cout << "WARNING: For " << close_tree_intersect_fails << " of Closeby Faces call to tree.intersect() failed" << endl << endl; 
    cout << timestamp << "Finished Sorting Faces..." << endl;
 	
	/*face->m_index[0] = face->m_index[0] + increment;
	face->m_index[1] += increment;
	face->m_index[2] += increment;
	
	m_global_faces.push_back(face);*/
    
   // cout << "Adding Tree Face - " << face->m_index[0] << " " << face->m_index[1] << " " << face->m_index[2] << " " << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobal(vector<FFace*>& faces)
{	
	int degentFaces = 0;
	
	//cout << "Start Remote Integrate..." << endl;
	
	MapIterator it;
	
	for(size_t i = 0; i < faces.size(); i++)
    {
		FFace* face = faces[i];
		
		for(int j = 0; j < 3; j++)
		{
			FVertex* v =  m_local_vertices[face->m_index[j]];
			std::pair<MapIterator,bool> const& r=global_vertices_map.insert(std::pair<VertexT, size_t>((VertexT)v->m_position, m_global_index));
			
				if (r.second) { // && (global_vertices_map.count(v->m_position) == 1)) {
// FEHLER: MANCHMAL WIRD NICHT ERKANNT DAS DIE VERTEX BEREITS IN DER MAP LIEGT
					addGlobalVertex(v);
					face->m_index[j] = v->m_self_index;
					face->vertices[j] = m_global_vertices[face->m_index[j]]; 
				} else {
					
					// value wasn't inserted because my_map[foo_obj] already existed.
					// note: the old value is available through r.first->second
					// and may not be "some value"
					 
					face->m_index[j] = r.first->second;
					
					if(face->m_index[j] >= m_global_vertices.size())
					{
					    cout << r.first->first << " " << r.first->second << endl;
						cout << "error: " <<  face->m_index[j] << " >=  " << m_global_vertices.size() << endl;
					}
					else {
						face->vertices[j] = m_global_vertices[face->m_index[j]];
					}
				}
		}
		//check for degenerated faces
		if (face->m_index[0] == face->m_index[1] || face->m_index[0] == face->m_index[2] || face->m_index[1] == face->m_index[2]) {
			degentFaces++;
		}	
		else {
			addGlobalFace(face);
		}
    }
    cout << "Skipped " << degentFaces << " Faces due to degeneration" << endl;
	//cout << "Finished Remote Integrate" << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::triangulateAndAdd(vector<Point>& vertices, Tree& tree)
{
	vector<FFace*> new_faces;
	Delaunay dt;
	// add points/vertices to Delauny Triangulation
	for (int i = 0; i < vertices.size(); i++) {
		PointD p(vertices[i].x(), vertices[i].y(), vertices[i].z());
		dt.push_back(p);
	}
	// extract new faces
	Delaunay::Finite_faces_iterator it;
	for (it = dt.finite_faces_begin(); it != dt.finite_faces_end(); it++)
	{	
		TriangleD tri = dt.triangle(it);
		int count_to_close_vertices = 0;
		int count_to_far_vertices = 0;
		for (int i = 0; i < 3; i++) {
			PointD pd = tri.vertex(i);
			Point p(pd.x(), pd.y(), pd.z());
			FT dist = tree.squared_distance(p);
			if (dist <= threshold && dist >= 0) {
				cout << "to close vertex dist: " << dist << endl;
				count_to_close_vertices++;
			}
			else {
				cout << "to far vertex dist: " << dist << endl;
				count_to_far_vertices++;
			}
			addVertex(VertexT(p.x(), p.y(), p.z()));
			cout << "added new vertex" << endl;
		}
		// check for remote faces that might have been created
		if (count_to_close_vertices < 3 && count_to_far_vertices < 3) {
			addTriangle(m_local_index-3, m_local_index-2, m_local_index-1);
			new_faces.push_back(m_local_faces[m_local_faces.size()-1]);
			cout << "pushed new face" << endl;
			/*FFace* colorFace = m_local_faces[m_local_faces.size()-1];
				colorFace->r = 200;
				colorFace->g = 0;
				colorFace->b = 0;*/
		}
	}
	addGlobal(new_faces);
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::assignToBorderRegion(vector<PointSet>& vertexRegions, vector<Point>new_vertices)
{	
	PointSetIterator it;
	bool inserted = false;
	
	// check existing regions
	for(int i = 0; i < vertexRegions.size(); i++)
	{	
		for(int j = 0; j < new_vertices.size(); j++) {
			Point p = new_vertices[j];
			it = vertexRegions[i].find(p);
			if (it != vertexRegions[i].end()){
				for(int k = 0; k < new_vertices.size(); k++) {
					p = new_vertices[k];
					vertexRegions[i].insert(p);
				}
				j = new_vertices.size();
				i = vertexRegions.size();
				inserted = true;
			}
		}
	}
	// otherwise add new region
	if (!inserted) {
		PointSet set;;
		for(int k = 0; k < new_vertices.size(); k++) {
			Point p = new_vertices[k];
			set.insert(p);
		}
		vertexRegions.push_back(set);
	}
}


template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::intersectIntegrate(vector<FFace*>& faces)
{
	cout << "Start Intersect Integrate..." << endl;
	
	set<int> all_intersect_ids;
	vector<int> intersect_ids; // ids from intersected triangles
	vector<FFace*> gl_intersect_tri;

	// split intersecting faces in local buffer
	splitIntersectFaces(faces, tree);

	// find intersected triangles in the globalmesh
	for(size_t i = 0; i < faces.size(); i++)
	{
		FFace* face = faces[i];
		intersect_ids = getIntersectingTriangles(face);
		all_intersect_ids.insert(intersect_ids.begin(), intersect_ids.end());
	}
	// build buffer of intersecting triangles
	for (set<int>::iterator it = all_intersect_ids.begin(); it != all_intersect_ids.end(); ++it)
	{
		gl_intersect_tri.push_back(m_global_faces[*it]);
	}
	// build tree of local intersecting triangles
	vector<ETriangle> local_tree_triangles;
	Tree local_tree;
	for (size_t i = 0; i < faces.size(); i++) {
		FFace* face = faces[i];
		ETriangle tri = faceToETriangle(face);
		local_tree_triangles.push_back(tri);
	}
	local_tree.insert(local_tree_triangles.begin(), local_tree_triangles.end());
	
	splitIntersectFaces(gl_intersect_tri, local_tree);
	
	cout << "Finished Intersect Integrate ..." << endl;
}

template<typename VertexT, typename NormalT>
void FusionMesh<VertexT, NormalT>::splitIntersectFaces(vector<FFace*>& faces, Tree& tree)
{
	vector<vector<Point> > polys;
	int counterVertices = 0;
	int counterPolygons = 0;
	
	for(size_t i = 0; i < faces.size(); i++)
	{
		cout << " ############### " << endl << endl << endl;

		FFace* face = faces[i];
		vector<Segment> segments;
		vector<Point> points;
		getIntersectionSegments(face, segments, tree); // get all intersected segments off the given face
		bool ok = sortSegments(segments, points);

		if(ok)
		{
			cout << "sort segments was ok: " << endl;
		}
		else
		{
			cout << "sort segments failed!" << endl;
			continue;
		}
		vector<vector<Point> > polygons = buildPolygons(face, points);
		
		AffineTF tf = calcTransform(face);
		AffineTF itf = tf.inverse();
		
		vector<Polygon> polys2D;
		vector<Triangle2D> triangles;
		
		for(int i=0; i<polygons.size(); i++)
		{
			counterVertices += polygons[i].size();
			Polygon polygon;
			for(int j=0; j<polygons[i].size(); j++)
			{
				
				Point p = tf.transform(polygons[i][j]); 
				polygon.push_back(Point2D(p.x(), p.y()));
			}
			
			polys2D.push_back(polygon);	
			polygonTriangulation(polygon, triangles);
		}
		vector<FFace*> newFaces;
		for(vector<Triangle2D>::iterator trit = triangles.begin(); trit != triangles.end(); ++trit)
		{
			Point p1(trit->vertex(0).x(), trit->vertex(0).y(), 0);
			Point p2(trit->vertex(1).x(), trit->vertex(1).y(), 0);
			Point p3(trit->vertex(2).x(), trit->vertex(2).y(), 0);
			p1 = itf.transform(p1);
			p2 = itf.transform(p2);
			p3 = itf.transform(p3);
			
			addVertex(VertexT(p1.x(), p1.y(), p1.z()));
			addVertex(VertexT(p2.x(), p2.y(), p2.z()));
			addVertex(VertexT(p3.x(), p3.y(), p3.z()));
			addTriangle(m_local_index-3, m_local_index-2, m_local_index-1);
			newFaces.push_back(m_local_faces[m_local_faces.size()-1]);
		}
		addGlobal(newFaces);
	}
	
	/*	
	ofstream polyfile;
	polyfile.open("polyfile.ply");
	polyfile << "ply" << endl;
	polyfile << "format ascii 1.0" << endl;
	polyfile << "comment test polygons" << endl;
	polyfile << "element vertex " << counterVertices << endl;
   	polyfile << "property float x" << endl;
   	polyfile << "property float y" << endl;
   	polyfile << "property float z" << endl;
	polyfile << "element face " << polys.size() << endl;
	polyfile << "property list uchar int vertex_indices" << endl;
	polyfile << "end_header" << endl;

	for(vector<vector<Point> >::iterator polyIter = polys.begin(); polyIter != polys.end(); ++polyIter)
	{
		for(vector<Point>::iterator pointIter = polyIter->begin(); pointIter != polyIter->end(); ++ pointIter)
		{
			polyfile << *pointIter << endl;
		}
	}

	counterVertices = 0;
	for(int i=0; i<polys.size(); i++)
	{
		polyfile << polys[i].size() << " ";
		for(int j=0; j<polys[i].size(); j++)
		{
			polyfile << j + counterVertices << " ";
		}
		polyfile << endl;
		counterVertices += polys[i].size();
	}


	vector<PointSet> local_vertexRegions;
	vector<PointSet> global_vertexRegions;
	vector<Point> new_local_vertices;
	vector<Point> new_global_vertices;
	
	FFace* face;
	FFace* glo_face;
	FVertex* v;
	
	local_tree_triangles.clear();
	local_tree.clear();
	
	for(size_t i = 0;./ i < faces.size(); i++)
    {	
		intersect_ids.clear();
		new_local_vertices.clear();
		new_global_vertices.clear();
		
		face = faces[i];
		//create vector of local intersection triangles for tree
		ETriangle tri = faceToETriangle(face);
		local_tree_triangles.push_back(tri);
		
		//check which vertices will be kept from local
		for (int j = 0; j < 3; j++) {
			v = face->vertices[j];
			if (v->m_tree_dist > threshold) {
				Point a(v->m_position.x, v->m_position.y, v->m_position.z);
				new_local_vertices.push_back(a);
			}
		}
		
		//check which vertices to keep from intersecting triangles
		intersect_ids = getIntersectingTriangles(face);
		for (int j = 0; j < intersect_ids.size(); j++) {
			Plane plane = faceToPlane(face);
			glo_face = m_global_faces[intersect_ids[j]];
			glo_face->is_valid = false;
			for (int k = 0; k < 3; k++) {
				v = glo_face->vertices[k];
				Point a(v->m_position.x, v->m_position.y, v->m_position.z);
				FT x = 0;
				try {
					x = CGAL::squared_distance(plane,a);
				} catch (...)
				{
					cout << "function intersectIntegrate: squared_distance(plane,point) failed" << endl;
				}
				if (x > threshold) {
					new_global_vertices.push_back(a);
					//PRÜFEN OB IMMER MINDESTENS EINE VERTEX HINZUGEFÜGT WIRD
				}
				else {
					//EVENTUELL NOCH ABSTAND ZU SEGMENTEN PRÜFEN
					v->is_valid = false; //v aus Global wirklich safe?
				}
			}
		}
		//get Intersection Points and assign them to Border Region
		getIntersectionPoints(face, new_local_vertices, new_global_vertices);
		assignToBorderRegion(local_vertexRegions, new_local_vertices);
		assignToBorderRegion(global_vertexRegions, new_global_vertices);
	}
	
	//build local tree from triangles
	local_tree.insert(local_tree_triangles.begin(), local_tree_triangles.end());
	bool result = false;
	try {
		result = local_tree.accelerate_distance_queries();
	} catch (...)
	{
		cout << "function intersectIntegrate: local_tree.accelerate_distance_queries() failed" << endl;
	}
	if (result) {
		cout << "successfully accelerated_distance_queries" << endl; 
	}
	
	// triangulate and add all Border Regions
	for(int i = 0; i < local_vertexRegions.size(); i++) {
		new_local_vertices.clear();
		for (PointSetIterator it = local_vertexRegions[i].begin(); it != local_vertexRegions[i].end(); ++it) {
			new_local_vertices.push_back(*it);
		}
		triangulateAndAdd(new_local_vertices, tree);
	}
	for(int i = 0; i < global_vertexRegions.size(); i++) {
		new_global_vertices.clear();
		for (PointSetIterator it = global_vertexRegions[i].begin(); it != global_vertexRegions[i].end(); ++it) {
			new_global_vertices.push_back(*it);
		}
		triangulateAndAdd(new_global_vertices, local_tree);
	}
	cout << "Finished Intersect Integrate ..." << endl;
*/

}


template<typename VertexT, typename NormalT>
vector<vector<Point> > FusionMesh<VertexT, NormalT>::buildPolygons(FFace *face, vector<Point>& points)
{
	vector<vector<Point> > polys;
	vector<Segment> faceSegs = face2Segments(face);
	Point startPoint = points.front();
	Point endPoint = points.back();

	bool sOnA = pointOnSegment(startPoint, faceSegs[0]);
	bool sOnB = pointOnSegment(startPoint, faceSegs[1]);
	bool sOnC = pointOnSegment(startPoint, faceSegs[2]);
	bool eOnA = pointOnSegment(endPoint, faceSegs[0]);
	bool eOnB = pointOnSegment(endPoint, faceSegs[1]);
	bool eOnC = pointOnSegment(endPoint, faceSegs[2]);

	Point p1 = faceSegs[0].source();
	Point p2 = faceSegs[1].source();
	Point p3 = faceSegs[2].source();

	vector<Point>polygon1;
	vector<Point>polygon2;
	vector<Point>polygon3;
	vector<Point>polygon4;
	
	polygon1.insert(polygon1.end(), points.begin(), points.end());	//add all sorted intersection points
	polygon2.insert(polygon2.end(), points.begin(), points.end());	//add all sorted intersection points

	if(eOnA) // between p1 and p2
	{
		polygon1.push_back(p1);
		polygon2.push_back(p2);
		if(sOnB)
		{
			polygon1.push_back(p3);
		}
		else if(sOnC)
		{
			polygon2.push_back(p3);
		}
		else if(sOnA)
		{
			// TODO 
		}
		else
		{
			polygon1.push_back(p3);
			polygon2.push_back(p3);	
		}
	}
	else if(eOnB) // between p2 and p3
	{
		polygon1.push_back(p2);
		polygon2.push_back(p3);
		if(sOnC)
		{
			polygon1.push_back(p1);
		}
		else if(sOnA)
		{
			polygon2.push_back(p1);
		}
		else if(sOnB)
		{
			// TODO 
		}
		else
		{
			polygon1.push_back(p1);
			polygon2.push_back(p1);	
		}
	}
	else if(eOnC) // between p3 and p1
	{
		polygon1.push_back(p3);
		polygon2.push_back(p1);
		if(sOnA)
		{
			polygon1.push_back(p2);
		}
		else if(sOnB)
		{
			polygon2.push_back(p2);
		}
		else if(sOnC)
		{
			// TODO 
		}
		else{
			polygon1.push_back(p2);
			polygon2.push_back(p2);	
		}
	}
	else if(sOnA)
	{
		polygon1.push_back(p3);
		polygon2.push_back(p3);	
		polygon1.push_back(p1);
		polygon2.push_back(p2);	
	}	
	else if(sOnB)
	{
		polygon1.push_back(p1);
		polygon2.push_back(p1);	
		polygon1.push_back(p2);
		polygon2.push_back(p3);	
	}	
	else if(sOnC)
	{
		polygon1.push_back(p2);
		polygon2.push_back(p2);	
		polygon1.push_back(p1);
		polygon2.push_back(p3);	
	}	


	polys.push_back(polygon1);
	polys.push_back(polygon2);


	return polys; 
}	

template<typename VertexT, typename NormalT>
bool FusionMesh<VertexT, NormalT>::pointOnSegment(Point& p, Segment& s)
{
	return CGAL::squared_distance(s, p) < POINT_DIST_EPSILON;
}


template<typename VertexT, typename NormalT>
vector<Segment> FusionMesh<VertexT, NormalT>::face2Segments(FFace *face)
{
	vector<Segment> segments;
	ETriangle t = faceToETriangle(face);
	Segment s1 = Segment(t[0], t[1]);
	Segment s2 = Segment(t[1], t[2]);
	Segment s3 = Segment(t[2], t[0]);

	segments.push_back(s1);
	segments.push_back(s2);
	segments.push_back(s3);

	return segments;
}



template<typename VertexT, typename NormalT>
vector<int> FusionMesh<VertexT, NormalT>::getIntersectingTriangles(FFace *face)
{
	list<Primitive_id> primitives;
	Primitive_id id;
	vector<int> global_ids;
	
	ETriangle tri = faceToETriangle(face);
	try {
		tree.all_intersected_primitives(tri, back_inserter(primitives));
	} catch (...)
	{
		cout << "function getIntersectingTriangles: tree.all_intersected_primitives() fails" << endl;
	}
	while (!primitives.empty()) {
		id = primitives.front();
		primitives.pop_front();
		ETriangle tri2 = *id;
		global_ids.push_back(tri2.m_self_index);
	}
	return global_ids;
}


template<typename VertexT, typename NormalT> 
bool FusionMesh<VertexT, NormalT>::sortSegments(vector<Segment> &segments, vector<Point> &points)
{
	bool regular = true;

	struct EpsilonComparator {
		double epsilon;
		Point p_zero;	// compairison point
		EpsilonComparator() 
			:	epsilon(POINT_DIST_EPSILON),
				p_zero(0,0,0)
		{
		}

		bool operator()(Point const& a, Point const& b) const
		{
			double dist = CGAL::squared_distance(a, b);
			if(dist < epsilon)
			{
				return false;
			}
			return CGAL::has_smaller_distance_to_point(p_zero, a, b);
		}
	};

	map<Point, vector< Segment>, EpsilonComparator >point2Seg;
	pair< map< Point, vector< Segment > >::iterator, bool> in1, in2;

	for(size_t i = 0; i < segments.size(); i++)
	{
		Segment& seg = segments[i];
		vector<Segment>v1, v2;
		in1 = point2Seg.insert(pair<Point, vector<Segment> >(seg.source(), v1));
		in2 = point2Seg.insert(pair<Point, vector<Segment> >(seg.target(), v2));
		in1.first->second.push_back(seg);
		in2.first->second.push_back(seg);
	}

	vector<Point> terminal_points;
	for(map< Point, vector< Segment > >::iterator iter = point2Seg.begin(); iter != point2Seg.end(); ++iter)
	{
		switch(iter->second.size())
		{
			case 1:
				// Endpunkt
				terminal_points.push_back(iter->first);
				break;
			case 2:
				//cout << "found connection... " << endl;
				// Verbindungspunkt
				break;
			default:
				cout << "WARNING: There are segments with zero or more then 2 connections!" << endl;
				regular = false;
				break;
		}
	}

	switch(terminal_points.size())
	{
		case 0:
			cout << "WARNING: There are no terminal points!" << endl;
			return false;
		case 1:
			cout << "WARNING: There is only one terminal point!" << endl;
			return false;
		case 2:
			cout << "Party check!" << endl;
			break;
		default:
			cout << "WARNING: There are more then two terminal points!" << endl;
			return false;
	}
	vector< Segment > sorted_segments;
	Point tmp_point = terminal_points.front();
	Point end_point = terminal_points.back();
	Segment tmp_segment = point2Seg[tmp_point].front();

	while(tmp_point != end_point)
	{
		points.push_back(tmp_point);
		Point first = tmp_segment.source();
		Point second = tmp_segment.target();
		if(POINT_DIST_EPSILON > CGAL::squared_distance(tmp_point, first))
		{
			tmp_point = second;
		}
		else
		{
			tmp_point = first;
		}
		sorted_segments.push_back(tmp_segment);
		const Segment first_seg = point2Seg[tmp_point].front();
		const Segment second_seg = point2Seg[tmp_point].back();
		
		// TODO Muss hier auch ein Epsilon verwendet werden?
		if(first_seg == tmp_segment)
		{
			tmp_segment = second_seg;
		}
		else
		{
			tmp_segment = first_seg;
		}
	}
	points.push_back(tmp_point);
	cout << "... sort done" << endl;
	segments.clear();
	segments.insert(segments.end(), sorted_segments.begin(), sorted_segments.end());
	return regular;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::getIntersectionSegments(FFace *face, vector<Segment>& segments, Tree& tree)
{
	list<Object_and_primitive_id> intersections;
	Object_and_primitive_id op;
	CGAL::Object object;

	ETriangle tri = faceToETriangle(face);
	try {
		tree.all_intersections(tri, back_inserter(intersections));
	} catch (...)
	{
		cout << "function getIntersectionPoints: tree.do_intersect() fails" << endl;
	}

	cout << intersections.size() << " intersections found" << endl;
	while (!intersections.empty()) {
		op = intersections.front();
		intersections.pop_front();
		object = op.first;
		//check wether intersection object is a segment
		Segment seg;
		Point p;
		if (CGAL::assign(seg, object)){
			segments.push_back(seg);
			//cout << "Found segment with x1: " << seg.source() << " x2: " << seg.target()  << endl;
		}
		else if (CGAL::assign(p, object)){
			cout << "intersection is a point not a segment" << endl;
		}
	}
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::getIntersectionPoints(FFace *face, vector<Point>& local_points, vector<Point>& global_points)
{
	vector<const Segment*> intersect_segments;
	list<Object_and_primitive_id> intersections;
	Object_and_primitive_id op;
	CGAL::Object object;
	Point p;

	ETriangle tri = faceToETriangle(face);
	try {
			tree.all_intersections(tri, back_inserter(intersections));
	} catch (...)
	{
		cout << "function getIntersectionPoints: tree.do_intersect() fails" << endl;
	}
	while (!intersections.empty()) {
		op = intersections.front();
		intersections.pop_front();
		object = op.first;
		//check wether intersection object is a segment
		if (const Segment* s = CGAL::object_cast<Segment>(&object)){
			intersect_segments.push_back(s);
		}
		else if (const Point* p = CGAL::object_cast<Point>(&object)){
			cout << "intersection is a point not a segment" << endl;
		}
	}
	for (int j = 0; j < intersect_segments.size(); j++) {
		const Segment* seg = intersect_segments[j];
		p = seg->source();
		//HIER ENTSTEHT BEIM HORN_MESH EIN PUNKT MIT e-316 Werten
		local_points.push_back(p);
		global_points.push_back(p);
		p = seg->target();
		local_points.push_back(p);
		global_points.push_back(p);
		//TODO intersect_segments wieder frei geben!

	}
}
	template<typename VertexT, typename NormalT> 
bool FusionMesh<VertexT, NormalT>::polygonTriangulation(Polygon& polygon, vector<Triangle2D>& triangles)
{
	ConstrainedDelaunay cdt;
	if ( polygon.is_empty() ) return false;
	//ConstrainedDelaunay::Vertex_handle v_prev=cdt.insert(*CGAL::cpp11::prev(polygon.vertices_end()));
	ConstrainedDelaunay::Vertex_handle v_prev=cdt.insert(polygon[polygon.size()-1]);
	for (Polygon::Vertex_iterator vit=polygon.vertices_begin();
			vit!=polygon.vertices_end();++vit)
	{
		ConstrainedDelaunay::Vertex_handle vh=cdt.insert(*vit);
		cdt.insert_constraint(vh,v_prev);
		v_prev=vh;
	} 
 	for (ConstrainedDelaunay::Finite_faces_iterator fit=cdt.finite_faces_begin();
			fit!=cdt.finite_faces_end();++fit)
 	{
		triangles.push_back(cdt.triangle(fit));
	}
	return true;
}
} // namespace lvr
