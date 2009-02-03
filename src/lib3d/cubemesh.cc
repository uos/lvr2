// ******************************
// mesh.cpp
//
// Mesh class, which stores a list
// of vertices & a list of triangles.
//
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

#include <assert.h>
#include <float.h>

#if defined (_MSC_VER) && (_MSC_VER >= 1020)
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#endif

#include <iostream>

#include "cubemesh.h"


CubeMesh::CubeMesh()
{
	_numVerts = _numTriangles = 0;
	_vlist.clear();
	_plist.clear();
}

CubeMesh::CubeMesh(const CubeMesh& m)
{
	_numVerts = m._numVerts;
	_numTriangles = m._numTriangles;
	_vlist = m._vlist; // NOTE: triangles are still pointing to original mesh
	_plist = m._plist;
	// NOTE: should reset tris in _vlist, _plist
}

CubeMesh& CubeMesh::operator=(const CubeMesh& m)
{
	if (this == &m) return *this; // don't assign to self
	_numVerts = m._numVerts;
	_numTriangles = m._numTriangles;
	_vlist = m._vlist; // NOTE: triangles are still pointing to original mesh
	_plist = m._plist;
	// NOTE: should reset tris in _vlist, _plist
	return *this;
}

CubeMesh::~CubeMesh()
{
	_numVerts = _numTriangles = 0;
	_vlist.erase(_vlist.begin(), _vlist.end());
	_plist.erase(_plist.begin(), _plist.end());
}

/**
 * add a vertex to the mesh
 */
bool CubeMesh::addVertex(float x,float y,float z)
{
  //cout<<"add vertex ("<<x<<","<<y<<","<<z<<")"<<endl;
  static int i = 0;
  vertex v(x, y, z);
  v.setIndex(i);
  _vlist.push_back(v); // push_back puts a *copy* of the element at the end of the list
  i++;
  _numVerts = i;
  return true;
}
/**
 * add a vertex to the mesh
 * v1 , v2 and v3 are the indices in the vertex list _vlist 
 */
bool CubeMesh::addTriangle(int v1,int v2,int v3)
{
  static int i = 0;
 
  // make sure verts in correct range
  assert(v1 < _numVerts && v2 < _numVerts && v3 < _numVerts);
  
  triangle t(this, v1, v2, v3);
  t.setIndex(i);
  //cout<<"add triangle index ("<<v1<<","<<v2<<","<<v3<<")"<<t<<endl; 
  _plist.push_back(t); // push_back puts a *copy* of the element at the end of the list
  
  // update each vertex w/ its neighbors (vertrices & triangles)
  _vlist[v1].addTriNeighbor(i);
  _vlist[v1].addVertNeighbor(v2);
  _vlist[v1].addVertNeighbor(v3);
  
  _vlist[v2].addTriNeighbor(i);
  _vlist[v2].addVertNeighbor(v1);
  _vlist[v2].addVertNeighbor(v3);
  
  _vlist[v3].addTriNeighbor(i);
  _vlist[v3].addVertNeighbor(v1);
  _vlist[v3].addVertNeighbor(v2);
  _numTriangles = i;
  i++;
  return true;
}
/**
 *
 */
bool CubeMesh::rotateMesh(float x,float y, float z)
{
  for(int i = 0; i <_numTriangles; i++) {
    _plist[i].rotate(x,y,z);
  }
  return true;
}
bool CubeMesh::closeHoles(int vertNumber)
{
  set<border> borderSet;
  _vlist[0].getAllBorderEdges(borderSet, *this);
  set<int>::iterator pos;
  /*  for (pos = borderSet.begin(); pos != borderSet.end(); ++pos){
    if(
  */
  /*
  vertex vert; 
  for(int i=0;i<_numVerts;i++){
    if(_vlist[i].isBorder() && _vlist[i].isActive()&& !_vlist[i].closed ) { //is there a active border vertex 
	 closeBorder(_vlist[i],_vlist[i].getIndex(),_vlist[i].getIndex());
    }
  }
  */
  return true;
}
bool CubeMesh::closeBorder(vertex vert,int lastIndex,int start) {
  /*
  set<int>::iterator pos;
  for (pos = vert.getVertNeighbors().begin(); pos != vert.getVertNeighbors().end(); ++pos){
    if(*pos.isBorder() && *pos.isActive() && *pos.getIndex!=lastIndex) { //is there a active neighbor border vertex thats not the vertex which calls the function
	 if(*pos.getIndex==start){ //loop closed
	   index_store.push_back(*pos.getIndex);
	   return true;
	 }
	 else {
	   if(closeBorder(*pos,*pos.getIndex,start)) { //recursive function
		index_store.push_back(*pos.getIndex);
		*pos.closed=true;
		return true;
	   }
	   else {
		return false;
	   }
	 }
    }
  }
  */
  return false;
}

// Recalculate the normal for one vertex
void CubeMesh::calcOneVertNormal(unsigned vert)
{
	vertex& v = getVertex(vert);
	const set<int>& triset = v.getTriNeighbors();
	set<int>::iterator iter;
	Vec3 vec;

	for (iter = triset.begin(); iter != triset.end(); ++iter)
	{
	  // get the triangles for each vertex & add up the normals.
		vec += getTri(*iter).getNormalVec3();
	}
	vec.normalize(); // normalize the vertex	
	v.setVertNomal(vec);
}


// Calculate the vertex normals after loading the mesh.
void CubeMesh::calcVertNormals()
{
	// Iterate through the vertices
	for (unsigned i = 0; i < _vlist.size(); ++i)
	{
		calcOneVertNormal(i);
	}
}


// Used for debugging
void CubeMesh::dump()
{
	std::cout << "*** Mesh Dump ***" << std::endl;
	std::cout << "# of vertices: " << _numVerts << std::endl;
	std::cout << "# of triangles: " << _numTriangles << std::endl;
	for (unsigned int i = 0; i < _vlist.size(); ++i)
	{
		std::cout << "\tVertex " << i << ": " << _vlist[i] << std::endl;
	}
	std::cout << std::endl;
	for (unsigned int j = 0; j < _plist.size(); ++j)
	{
		std::cout << "\tTriangle " << j << ": " << _plist[j] << std::endl;
	}
	std::cout << "*** End of Mesh Dump ***" << std::endl;
	std::cout << std::endl;
}

// Get min, max values of all verts
void CubeMesh::setMinMax(float min[3], float max[3])
{
	max[0] = max[1] = max[2] = -FLT_MAX;
	min[0] = min[1] = min[2] = FLT_MAX;

	for (unsigned int i = 0; i < _vlist.size(); ++i)
	{
		const float* pVert = _vlist[i].getArrayVerts();
		if (pVert[0] < min[0]) min[0] = pVert[0];
		if (pVert[1] < min[1]) min[1] = pVert[1];
		if (pVert[2] < min[2]) min[2] = pVert[2];
		if (pVert[0] > max[0]) max[0] = pVert[0];
		if (pVert[1] > max[1]) max[1] = pVert[1];
		if (pVert[2] > max[2]) max[2] = pVert[2];
	}
}

// Center mesh around origin.
// Fit mesh in box from (-1, -1, -1) to (1, 1, 1)
void CubeMesh::Normalize()  
{
	float min[3], max[3], Scale;

	setMinMax(min, max);

	Vec3 minv(min);
	Vec3 maxv(max);

	Vec3 dimv = maxv - minv;
	
	if (dimv.x >= dimv.y && dimv.x >= dimv.z) Scale = 2.0f/dimv.x;
	else if (dimv.y >= dimv.x && dimv.y >= dimv.z) Scale = 2.0f/dimv.y;
	else Scale = 2.0f/dimv.z;

	Vec3 transv = minv + maxv;

	transv *= 0.5f;

	for (unsigned int i = 0; i < _vlist.size(); ++i)
	{
		_vlist[i].getXYZ() -= transv;
		_vlist[i].getXYZ() *= Scale;
	}
}

// bool CubeMesh::toDXF(char *dir)
// {
//   //open dxfstream
//   vertex dummyVert;
//   Vec3 vec3vert1,vec3vert2,vec3vert3;
//   ofstream dxfstream;
//   char fname[64];
//   strcpy(fname,dir);
//   strcat(fname, "/simpleMesh.dxf");
//   dxfstream.open(fname,ios::out);
//   if (!dxfstream) {
//     cerr << "*** error: can't create file " << fname << " ***\n";
//     exit(2);
//   }
//   outputDXFHeader(dxfstream);
//   for(unsigned int i = 0; i < _plist.size() ; i++) {
//     if(!_plist[i].isActive()) {
// 	 continue;
//     }
//     if(_plist[i].calcArea() <= 5) {
// 	 // cout<<"dxf output: triangle area less then zero"<<endl;
// 	 continue;
//     }
//     dummyVert = _vlist[_plist[i].getVert1Index()];
//     vec3vert1 = dummyVert.getXYZ();
//     dummyVert = _vlist[_plist[i].getVert2Index()];
//     vec3vert2 = dummyVert.getXYZ();
//     dummyVert = _vlist[_plist[i].getVert3Index()];
//     vec3vert3 = dummyVert.getXYZ();
//     if(vec3vert1.x == vec3vert2.x && vec3vert1.y == vec3vert2.y && vec3vert1.z == vec3vert2.z ||
// 	  vec3vert2.x == vec3vert3.x && vec3vert2.y == vec3vert3.y && vec3vert2.z == vec3vert3.z ||
// 	  vec3vert3.x == vec3vert1.x && vec3vert3.y == vec3vert1.y && vec3vert3.z == vec3vert1.z)
// 	 {
// 	   //cout<<"dxf output: two identical points in simple triangle continue index "<<i<<endl;
// 	 continue;
//     }
    
//     dxfstream << "0"  << endl << "3DFACE" << endl;
//     dxfstream << "8"  << endl << "Cube2" << endl; // kommt von oben
//     dxfstream << "62" << endl << 0 << endl; // 1 ist color index
//     dxfstream << "10" << endl << vec3vert1.x << endl;
//     dxfstream << "20" << endl << vec3vert1.y << endl;
//     dxfstream << "30" << endl << vec3vert1.z << endl;
//     dxfstream << "11" << endl << vec3vert2.x << endl;
//     dxfstream << "21" << endl << vec3vert2.y << endl;
//     dxfstream << "31" << endl << vec3vert2.z << endl;
//     dxfstream << "12" << endl << vec3vert3.x << endl;
//     dxfstream << "22" << endl << vec3vert3.y << endl;
//     dxfstream << "32" << endl << vec3vert3.z << endl;
//     dxfstream << "13" << endl << vec3vert3.x << endl;
//     dxfstream << "23" << endl << vec3vert3.y << endl;
//     dxfstream << "33" << endl << vec3vert3.z << endl;
//   }
//   dxfstream << "0" << endl << "ENDSEC" << endl;
//   dxfstream << "0" << endl << "EOF" << endl;
//   dxfstream.close();
//   return true;
// }
