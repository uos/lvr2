/*
 * FastGrid.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef FASTGRID_H_
#define FASTGRID_H_

#include <vector>
#include <ext/hash_map>

using namespace std;
using __gnu_cxx::hash_map;

#include <ANN/ANN.h>

#include <lib3d/BaseVertex.h>
#include <lib3d/BoundingBox.h>
#include <lib3d/TriangleMesh.h>
#include <lib3d/HalfEdgeMesh.h>
#include <lib3d/LinkedTriangleMesh.h>
#include <lib3d/ProgressiveMesh.h>

#include "ANNInterpolator.h"
#include "Interpolator.h"
#include "StannInterpolator.h"
#include "QueryPoint.h"
#include "FastBox.h"
#include "Tables.h"

class FastGrid {
public:
	FastGrid(string filename, float voxelsize);

	void writeGrid();

	virtual ~FastGrid();

private:

	inline int hashValue(int i, int j, int k);
	inline int calcIndex(float f);

	int  findQueryPoint(int Position, int index_x, int index_y, int index_z);

	void readPoints(string filename);
	void readPlainASCII(string filename);
	void readPLY(string filename);

	void calcIndices();
	void calcQueryPointValues();
	void createGrid();
	void createMesh();

	int  getFieldsPerLine(string filename);

	float                   voxelsize;
	int                     number_of_points;
	int                     max_index;
    int                     max_index_square;
	int                     max_index_x;
	int                     max_index_y;
	int                     max_index_z;

	BoundingBox             bounding_box;
	vector<QueryPoint>      query_points;
	float**          		points;
	Interpolator*           interpolator;
	HalfEdgeMesh*     		mesh;
	HalfEdgeMesh            he_mesh;

	hash_map<int, FastBox*> cells;

};

inline int FastGrid::hashValue(int i, int j, int k){
  return i * max_index_square + j * max_index + k;
}

inline int FastGrid::calcIndex(float f){
  return f < 0 ? f-.5:f+.5;
}


#endif /* FASTGRID_H_ */
