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

#include "../model3d/BaseVertex.h"
#include "../model3d/BoundingBox.h"
#include "../model3d/TriangleMesh.h"
#include "../model3d/HalfEdgeMesh.h"
#include "../model3d/LinkedTriangleMesh.h"
#include "../model3d/ProgressiveMesh.h"

#include "Interpolator.h"
#include "StannInterpolator.h"
#include "QueryPoint.h"
#include "FastBox.h"
#include "Tables.h"
#include "Options.h"

class FastGrid {
public:
	FastGrid(Options* options);

	void writeGrid();

	virtual ~FastGrid();

private:

	inline int hashValue(int i, int j, int k);
	inline int calcIndex(float f);

	int  findQueryPoint(int Position, int index_x, int index_y, int index_z);

	void readPoints(string filename);
	void readPlainASCII(string filename);
	void readPLY(string filename);

	void savePointsAndNormals();

	void calcIndices();
	void calcQueryPointValues();
	void createGrid();
	void createMesh();

	int  getFieldsPerLine(string filename);

	float                   voxelsize;
	size_t                  number_of_points;
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

	Options* 				m_options;

};

inline int FastGrid::hashValue(int i, int j, int k){
  return i * max_index_square + j * max_index + k;
}

inline int FastGrid::calcIndex(float f){
  return f < 0 ? f-.5:f+.5;
}


#endif /* FASTGRID_H_ */
