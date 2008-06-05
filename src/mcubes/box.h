#ifndef __BOX_H__
#define __BOX_H__

#include "tables.h"
#include "distanceFunction.h"
#include "interpolator.h"

#include "../mesh/colorVertex.h"
#include "../mesh/staticMesh.h"

class Box{

public:
  Box();
  Box(const Box &o);
  Box(Vertex v, float voxelsize);

  
  int getApproximation(int globalIndex,
				   StaticMesh &mesh,
				   Interpolator* df);
			    
  void setConfigurationCorner(int i) {configuration[i] = true;};
  int getIndex() const;
  ColorVertex getBaseCorner() const{ return baseVertex;};

  Box* nb[27];

  
private:

  bool sign(float v){ return v > 0;};
  
  void getCorners(ColorVertex corners[]);
  void getIntersections(ColorVertex corners[],
				    Interpolator* df,
				    ColorVertex intersections[]);

  float calcIntersection(float x1, float x2,
					float v1, float v2, bool interpolate);

  void setColor(uchar r, uchar g, uchar b){
    current_color[0] = r;
    current_color[1] = g;
    current_color[2] = b;
  };

  float distance[8];
  bool configuration[8];
  int indices[12];

  uchar current_color[3];

  bool approx_ok;
  
  float voxelsize;
  ColorVertex baseVertex;
  
};

inline ostream& operator<<(ostream& os, const Box b){
  os << "## Box: ## " << endl
	<< b.getBaseCorner()
	<< "Index: " << b.getIndex() << endl << endl; 
  return os;
}

#endif
