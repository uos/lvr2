#include "distanceFunction.h"

DistanceFunction::DistanceFunction(ANNpointArray p, int n, int k, bool tp){

  points = p;
  number_of_points = n;
  create_tangent_planes = tp;

  cout << "##### Creating kd-Tree containing data points..." << endl;

  //Create kd-tree with data points
  ANNsplitRule split = ANN_KD_SUGGEST;
  point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);
  cout << "##### Done. " << endl;

  if(create_tangent_planes){
    for(int i = 0; i < number_of_points; i++){
	 if(i % 10000 == 0) cout << "##### Creating Tangent Planes " << i << endl;
	 BaseVertex v = BaseVertex(points[i][0],
						  points[i][1],
						  points[i][2]);
	 TangentPlane p(v, points, point_tree, k);
	 tangentPlanes.push_back(p);
    }
    cout << "##### Created Tangent Planes. Number Planes: " << tangentPlanes.size() << endl;

    //Create kd-tree with tangent plane centers
    cout << "##### Creating kd-Tree containing tangent plane centers..." << endl;
    tp_centers = annAllocPts(n, 3);
    for(size_t i = 0; i < tangentPlanes.size(); i++){
	 BaseVertex c = tangentPlanes[i].getCenter();
	 tp_centers[i][0] = c.x;
	 tp_centers[i][1] = c.y;
	 tp_centers[i][2] = c.z;
    }
    tp_tree = new ANNkd_tree(tp_centers, number_of_points, 3, 10, split);
  }
}

float DistanceFunction::distance(const BaseVertex v) const{

  int k = 1;

  //Get id of nearest tangent plane
  ANNidxArray id = new ANNidx[k];
  ANNdistArray di = new ANNdist[k];
  ANNpoint p = annAllocPt(3);

  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;

  if(create_tangent_planes)
    tp_tree->annkSearch(p, 1, id, di);
  else
    point_tree->annkSearch(p, 1, id, di);

  annDeallocPt(p);

  BaseVertex nearest(points[id[0]][0],
				 points[id[0]][1],
				 points[id[0]][2]);

  BaseVertex diff = v - nearest;

  delete[] id;
  delete[] di;
  
  return diff.length();
  
}

DistanceFunction::~DistanceFunction(){
  tangentPlanes.clear();
}
