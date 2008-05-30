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

  int k = 10;

  //Get id of nearest tangent plane
  ANNidxArray id = new ANNidx[k];
  ANNdistArray di = new ANNdist[k];
  ANNpoint p = annAllocPt(3);

  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;

  if(create_tangent_planes)
    tp_tree->annkSearch(p, k, id, di);
  else
    point_tree->annkSearch(p, k, id, di);

  annDeallocPt(p);

  //Get nearest point
  BaseVertex nearest(points[id[0]][0],
				 points[id[0]][1],
				 points[id[0]][2]);


  //Calculate Distance
  BaseVertex diff = v - nearest;


  //Approximate plane
  ColumnVector C(3);
  ColumnVector F(k);
  Matrix B(k, 3);

  for(int i = 1; i <= k; i++){
    F(i) = points[id[i-1]][2];
    B(i, 1) = 1;
    B(i, 2) = points[id[i-1]][0];
    B(i, 3) = points[id[i-1]][1];
  }

  Matrix Bt = B.t();
  Matrix BtB = Bt * B;
  Matrix BtBinv = BtB.i();
  Matrix M = BtBinv * Bt;
  
  C = M * F;

  
  //Estimate surface normal
  //f(x) = c1 + c2 * x + c3 * y -> z

  float z1 = C(1) + C(2) * (nearest.x + 0.3) + C(3) * nearest.y;
  float z2 = C(1) + C(2) * nearest.x + C(3) * (nearest.y + 0.3);

  BaseVertex diff1 = BaseVertex(nearest.x + 0.3, nearest.y, z1) - nearest;
  BaseVertex diff2 = BaseVertex(nearest.x, nearest.y + 0.3, z2) - nearest;

  Normal normal = diff1.cross(diff2);

  float sign = normal * diff;

  //cout << "SIGN: " << sign << endl;
  
  //Release memory
  delete[] id;
  delete[] di;

  if(sign > 0) 
    return diff.length();
  else
    return - diff.length();
  
}

int DistanceFunction::getSign(BaseVertex v, int k){

  //We need at least 3 points
  if(k < 3) k = 3;

  const float epsilon = 5.0;
  
  ColumnVector C(4);
  ColumnVector F(3 * k);
  Matrix B(3 * k, 4);

  Normal normals[k];
  BaseVertex nearest_points[k];
 

  ANNidxArray id = new ANNidx[k];
  ANNdistArray di = new ANNdist[k];
  ANNpoint p = annAllocPt(3);

  //Get k nearest points
  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;
  point_tree->annkSearch(p, k, id, di);

  //Create points
  for(int i = 0; i < k; i++){
    nearest_points[i] = BaseVertex(points[id[i]][0],
							points[id[i]][1],
							points[id[i]][2]);
  }


  for(int i = 0; i < k; i++){
    TangentPlane tangentPlane(v, points, point_tree, k);
    normals[i] = tangentPlane.getNormal();
    //cout << normals[i];
  }

  
  
  //Build equation system
  for(int i = 0; i < k; i++){

    F(3 * i + 1) = 0;
    F(3 * i + 2) =  epsilon;
    F(3 * i + 3) = -epsilon;
    
    B(3 * i + 1, 1) = 1;
    B(3 * i + 1, 2) = nearest_points[i].x;
    B(3 * i + 1, 3) = nearest_points[i].y;
    B(3 * i + 1, 4) = nearest_points[i].z;

    B(3 * i + 2, 1) = 1;
    B(3 * i + 2, 2) = epsilon * normals[i].x;
    B(3 * i + 2, 3) = epsilon * normals[i].y;
    B(3 * i + 2, 4) = epsilon * normals[i].z;

    B(3 * i + 3, 1) = 1;
    B(3 * i + 3, 2) = -epsilon * normals[i].x;
    B(3 * i + 3, 3) = -epsilon * normals[i].y;
    B(3 * i + 3, 4) = -epsilon * normals[i].z;
    
  }
  
  
  //Solve equation B * c = F
  Matrix Bt = B.t();
  Matrix BtB = Bt * B;
  Matrix BtBinv = BtB.i();
  Matrix M = BtBinv * Bt;
  
  C = M * F;

  //cout << C(1) << " " << C(2) << " " << C(3) << " " << C(4) << endl;
  
  float f = C(1) + C(2) * v.x + C(3) * v.y + C(4) * v.z;

  cout << f << endl;
  
//   delete[] id;
//   delete[] di;

  if(f > 0) return 1; else return -1;

  //return -1;
}

DistanceFunction::~DistanceFunction(){
  tangentPlanes.clear();
}
