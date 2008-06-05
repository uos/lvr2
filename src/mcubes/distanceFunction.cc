#include "distanceFunction.h"

DistanceFunction::DistanceFunction(ANNpointArray p, int n, int k, bool tp){

  points = p;
  number_of_points = n;
  create_tangent_planes = tp;

  ofstream out("tangets.nor");
  
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

	 normals.push_back(p.getNormal());
    }
    cout << "##### Created Tangent Planes. Number of Planes: " << tangentPlanes.size() << endl;


  }

  out.close();
  
}

void DistanceFunction::distance(ColorVertex vertices[], float distances[], int k, float epsilon){

  for(int i = 0; i < 8; i++){

    ANNidxArray id = new ANNidx[1];
    ANNdistArray di = new ANNdist[1];
    ANNpoint p = annAllocPt(3);

    p[0] = vertices[i].x;
    p[1] = vertices[i].y;
    p[2] = vertices[i].z;

    point_tree->annkSearch(p, 1, id, di);

    annDeallocPt(p);

    BaseVertex nearest(points[id[0]][0],
				   points[id[0]][1],
				   points[id[0]][2]);

    BaseVertex diff = vertices[i] - nearest;
    
    float sign = diff * normals[id[0]];
    float length = diff.length();

    if(sign < 0)
	 distances[i] = - length;
    else
	 distances[i] = length;

    //cout << length << " ";
  }
  //cout << endl;
  
  
}


float DistanceFunction::distance(const BaseVertex v,
						   Normal &normal, int k, float epsilon, direction dir, bool &ok) const{

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

  ok = true;
  
  //Get nearest point
  BaseVertex nearest(points[id[0]][0],
				 points[id[0]][1],
				 points[id[0]][2]);


  //Calculate Distance
  BaseVertex diff = v - nearest;


  //Approximate plane in z-Direction
  float sign = 0.0;
  float z1 = 0.0, z2 = 0.0;
  
  BaseVertex diff1, diff2;

  if(dir == Z){
  
    try{
  
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
	 z1 = C(1) + C(2) * (nearest.x + epsilon) + C(3) * nearest.y;
	 z2 = C(1) + C(2) * nearest.x + C(3) * (nearest.y + epsilon);

		 
	 diff2 = BaseVertex(nearest.x, nearest.y + epsilon, z2) - nearest;
	 diff1 = BaseVertex(nearest.x + epsilon, nearest.y, z1) - nearest;

	 normal = diff1.cross(diff2);

	 if(v.z <= 0){
	   normal.x = -normal.x;
	   normal.y = -normal.y;
	   normal.z = -normal.z;
	 }
	 
    } catch (Exception& e){
	 //cout << e.what();
	 cout << "Singular Exception in z-Direction" << endl;
	 ok = false;
    }
    
    
    

  } else if (dir == Y){

    try{
	 ColumnVector C(3);
	 ColumnVector F(k);
	 Matrix B(k, 3);

	 for(int i = 1; i <= k; i++){
	   F(i) = points[id[i-1]][1];
	   B(i, 1) = 1;
	   B(i, 2) = points[id[i-1]][0];
	   B(i, 3) = points[id[i-1]][2];
	 }
    
	 Matrix Bt = B.t();
	 Matrix BtB = Bt * B;
	 Matrix BtBinv = BtB.i();
	 Matrix M = BtBinv * Bt;
  
	 C = M * F;

	 //Estimate surface normal
	 z1 = C(1) + C(2) * (nearest.x + epsilon) + C(3) * nearest.z;
	 z2 = C(1) + C(2) * nearest.x + C(3) * (nearest.z + epsilon);
    
	 diff1 = BaseVertex(nearest.x + epsilon, z1, nearest.z) - nearest;
	 diff2 = BaseVertex(nearest.x, z2, nearest.z + epsilon) - nearest;

    } catch (Exception& e){
	 cout << "Singular Exception in y-Direction" << endl;
	 ok = false;
    }
    
  } else {

     try{
	 ColumnVector C(3);
	 ColumnVector F(k);
	 Matrix B(k, 3);

	 for(int i = 1; i <= k; i++){
	   F(i) = points[id[i-1]][0];
	   B(i, 1) = 1;
	   B(i, 2) = points[id[i-1]][1];
	   B(i, 3) = points[id[i-1]][2];
	 }
    
	 Matrix Bt = B.t();
	 Matrix BtB = Bt * B;
	 Matrix BtBinv = BtB.i();
	 Matrix M = BtBinv * Bt;
  
	 C = M * F;

	 //Estimate surface normal
	 z1 = C(1) + C(2) * (nearest.y + epsilon) + C(3) * nearest.z;
	 z2 = C(1) + C(2) * nearest.y + C(3) * (nearest.z + epsilon);
    
	 diff1 = BaseVertex(z1, nearest.y + epsilon, nearest.z) - nearest;
	 diff2 = BaseVertex(z2, nearest.y, nearest.z + epsilon) - nearest;

    } catch (Exception& e){
	  cout << "Singular Exception in x-Direction" << endl;
	  ok = false;
    }
    
  }

  //normal = diff1.cross(diff2);
  sign = normal * diff;
  
  //Release memory
  delete[] id;
  delete[] di;



  
  //Return distance value
  if(sign > 0){
    return diff.length();
  } else {
    return - diff.length();
  }
  
}



DistanceFunction::~DistanceFunction(){
  tangentPlanes.clear();
}
