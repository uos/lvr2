#include "kdppInterpolator.h"

KDPPInterpolator::KDPPInterpolator(ANNpointArray pts, int n, float vs, int k_max, float epsilon){

  voxelsize = vs;
  points = pts;
  number_of_points = n;

  t = new tree_type(std::ptr_fun(tac));

  cout << "##### Building kd-Tree using libkd+++" << endl;
  for(int i = 0; i < n; i++){
    IdPoint p;
    p.p = points[i];
    p.id = i;
    t->insert(p);
  }
  
  cout << "##### Optimizing kd-Tree" << endl;
  t->optimize();

  vector<IdPoint> points_in_range;

  float z1 = 0.0, z2 = 0.0;

  Vertex diff1, diff2, point;
  Normal normal;

  normals = new Normal[number_of_points];
  
  for(int i = 0; i < n; i++){

    if(i % 10000 == 0) cout << "Estimating Normals... " << i << " / " << n << endl;

    IdPoint p;
    p.p = points[i];
    p.id = i;
    
    t->find_within_range(p, voxelsize, back_inserter(points_in_range));

    int n_in_range = (int)points_in_range.size();

    ColumnVector C(3);
    ColumnVector F(n_in_range);
    Matrix B(n_in_range, 3);

    for(size_t j = 1; j <= points_in_range.size(); j++){
	 F(j) = points_in_range[j-1][2];
	 B(j, 1) = 1;
	 B(j, 2) = points_in_range[j-1][0];
	 B(j, 3) = points_in_range[j-1][1];
    }

    try{
	 //Solve it
	 Matrix Bt = B.t();
	 Matrix BtB = Bt * B;
	 Matrix BtBinv = BtB.i();
	 Matrix M = BtBinv * Bt;
	 
	 C = M * F;

	 //Estimate surface normal
	 z1 = C(1) + C(2) * (point.x + epsilon) + C(3) * point.y;
	 z2 = C(1) + C(2) * point.x + C(3) * (point.y + epsilon);
		 
	 diff2 = BaseVertex(point.x, point.y + epsilon, z2) - point;
	 diff1 = BaseVertex(point.x + epsilon, point.y, z1) - point;

	 normal = diff1.cross(diff2);

	 //cout << normal;
	 
	 if(points[i][2] <= 0){
	   normal.x = -normal.x;
	   normal.y = -normal.y;
	   normal.z = -normal.z;
	 }
    
	 //normals.push_back(normal);
	 normals[i] = normal;
    } catch (Exception& e){
	 
    }
    
    points_in_range.clear(); 
  }
  
  cout << "##### Interpolating normals..." << endl;

  float x, y, z;
  
  for(int i = 0; i < number_of_points; i++){
    
    x = y = z = 0.0;
    
    IdPoint p;
    p.p = points[i];
    p.id = i;
    
    t->find_within_range(p, voxelsize, back_inserter(points_in_range));
    
    for(size_t j = 0; j < points_in_range.size() ; j++){
	 int id = points_in_range[j].id;
	 x += normals[id].x;
	 y += normals[id].y;
	 z += normals[id].z;
    }

    normals[i] = Normal(x, y, z);

    points_in_range.clear();
    
    if(i % 10000 == 0) cout << "##### Interpolating normals..." << i << " / " << number_of_points << endl;
    
  }

//   cout << "DONE" << endl;

}

float KDPPInterpolator::distance(ColorVertex v){

  ANNpoint s;
  s = annAllocPt(3);
  s[0] = v.x;
  s[1] = v.y;
  s[2] = v.z;

  IdPoint idp;
  idp.p = s;
  idp.id = 0; 
  
  //IdPoint nb = *t->find_nearest(idp,std::numeric_limits<double>::max()).first;

  //cout << "ID: " << nb.id << endl;


  vector<IdPoint> nb;
  t->find_within_range(idp, voxelsize, back_inserter(nb));

  if(nb.size() == 0) cout << "No NBs found!" << endl;

  float  x = 0.0,  y = 0.0,  z = 0.0;
  float nx = 0.0, ny = 0.0, nz = 0.0;
  
  for(size_t i = 0; i < nb.size(); i++){
    ANNpoint annp = nb[i].p;
    int id = nb[i].id;
    x += annp[0];
    y += annp[1];
    z += annp[2];

    Normal n = normals[id];
    nx += n.x;
    ny += n.y;
    nz += n.z;
  }

  if(nb.size() > 0){
    x /= nb.size();
    y /= nb.size();
    z /= nb.size();
  }
  
  Normal normal(nx, ny, nz);
  Vertex nearest(x, y, z);
  
 
  Vertex diff = v - nearest;

  float length = diff.length();
  float sign = diff * normal;

  if(sign >= 0)
    return length;
  else
    return -length;

  return 0.0;
  
}

void KDPPInterpolator::write(string filename){

  ofstream out(filename.c_str());

  if(!out.good()){
    cout << "Warning: KDPPInterpolator: could not open file " << filename << "." << endl;
    return;
  }

  cout << "##### Writing '" << filename << "'..." << endl;
  for(int i = 0; i < number_of_points; i++){

    out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
	   << normals[i].x << " " << normals[i].y << " " << normals[i].z << endl;
    
  }
  
}


