#include "interpolator.h"

Interpolator::Interpolator(ANNpointArray pts, int n,
					  int k_initial, int k_interpolate, int epsilon){

  number_of_points = n;
  points = pts;
  
  //Build kd tree with data points
  cout << "##### Building KD-Tree with all data points..." <<  endl;
  ANNsplitRule split = ANN_KD_SUGGEST;
  point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

  cout << "##### Calculating initial normal estimations..." << endl;

  //Create initial normal estimation

  float z1 = 0.0, z2 = 0.0;

  Vertex diff1, diff2, point;
  Normal normal;
  
  ColumnVector C(3);
  ColumnVector F(k_initial);
  Matrix B(k_initial, 3);

  ANNidxArray id = new ANNidx[k_initial];
  ANNdistArray di = new ANNdist[k_initial];

  normals = new Normal[number_of_points];

  int n_pts_ret = 0;
  
  for(int i = 0; i < number_of_points; i++){

    //Create vertex representation of current point
    point = Vertex(points[i][0], points[i][1], points[i][2]);

    //Get k_initial nearest points
    point_tree->annkFRSearch(points[i], 100.0,  n_pts_ret, id, di, 0.0);
    cout << n_pts_ret << endl;
  
   //  //Build equation system
//     for(int j = 1; j <= k_initial; j++){
// 	 F(j) = points[id[j-1]][2];
// 	 B(j, 1) = 1;
// 	 B(j, 2) = points[id[j-1]][0];
// 	 B(j, 3) = points[id[j-1]][1];
//     }

//     try{
//     //Solve it
//     Matrix Bt = B.t();
//     Matrix BtB = Bt * B;
//     Matrix BtBinv = BtB.i();
//     Matrix M = BtBinv * Bt;
	 
//     C = M * F;

//     //Estimate surface normal
//     z1 = C(1) + C(2) * (point.x + epsilon) + C(3) * point.y;
//     z2 = C(1) + C(2) * point.x + C(3) * (point.y + epsilon);
		 
//     diff2 = BaseVertex(point.x, point.y + epsilon, z2) - point;
//     diff1 = BaseVertex(point.x + epsilon, point.y, z1) - point;

//     normal = diff1.cross(diff2);

//     if(points[i][2] <= 0){
// 	 normal.x = -normal.x;
// 	 normal.y = -normal.y;
// 	 normal.z = -normal.z;
//     }
    
//     //normals.push_back(normal);
//     normals[i] = normal;
//     } catch (Exception& e){
	 
//     }
    
//     if(i % 10000 == 0) cout << "##### Estimating normals..." << i << " / " << number_of_points << endl;
  
  }
 
  
  cout << "##### Interpolating normals..." << endl;

 //  float x, y, z;
  
//   for(int i = 0; i < number_of_points; i++){

//     x = y = z = 0.0;
    
//     point_tree->annkPriSearch(points[i], k_interpolate, id, di);
    
//     for(int j = 0; j < k_interpolate; j++){
// 	 x += normals[id[k_interpolate]].x;
// 	 y += normals[id[k_interpolate]].y;
// 	 z += normals[id[k_interpolate]].z;
// 	 normals[id[k_interpolate]] = Normal(x, y, z);
//     }

//     if(i % 10000 == 0) cout << "##### Interpolating normals..." << i << " / " << number_of_points << endl;
    
//   }

  

}

float Interpolator::distance(ColorVertex v){

  ANNidxArray id = new ANNidx[1];
  ANNdistArray di = new ANNdist[1];
  ANNpoint p = annAllocPt(3);

  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;

  point_tree->annkSearch(p, 1, id, di);

  Vertex nearest = Vertex(points[id[0]][0],
					 points[id[0]][1],
					 points[id[0]][2]);

  Vertex diff = v - nearest;

  float length = diff.length();
  float sign = diff * normals[id[0]];

  if(sign >= 0)
    return length;
  else
    return -length;
  
}

void Interpolator::write(string filename){

  ofstream out(filename.c_str());

  if(!out.good()){
    cout << "Warning: Interpolator: could not open file " << filename << "." << endl;
    return;
  }

  cout << "##### Writing '" << filename << "'..." << endl;
  for(int i = 0; i < number_of_points; i++){

    out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
	   << normals[i].x << " " << normals[i].y << " " << normals[i].z << endl;
    
  }
  
}


