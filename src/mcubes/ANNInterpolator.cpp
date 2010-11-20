#include "ANNInterpolator.h"

ANNInterpolator::ANNInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon){

  k_max = km;
  points = pts;
  number_of_points = n;
  voxelsize = vs;
  vs_sq = 0.5f * vs * vs;

  normals = new float*[number_of_points];
  for(size_t i = 0; i < number_of_points; i++) normals[i] = new float[3];

  float z1, z2;

  BaseVertex query_point, diff1, diff2, center;
  Normal normal;

  ColumnVector C(3);

  center = BaseVertex(0.0, 0.0, 0.0);

  cout << "##### Creating ANN Kd-Tree..." << endl;
  ANNsplitRule split = ANN_KD_SUGGEST;
  point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

  ANNidxArray id = new ANNidx[k_max];
  ANNdistArray di = new ANNdist[k_max];

  cout << "##### Estimating normals..." << endl;

  for(size_t i = 0; i < number_of_points; i++){

    query_point = BaseVertex((float)points[i][0],
							 (float)points[i][1],
							 (float)points[i][2]);

    center += query_point;

    point_tree->annkFRSearch(points[i], vs_sq, k_max, id, di, 0.01 * voxelsize);

    int n_nb = 0;
    for(int j = 0; j < k_max; j++){
	 if(id[j] != ANN_NULL_IDX) n_nb++; else break;
    }

    if (n_nb == 0) cout << "OKOKOKOK" << endl;

    try{
	 ColumnVector F(n_nb);
	 Matrix B(n_nb, 3);

	 for(int j = 1; j <= n_nb; j++){

	   F(j) = points[id[j-1]][1];
	   B(j, 1) = 1;
	   B(j, 2) = points[id[j-1]][0];
	   B(j, 3) = points[id[j-1]][2];

	 }

	 Matrix Bt = B.t();
	 Matrix BtB = Bt * B;
	 Matrix BtBinv = BtB.i();
	 Matrix M = BtBinv * Bt;

	 C = M * F;

	 z1 = (float)(C(1) + C(2) * (query_point.x + epsilon) + C(3) * query_point.z);
	 z2 = (float)(C(1) + C(2) * query_point.x + C(3) * (query_point.z + epsilon));

	 diff1 = BaseVertex(query_point.x + epsilon, z1, query_point.z) - query_point;
	 diff2 = BaseVertex(query_point.x, z2, query_point.z + epsilon) - query_point;

	 normal = diff1.cross(diff2);

// 	 if(normal * query_point > 0){
// 	   normal.x = -1 * normal.x;
// 	   normal.y = -1 * normal.y;
// 	   normal.z = -1 * normal.z;
// 	 }


	 normals[i][0] = normal[0];
	 normals[i][1] = normal[1];
	 normals[i][2] = normal[2];

    } catch(Exception e){
	 //Ignore
		cout << e.what();
    }

    if(i % 1000 == 0) cout << "##### Estimating normals... " << i << " / " << n << endl;
  }

  center.x /= number_of_points;
  center.y /= number_of_points;
  center.z /= number_of_points;

  cout << "##### Interpolating Normals... " << endl;

  for(size_t i = 0; i < number_of_points; i++){

    point_tree->annkFRSearch(points[i], vs_sq, k_max, id, di, 0.01 * voxelsize);
    int n_nb = 0;
    for(int j = 0; j < k_max; j++){
	 if(id[j] != ANN_NULL_IDX) n_nb++; else break;
    }

    float x = 0, y = 0, z = 0;

    for(int j = 0; j < n_nb; j++){
	 x += normals[id[j]][0];
	 y += normals[id[j]][1];
	 z += normals[id[j]][2];
    }

    normal = Normal(x, y, z);

    if(normal * center < 0){
	 normal.x = -1 * normal.x;
	 normal.y = -1 * normal.y;
	 normal.z = -1 * normal.z;
    }

    normals[i][0] = normal[0];
    normals[i][1] = normal[1];
    normals[i][2] = normal[2];



    if(i % 10000 == 0) cout << "##### Interpolating normals... " << i << " / " << n << endl;
  }

  delete[] di;
  delete[] id;

  write("normals.nor");

}

float** ANNInterpolator::getNormals(size_t &n)
{
	n = number_of_points;
	return normals;
}

float ANNInterpolator::distance(ColorVertex v){

  // ANNpoint s;
//   s = annAllocPt(3);
//   s[0] = v.x;
//   s[1] = v.y;
//   s[2] = v.z;

//   IdPoint idp;
//   idp.p = s;
//   idp.id = 0;

//   //IdPoint nb = *t->find_nearest(idp,std::numeric_limits<double>::max()).first;

//   //cout << "ID: " << nb.id << endl;

  ANNpoint p;
  p = annAllocPt(3);
  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;

  ANNidxArray id = new ANNidx[k_max];
  ANNdistArray di = new ANNdist[k_max];

  float radius = vs_sq;
  int n_nb = 0;
  int c = 0;

  do{
    n_nb = 0;
    radius = radius * 2;
    point_tree->annkFRSearch(p, radius, k_max, id, di, 0.01 * voxelsize);

    for(int j = 0; j < k_max; j++){
	 if(id[j] != ANN_NULL_IDX) n_nb++; else break;
    }
    c++;
  } while(n_nb < 3);

  float  x = 0.0,  y = 0.0,  z = 0.0;
  float nx = 0.0, ny = 0.0, nz = 0.0;

  for(int i = 0; i < n_nb; i++){
    Normal n = Normal(normals[id[i]][0], normals[id[i]][1], normals[id[i]][2]);
    nx += n.x;
    ny += n.y;
    nz += n.z;

    x += (float)points[id[i]][0];
    y += (float)points[id[i]][1];
    z += (float)points[id[i]][2];
  }

  if(n_nb > 0){
    x /= n_nb;
    y /= n_nb;
    z /= n_nb;
  }

  delete[] id;
  delete[] di;

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

void ANNInterpolator::write(string filename){

  ofstream out(filename.c_str());

  if(!out.good()){
    cout << "Warning: ANNInterpolator: could not open file " << filename.c_str() << "." << endl;
    return;
  }

  cout << "##### Writing '" << filename.c_str() << "'..." << endl;
  for(size_t i = 0; i < number_of_points; i++){

    out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
	   << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << endl;

  }

}


