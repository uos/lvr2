#include "tangentPlane.h"

TangentPlane::TangentPlane(BaseVertex v, ANNpointArray points, ANNkd_tree* kd_tree, int k){

  //Create reference point
  ANNpoint p = annAllocPt(3);
  p[0] = v.x;
  p[1] = v.y;
  p[2] = v.z;

  //Create Arrays for point indices and distances
  ANNidxArray indices = new ANNidx[k];
  ANNdistArray distances = new ANNdist[k];

  //Find the k nearest neighbours
  kd_tree->annkSearch(p, k, indices, distances, 0.0);

  //Calc centroid vertex
  BaseVertex centroid;
  for(int i = 0; i < k; i++){
    BaseVertex current_vertex = BaseVertex(points[indices[i]][0],
								   points[indices[i]][1],
								   points[indices[i]][2]);
    centroid += current_vertex;
  }
  centroid /= k;

  //Find nearest point to centroid -> center
  float min_distance = 1e15;
  float current_distance = 0.0;

  for(int i = 0; i < k; i++){
    BaseVertex current_vertex = BaseVertex(points[indices[i]][0],
								   points[indices[i]][1],
								   points[indices[i]][2]);

    BaseVertex diff = current_vertex - centroid;
    current_distance = sqrt(diff.length());
    if(current_distance < min_distance){
	 min_distance = current_distance;
	 center = current_vertex;
    }
  }

  //Calculate normal
  try{

    float epsilon = 10.0;
    
    ColumnVector C(3);
    ColumnVector F(k);
    Matrix B(k, 3);

    for(int i = 1; i <= k; i++){
	 F(i) = points[indices[i-1]][2];
	 B(i, 1) = 1;
	 B(i, 2) = points[indices[i-1]][0];
	 B(i, 3) = points[indices[i-1]][1];
    }
    
    Matrix Bt = B.t();
    Matrix BtB = Bt * B;
    Matrix BtBinv = BtB.i();
    Matrix M = BtBinv * Bt;
  
    C = M * F;

    float z1 = C(1) + C(2) * (center.x + epsilon) + C(3) * center.y;
    float z2 = C(1) + C(2) * center.x + C(3) * (center.y + epsilon);
    
    BaseVertex diff1_z = BaseVertex(center.x + epsilon, center.y, z1) - center;
    BaseVertex diff2_z = BaseVertex(center.x, center.y + epsilon, z2) - center;

    normal = diff1_z.cross(diff2_z);
    
  } catch(Exception& e){
    //do nothing...
    cout << "NEWMAT EXEPTION" << endl;
  }
  

  // //Calculate covariance matrix
//   float matrix[3][3];
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 3; j++){
// 	 matrix[i][j] = 0.0;
//     }
//   }

//   for(int i = 0; i < k; i++){
//     BaseVertex current_vertex = BaseVertex(points[indices[i]][0],
// 								   points[indices[i]][1],
// 								   points[indices[i]][2]);
    
//     float vector_to_centroid[3];
//     vector_to_centroid[0] = current_vertex.x - center.x;
//     vector_to_centroid[1] = current_vertex.y - center.y;
//     vector_to_centroid[2] = current_vertex.z - center.z;

//     for(int a = 0; a < 3; a++){
// 	 for(int b = 0; b < 3; b++){
// 	   matrix[a][b] += vector_to_centroid[a] * vector_to_centroid[b];
// 	 }
//     }
//   }

//   //Copy convariance matrix into GSL matrix
//   gsl_matrix* mat = gsl_matrix_alloc(3, 3);
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 3; j++){
// 	 gsl_matrix_set(mat, i, j, matrix[i][j]);
//     }
//   }

//  //Calculate eigenvalues
//   gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(3);
//   gsl_vector *eval = gsl_vector_alloc(3);
//   gsl_matrix *evec = gsl_matrix_alloc(3, 3);
//   gsl_eigen_symmv (mat, eval, evec, w);
 
//   gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

//   //Get fist eigenvalue != 0
//   double i = 0.0;
//   int stelle = 0;
//   while (i != 0.0 && stelle <= 3) {
//        i = gsl_vector_get(eval, stelle); 
//        stelle++;
//   }

//   //Set eigenvector
//   gsl_vector *re = gsl_vector_alloc(3);
//   for (int j = 0; j < 3; j++) {
//      gsl_vector_set(re, j, gsl_matrix_get(evec, stelle, j));
//   } 
 
//   normal = Normal(gsl_vector_get(re, 0),
// 			   gsl_vector_get(re, 1),
// 			   gsl_vector_get(re, 2));

//   //Free memory
//   gsl_vector_free(re);
//   gsl_vector_free(eval);
//   gsl_matrix_free(mat);
//   gsl_matrix_free(evec);
//   gsl_eigen_symmv_free(w);

  
}

