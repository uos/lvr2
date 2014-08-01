/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /*
 * PointCloudManager.cpp
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 *   co-Author: Dominik Feldschnieders (dofeldsc@uos.de)
 */


// External libraries in lvr source tree
#include <Eigen/Dense>

// boost libraries
#include <boost/filesystem.hpp>

namespace lvr
{

typedef ColorVertex<float, unsigned char>               cVertex;
typedef Normal<float>                                   cNormal;
typedef SearchTree<cVertex> search_tree;

template<typename VertexT, typename NormalT>
AdaptiveKSearchSurface<VertexT, NormalT>::AdaptiveKSearchSurface()
{
	m_useRANSAC = true;
    this->m_ki = 10;
    this->m_kn = 10;
    this->m_kd = 10;
}

template<typename VertexT, typename NormalT>
AdaptiveKSearchSurface<VertexT, NormalT>::AdaptiveKSearchSurface(
        PointBufferPtr loader,
        std::string searchTreeName,
        const int &kn,
        const int &ki,
        const int &kd,
        const bool &useRansac )
: PointsetSurface<VertexT>( loader )
{
   // Init:
//    srand(time(NULL));

   size_t n_points, n_normals;
   this->m_points = loader->getIndexedPointArray(n_points);
   this->m_normals = loader->getIndexedPointNormalArray(n_normals);
   this->m_numPoints = n_points;

   size_t n(0);
   this->m_colors = loader->getIndexedPointColorArray( n );
   if( n != this->m_numPoints )
   {
      this->m_colors.reset();
   }

    this->m_ki = ki;
    this->m_kn = kn;
    this->m_kd = kd;

    m_useRANSAC = useRansac;

    init();

    if( searchTreeName == "flann"  || searchTreeName == "FLANN" )
    {
#ifdef _USE_PCL_
        this->m_searchTree = search_tree::Ptr( new SearchTreeFlann<VertexT>(loader, this->m_numPoints, kn, ki, kd) );
#else
        cout << timestamp << "Warning: PCL is not installed. Using STANN search tree in AdaptiveKSearchSurface." << endl;
        this->m_searchTree = search_tree::Ptr( new SearchTreeStann<VertexT>(loader, this->m_numPoints, kn, ki, kd) );
#endif
    }
    else if( searchTreeName == "stann" || searchTreeName == "STANN" )
    {
        this->m_searchTree = search_tree::Ptr( new SearchTreeStann<VertexT>(loader, this->m_numPoints, kn, ki, kd) );
    }
    else if( searchTreeName == "nanoflann" || searchTreeName == "NANOFLANN")
    {
        this->m_searchTree = search_tree::Ptr( new SearchTreeNanoflann<VertexT>(loader, this->m_numPoints, kn, ki, kd));
    }
#ifdef _USE_NABO
    else if( searchTreeName == "nabo" || searchTreeName == "NABO" )
    {
        this->m_searchTree = search_tree::Ptr( new SearchTreeNabo<VertexT>(loader, this->m_numPoints, kn, ki, kd));
    }
#endif
    else
    {
       cout << timestamp << "No Valid Searchtree class specified!" << endl;
       cout << timestamp << "Class: " << searchTreeName << endl;
    }
}

template<typename VertexT, typename NormalT>
void AdaptiveKSearchSurface<VertexT, NormalT>::init()
{
    // Be sure that point information was given
    assert(this->m_points);

    // Calculate bounding box
    cout << timestamp << "Calculating bounding box." << endl;
    for( size_t i(0); i < this->m_numPoints; i++ )
    {
        this->m_boundingBox.expand(this->m_points[i][0],
                                   this->m_points[i][1],
                                   this->m_points[i][2]);
    }

    cout << timestamp << "##### Dataset statatistics: ##### " << endl << endl;
    cout << "Num points \t: " << this->m_numPoints << endl;
    cout <<  this->m_boundingBox << endl;
    cout << endl;
    this->m_centroid = VertexT(0.0, 0.0, 0.0);

    //this->m_boundingBox.getCentroid();
    // Create kd tree
    //cout << timestamp << "Creating STANN Kd-Tree..." << endl;
    //m_pointTree = sfcnn< coord<float>, 3, float>( this->m_points.get(),
    //        this->m_numPoints, omp_get_num_procs() );


}



template<typename VertexT, typename NormalT>
void AdaptiveKSearchSurface<VertexT, NormalT>::calculateSurfaceNormals()
{
    int k_0 = this->m_kn;

    cout << timestamp << "Initializing normal array..." << endl;

    //Initialize normal array
    this->m_normals = coord3fArr( new coord<float>[this->m_numPoints] );
    this->m_pointBuffer->setIndexedPointNormalArray(this->m_normals, this->m_numPoints);

    //float mean_distance;
    // Create a progress counter
    string comment = timestamp.getElapsedTime() + "Estimating normals ";
    ProgressBar progress(this->m_numPoints, comment);

    #pragma omp parallel for schedule(static)
    for( int i = 0; i < (int)this->m_numPoints; i++){

        Vertexf query_point;
        Normalf normal;

        // We have to fit these vector to have the
        // correct return values when performing the
        // search on the stann kd tree. So we don't use
        // the template parameter T for di
        vector<unsigned long> id;
        vector<double> di;

        int n = 0;
        size_t k = k_0;

        while(n < 5){

            n++;
            /**
             *  @todo Maybe this should be done at the end of the loop
             *        after the bounding box check
             */
            k = k * 2;

            //T* point = this->m_points[i];
            this->m_searchTree->kSearch(this->m_points[i], k, id, di);

            float min_x = 1e15;
            float min_y = 1e15;
            float min_z = 1e15;
            float max_x = - min_x;
            float max_y = - min_y;
            float max_z = - min_z;

            float dx, dy, dz;
            dx = dy = dz = 0;

            // Calculate the bounding box of found point set
            /**
             * @todo Use the bounding box object from the old model3d
             *       library for bounding box calculation...
             */
            for(size_t j = 0; j < k; j++){
                min_x = min(min_x, this->m_points[id[j]][0]);
                min_y = min(min_y, this->m_points[id[j]][1]);
                min_z = min(min_z, this->m_points[id[j]][2]);

                max_x = max(max_x, this->m_points[id[j]][0]);
                max_y = max(max_y, this->m_points[id[j]][1]);
                max_z = max(max_z, this->m_points[id[j]][2]);

//                cout << "Points: " << this->m_numPoints << "Point[" << id[j] << "].x: " <<  this->m_points[id[j]][0] << endl ;
//                cout << "Points: " << this->m_numPoints << "Point[" << id[j] << "].y: " <<  this->m_points[id[j]][1] << endl ;
//                cout << "Points: " << this->m_numPoints << "Point[" << id[j] << "].z: " <<  this->m_points[id[j]][2] << endl ;


                dx = max_x - min_x;
                dy = max_y - min_y;
                dz = max_z - min_z;
            }

            if(boundingBoxOK(dx, dy, dz)) break;
            //break;

        }

        // Create a query point for the current point
        query_point = VertexT(this->m_points[i][0],
                			  this->m_points[i][1],
                			  this->m_points[i][2]);

        // Interpolate a plane based on the k-neighborhood
        Plane<VertexT, NormalT> p;
        bool ransac_ok;
        if(m_useRANSAC)
        {
            p = calcPlaneRANSAC(query_point, k, id, ransac_ok);
            // Fallback if RANSAC failed
            if(!ransac_ok)
            {
                p = calcPlane(query_point, k, id);
            }
        }
        else
        {
            p = calcPlane(query_point, k, id);
        }
        // Get the mean distance to the tangent plane
        //mean_distance = meanDistance(p, id, k);

        // Flip normals towards the center of the scene
        normal =  p.n;

        if(normal * (query_point - m_centroid) < 0) normal = normal * -1;

        // Save result in normal array
        this->m_normals[i][0] = normal[0];
        this->m_normals[i][1] = normal[1];
        this->m_normals[i][2] = normal[2];
        ++progress;
    }
    cout << endl;

    if(this->m_ki) interpolateSurfaceNormals();
}


template<typename VertexT, typename NormalT>
void AdaptiveKSearchSurface<VertexT, NormalT>::interpolateSurfaceNormals()
{
    // Create a temporal normal array for the
    vector<NormalT> tmp(this->m_numPoints, NormalT());

    // Create progress output
    string comment = timestamp.getElapsedTime() + "Interpolating normals ";
    ProgressBar progress(this->m_numPoints, comment);

    // Interpolate normals
    #pragma omp parallel for schedule(static)
    for( int i = 0; i < (int)this->m_numPoints; i++){

        vector<unsigned long> id;
        vector<double> di;

        this->m_searchTree->kSearch(this->m_points[i], this->m_ki, id, di);

        VertexT mean;
        NormalT mean_normal;

        for(int j = 0; j < this->m_ki; j++)
        {
            mean += VertexT(this->m_normals[id[j]][0],
                            this->m_normals[id[j]][1],
                            this->m_normals[id[j]][2]);
        }
        mean_normal = NormalT(mean);

        tmp[i] = mean;

        ///todo Try to remove this code. Should improve the results at all.
        for(int j = 0; j < this->m_ki; j++)
        {
            NormalT n(this->m_normals[id[j]][0],
                      this->m_normals[id[j]][1],
                      this->m_normals[id[j]][2]);


            // Only override existing normals if the interpolated
            // normals is significantly different from the initial
            // estimation. This helps to avoid a too smooth normal
            // field
            if(fabs(n * mean_normal) > 0.2 )
            {
                this->m_normals[id[j]][0] = mean_normal[0];
                this->m_normals[id[j]][1] = mean_normal[1];
                this->m_normals[id[j]][2] = mean_normal[2];
            }
        }
        ++progress;
    }
    cout << endl;
    cout << timestamp << "Copying normals..." << endl;

    for(size_t i = 0; i < this->m_numPoints; i++){
        this->m_normals[i][0] = tmp[i][0];
        this->m_normals[i][1] = tmp[i][1];
        this->m_normals[i][2] = tmp[i][2];
    }
}

template<typename VertexT, typename NormalT>
bool AdaptiveKSearchSurface<VertexT, NormalT>::boundingBoxOK(const float &dx, const float &dy, const float &dz)
{
    /**
     * @todo Replace magic number here.
     */
    float e = 0.05;
    if(dx < e * dy) return false;
    else if(dx < e * dz) return false;
    else if(dy < e * dx) return false;
    else if(dy < e * dz) return false;
    else if(dz < e * dx) return false;
    else if(dy < e * dy) return false;
    return true;
}

template<typename VertexT, typename NormalT>
void AdaptiveKSearchSurface<VertexT, NormalT>::getkClosestVertices(const VertexT &v,
        const size_t &k, vector<VertexT> &nb)
{
    vector<unsigned long> id;

    //Allocate ANN point
    {
        coord<float> p;
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];

        //Find nearest tangent plane
        // m_pointTree.ksearch( p, k, id, 0 );
        this->m_searchTree->kSearch( p, k, id );
    }

    //parse result
	if ( this->m_colors )
	{
		for ( size_t i = 0; i < k; i++ )
		{
			VertexT tmp;
			nb.push_back( tmp );
            nb[i].x = this->m_points[id[i]][0];
            nb[i].y = this->m_points[id[i]][1];
            nb[i].z = this->m_points[id[i]][2];
			nb[i].r = this->m_colors[id[i]][0];
			nb[i].g = this->m_colors[id[i]][1];
			nb[i].b = this->m_colors[id[i]][2];
		}
	}
	else
	{
		for ( size_t i = 0; i < k; i++ )
		{
		    VertexT tmp( this->m_points[id[i]][0], this->m_points[id[i]][1],
		            this->m_points[id[i]][2] );
		    nb.push_back( tmp );
    	}
    }
}

template<typename VertexT, typename NormalT>
float AdaptiveKSearchSurface<VertexT, NormalT>::meanDistance(const Plane<VertexT, NormalT> &p,
        const vector<unsigned long> &id, const int &k)
{
    float sum = 0;
    for(int i = 0; i < k; i++){
        sum += distance(fromID(id[i]), p);
    }
    sum = sum / k;
    return sum;
}

template<typename VertexT, typename NormalT>
float AdaptiveKSearchSurface<VertexT, NormalT>::distance(VertexT v, Plane<VertexT, NormalT> p)
{
    return fabs((v - p.p) * p.n);
}

template<typename VertexT, typename NormalT>
void AdaptiveKSearchSurface<VertexT, NormalT>::distance(VertexT v, float &projectedDistance, float &euklideanDistance)
{
    int k = this->m_kd;

    vector<unsigned long> id;
    vector<double> di;

    //Allocate ANN point
    {
        coord<float> p;
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];

        // Find nearest tangent plane
        // m_pointTree.ksearch( p, k, id, di, 0 );
        this->m_searchTree->kSearch( p, k, id, di );
    }

    VertexT nearest;
    NormalT normal;

    for ( int i = 0; i < k; i++ )
    {
        //Get nearest tangent plane
        VertexT vq( this->m_points[id[i]][0], this->m_points[id[i]][1], this->m_points[id[i]][2] );

        //Get normal
        NormalT n( this->m_normals[id[i]][0], this->m_normals[id[i]][1], this->m_normals[id[i]][2] );

        nearest += vq;
        normal += n;

    }

    normal /= k;
    nearest /= k;
    normal.normalize();

    //Calculate distance
    projectedDistance = (v - nearest) * normal;
    euklideanDistance = (v - nearest).length();

}

template<typename VertexT, typename NormalT>
VertexT AdaptiveKSearchSurface<VertexT, NormalT>::fromID(int i){
    return VertexT(
            this->m_points[i][0],
            this->m_points[i][1],
            this->m_points[i][2]);
}

template<typename VertexT, typename NormalT>
Plane<VertexT, NormalT> AdaptiveKSearchSurface<VertexT, NormalT>::calcPlane(const VertexT &queryPoint,
        const int &k,
        const vector<unsigned long> &id)
{
    /**
     * @todo Think of a better way to code this magic number.
     */
    float epsilon = 100.0;

    VertexT diff1, diff2;
    NormalT normal;

    float z1 = 0;
    float z2 = 0;

    // Calculate a least sqaures fit to the given points
    Eigen::Vector3f C;
    Eigen::VectorXf F(k);
    Eigen::MatrixXf B(k,3);

    for(int j = 0; j < k; j++){
        F(j)    =  this->m_points[id[j]][1];
        B(j, 0) = 1.0f;
        B(j, 1) = this->m_points[id[j]][0];
        B(j, 2) = this->m_points[id[j]][2];
    }

    C = B.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(F);

    // Calculate to vectors in the fitted plane
    z1 = C(0) + C(1) * (queryPoint[0] + epsilon) + C(2) * queryPoint[2];
    z2 = C(0) + C(1) * queryPoint[0] + C(2) * (queryPoint[2] + epsilon);

    // Calculcate the plane's normal via the cross product
    diff1 = VertexT(queryPoint[0] + epsilon, z1, queryPoint[2]) - queryPoint;
    diff2 = VertexT(queryPoint[0], z2, queryPoint[2] + epsilon) - queryPoint;

    normal = diff1.cross(diff2);

    if(isnan(normal[0]) || isnan(normal[1]) || isnan(normal[2]))
    {
        cout << "Warning: Nan-coordinate in plane normal." << endl;
    }

    // Create a plane representation and return the result
    Plane<VertexT, NormalT> p;
    p.a = C(0);
    p.b = C(1);
    p.c = C(2);
    p.n = normal;
    p.p = queryPoint;

    return p;
}

template<typename VertexT, typename NormalT>
const VertexT AdaptiveKSearchSurface<VertexT, NormalT>::operator[]( const size_t& index ) const
{
    return VertexT(
            m_points[index].x, m_points[index].y, m_points[index].z,
            m_colors[index].r, m_colors[index].g, m_colors[index].b );
}

   template<typename VertexT, typename NormalT>
size_t AdaptiveKSearchSurface<VertexT, NormalT>::getNumPoints()
{
    return m_numPoints;
}

   template<typename VertexT, typename NormalT>
   Plane<VertexT, NormalT> AdaptiveKSearchSurface<VertexT, NormalT>::calcPlaneRANSAC(const VertexT &queryPoint,
           const int &k,
           const vector<unsigned long> &id,
           bool &ok)
           {

       Plane<VertexT, NormalT> p;

       VertexT point1;
       VertexT point2;
       VertexT point3;

       //representation of best regression plane by point and normal
       VertexT bestpoint;
       NormalT bestNorm;

       float bestdist = numeric_limits<float>::max();
       float dist     = 0;

       int iterations              = 0;
       int nonimproving_iterations = 0;

       //  int max_nonimproving = max(5, k / 2);
       int max_interations  = 10;

       while((nonimproving_iterations < 5) && (iterations < max_interations))
       {
           NormalT n0;

           //randomly choose 3 disjoint points
           int c = 0;
           do{
               //cout << "AAA" << endl;
               int index[3];
               for(int i = 0; i < 3; i++)
               {
                   float f = 1.0 * rand() / RAND_MAX;
                   int r = (int)(f * id.size());
                   index[i] = id[r];
               }

               if(id[0] != id[1] && id[1] != id[2] && id[2] != id[0])
               {
                   break;
               }

               point1 = VertexT(this->m_points[index[0]][0],this->m_points[index[0]][1], this->m_points[index[0]][2]);
               point2 = VertexT(this->m_points[index[1]][0],this->m_points[index[1]][1], this->m_points[index[1]][2]);
               point3 = VertexT(this->m_points[index[2]][0],this->m_points[index[2]][1], this->m_points[index[2]][2]);

               //compute normal of the plane given by the 3 points
               n0 = (point1 - point2).cross(point1 - point3);
               n0.normalize();

               //            if( (point1 != point2) && (point2 != point3) && (point3 != point1) )
               //            {
               //                break;
               //            }
               c++;

               //            cout << index[0] << " " << index[1] << " " << index[2] << " " << id.size() << endl;
               //            cout << point1;
               //            cout << point2;
               //            cout << point3;
               //            cout << endl;

               // Check for deadlock
               if(c > 50)
               {
                   cout << "DL " << k << endl;
                   ok = false;
                   return p;
               }
           }
           while(true);

           //compute error to at most 50 other randomly chosen points
           dist = 0;
           int n = min(50,k);
           for(int i = 0; i < n; i++)
           {
               int index = id[rand() % k];
               VertexT refpoint = VertexT(this->m_points[index][0], this->m_points[index][1] ,this->m_points[index][2]);
               dist += fabs(refpoint * n0 - point1 * n0);
           }
           if(n != 0) dist /= n;

           //a new optimum is found
           if(dist < bestdist)
           {
               bestdist = dist;

               bestpoint = point1;
               bestNorm = n0;

               nonimproving_iterations = 0;
           }
           else
           {
               nonimproving_iterations++;
           }

           iterations++;
       }

       // Save plane parameters
       p.a = 0;
       p.b = 0;
       p.c = 0;
       p.n = bestNorm;
       p.p = bestpoint;


       return p;
           }

template<typename VertexT, typename NormalT>
Plane<VertexT, NormalT> AdaptiveKSearchSurface<VertexT, NormalT>::calcPlaneRANSACfromPoints(const VertexT &queryPoint,
        const int &k,
        const vector<VertexT> points,
        NormalT c_normal,
        bool &ok)
{
	// the resulting plane
    Plane<VertexT, NormalT> p;

    VertexT point1;
    VertexT point2;
    VertexT point3;

    //representation of best regression plane by point and normal
    VertexT bestpoint;
    NormalT bestNorm;

    float bestdist = numeric_limits<float>::max();
    float dist     = 0;

    int iterations              = 0;
    int nonimproving_iterations = 0;

    int max_interations = max(10, k / 2);
    //int max_interations  = 10;

    bool first_it = true;
    while((nonimproving_iterations < 5) && (iterations < max_interations))
    {
        NormalT n0;
        //randomly choose 3 disjoint points
        int c = 0;
        do{
            int index[3];
            for(int i = 0; i < 3; i++)
            {
                float f = 1.0 * rand() / RAND_MAX;
                int r = (int)(f * points.size() - 1);
                index[i] = r;
            }

            point1 = VertexT(this->m_points[index[0]][0],this->m_points[index[0]][1], this->m_points[index[0]][2]);
            point2 = VertexT(this->m_points[index[1]][0],this->m_points[index[1]][1], this->m_points[index[1]][2]);
            point3 = VertexT(this->m_points[index[2]][0],this->m_points[index[2]][1], this->m_points[index[2]][2]);

            //compute normal of the plane given by the 3 points (look at the end)
    		n0 = (point1 - point2).cross(point1 - point3);
    		n0.normalize();
            c++;

            // check if the three points are disjoint
            if( (point1 != point2) && (point2 != point3) && (point3 != point1) )
            {
            	// at first, use interpolated normal
            	NormalT check(0.0, 0.0, 0.0);
            	if(first_it && !(check == c_normal))
            	{
            		n0 = c_normal;
            		n0.normalize();
            		first_it = false;
            	}
                break;
            }
            // Check for deadlock
            if(c > 50)
            {
                cout << "DL " << k << endl;
                ok = false;
                return p;
            }
        }
        while(true);

        //compute error to at most 10 other randomly chosen points
        dist = 0;
        int n = min(10,k);
        for(int i = 0; i < n; i++)
        {
            int index = rand() % points.size();
            VertexT refpoint = VertexT(points[index][0], points[index][1] ,points[index][2]);
            dist += fabs(refpoint * n0 - point1 * n0);
        }
        if(n != 0) dist /= n;

        //a new optimum is found
        if(dist < bestdist)
        {
            bestdist = dist;
            bestpoint = point1;
            bestNorm = n0;

            nonimproving_iterations = 0;
        }
        else
        {
            nonimproving_iterations++;
        }

        iterations++;
    } // end while

    // Save plane parameters
    p.a = 0;
    p.b = 0;
    p.c = 0;
    p.n = bestNorm;
    p.p = bestpoint;

    return p;
}

template<typename VertexT, typename NormalT>
void AdaptiveKSearchSurface<VertexT, NormalT>::colorizePointCloud(
      AdaptiveKSearchSurface<VertexT, NormalT>::Ptr pcm, const float& sqrtMaxDist,
      const uchar* blankColor)
{
//    if( !m_colors )
//    {
//        m_colors = color3bArr( new color<uchar>[ m_numPoints ] );
//    }
//
//#pragma omp parallel for schedule(dynamic)
//    for( size_t i = 0; i < m_numPoints; i++ )
//    {
//        std::vector< VertexT > nearestPoint( 1 );
//
//        VertexT p( this->getPoint( i ) );
//        pcm->getkClosestVertices( p, 1, nearestPoint );
//        if(nearestPoint.size() )
//        {
//            if( p.sqrDistance( nearestPoint[0] ) < sqrtMaxDist )
//            {
//                m_colors[i][0] = nearestPoint[0].r;
//                m_colors[i][1] = nearestPoint[0].g;
//                m_colors[i][2] = nearestPoint[0].b;
//            }
//            else if( blankColor )
//            {
//                m_colors[i][0] = blankColor[0];
//                m_colors[i][1] = blankColor[1];
//                m_colors[i][2] = blankColor[2];
//            }
//        }
//    }
}




} // namespace lvr


