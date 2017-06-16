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
 * AdaptiveKSearchSurface.cpp
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 *   co-Author: Dominik Feldschnieders (dofeldsc@uos.de)
 */


// External libraries in lvr source tree
#include <Eigen/Dense>

// boost libraries
#include <boost/filesystem.hpp>

#include <fstream>
#include <set>
#include <random>

#include <lvr2/util/Factories.hpp>

namespace lvr2
{

template<typename BaseVecT>
AdaptiveKSearchSurface<BaseVecT>::AdaptiveKSearchSurface()
    : m_useRANSAC(true)
{
    this->setKi(10);
    this->setKn(10);
    this->setKd(10);
}

template<typename BaseVecT>
AdaptiveKSearchSurface<BaseVecT>::AdaptiveKSearchSurface(
    PointBufferPtr<BaseVecT> buffer,
    std::string searchTreeName,
    int kn,
    int ki,
    int kd,
    bool useRansac,
    string posefile
) :
    PointsetSurface<BaseVecT>(buffer),
    m_searchTreeName(searchTreeName),
    m_useRANSAC(useRansac)
{
    this->setKi(ki);
    this->setKn(kn);
    this->setKd(kd);

    init();


    this->m_searchTree = getSearchTree(m_searchTreeName, buffer);

    if(!this->m_searchTree)
    {
       this->m_searchTree = getSearchTree("flann", buffer);
       cout << lvr::timestamp << "No valid search tree specified (" << searchTreeName << ")." << endl;
       cout << lvr::timestamp << "Maybe you did not install the required library." << endl;
       cout << lvr::timestamp << "Defaulting to flann." << endl;
    }

    if(posefile != "")
    {
        panic_unimplemented("posefile handling");
        // parseScanPoses(posefile);
    }

}

// template<typename BaseVecT>
// void AdaptiveKSearchSurface<BaseVecT>::parseScanPoses(string posefile)
// {
//     cout << lvr::timestamp << "Parsing scan poses." << endl;
//     std::ifstream in(posefile.c_str());
//     if(!in.good())
//     {
//         cout << lvr::timestamp << "Unable to open scan pose file " << posefile << endl;
//         return;
//     }

//     // Read vertex information
//     float x, y, z;
//     std::vector<VertexT> v;
//     while(in.good())
//     {
//         in >> x >> y >> z;
//         v.push_back(VertexT(x, y, z));
//     }

//     if(v.size() > 0)
//     {
//         PointBufferPtr loader (new PointBuffer);
//         floatArr points(new float[3 * v.size()]);
//         for(size_t i = 0; i < v.size(); i++)
//         {
//             points[3 * i]       = v[i][0];
//             points[3 * i + 1]   = v[i][1];
//             points[3 * i + 2]   = v[i][2];
//         }

//         loader->setPointArray(points, v.size());
//         size_t n = v.size();

//         cout << lvr::timestamp << "Creating pose search tree(" << m_searchTreeName << ") with " << n << " poses." << endl;

// #ifdef LVR_USE_PCL
//         if( m_searchTreeName == "pcl"  || m_searchTreeName == "PCL" )
//         {
//             this->m_poseTree = search_tree::Ptr( new SearchTreeFlannPCL<VertexT>(loader, n, 1, 1, 1) );
//         }
// #endif
// #ifdef LVR_USE_STANN
//         if( m_searchTreeName == "stann" || m_searchTreeName == "STANN" )
//         {
//             this->m_poseTree = search_tree::Ptr( new SearchTreeStann<VertexT>(loader, n, 1, 1, 1) );
//         }
// #endif
//         if( m_searchTreeName == "nanoflann" || m_searchTreeName == "NANOFLANN")
//         {
//             this->m_poseTree = search_tree::Ptr( new SearchTreeNanoflann<VertexT>(loader, n, 1, 1, 1));
//         }
// #ifdef LVR_USE_NABO
//         if( m_searchTreeName == "nabo" || m_searchTreeName == "NABO" )
//         {
//             this->m_poseTree = search_tree::Ptr( new SearchTreeNabo<VertexT>(loader, n, 1, 1, 1));
//         }
// #endif
//         if( m_searchTreeName == "flann" || m_searchTreeName == "FLANN")
//         {
//             this->m_searchTree = search_tree::Ptr(new SearchTreeFlann<VertexT>(loader, n, 1, 1, 1));
//         }

//         if( !this->m_poseTree )
//         {
//             cout << lvr::timestamp << "No Valid Searchtree class specified!" << endl;
//             cout << lvr::timestamp << "Class: " << m_searchTreeName << endl;
//         }
//     }
// }

template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::init()
{
    cout << lvr::timestamp << "##### Dataset statatistics: ##### " << endl << endl;
    cout << "Num points \t: " << this->m_pointBuffer->getNumPoints() << endl;
    cout <<  this->m_boundingBox << endl;
    cout << endl;
    this->m_centroid = Point<BaseVecT>(0.0, 0.0, 0.0);
}



template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::calculateSurfaceNormals()
{
    int k_0 = this->m_kn;

    cout << lvr::timestamp << "Initializing normal array..." << endl;

    this->m_pointBuffer->addNormalChannel();

    // Create a progress counter
    string comment = lvr::timestamp.getElapsedTime() + "Estimating normals ";
    lvr::ProgressBar progress(this->m_pointBuffer->getNumPoints(), comment);

    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < this->m_pointBuffer->getNumPoints(); i++) {
        // We have to fit these vector to have the
        // correct return values when performing the
        // search on the stann kd tree. So we don't use
        // the template parameter T for di
        vector<size_t> id;
        vector<float> di;

        int n = 0;
        size_t k = k_0;

        while(n < 5)
        {
            n++;
            /**
             *  @todo Maybe this should be done at the end of the loop
             *        after the bounding box check
             */
            k = k * 2;

            //T* point = this->m_points[i];
            this->m_searchTree->kSearch(this->m_pointBuffer->getPoint(i), k, id, di);

            float min_x = 1e15f;
            float min_y = 1e15f;
            float min_z = 1e15f;
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
            for(size_t j = 0; j < k; j++) {
                min_x = min(min_x, this->m_pointBuffer->getPoint(id[j]).x);
                min_y = min(min_y, this->m_pointBuffer->getPoint(id[j]).y);
                min_z = min(min_z, this->m_pointBuffer->getPoint(id[j]).z);

                max_x = max(max_x, this->m_pointBuffer->getPoint(id[j]).x);
                max_y = max(max_y, this->m_pointBuffer->getPoint(id[j]).y);
                max_z = max(max_z, this->m_pointBuffer->getPoint(id[j]).z);

                dx = max_x - min_x;
                dy = max_y - min_y;
                dz = max_z - min_z;
            }

            if(boundingBoxOK(dx, dy, dz))
            {
                break;
            }
        }

        // Create a query point for the current point
        auto queryPoint = this->m_pointBuffer->getPoint(i);

        // Interpolate a plane based on the k-neighborhood
        Plane<BaseVecT> p;
        bool ransac_ok;
        if(m_useRANSAC)
        {
            p = calcPlaneRANSAC(queryPoint, k, id, ransac_ok);
            // Fallback if RANSAC failed
            if(!ransac_ok)
            {
                p = calcPlane(queryPoint, k, id);
            }
        }
        else
        {
            p = calcPlane(queryPoint, k, id);
        }
        // Get the mean distance to the tangent plane
        //mean_distance = meanDistance(p, id, k);
        Normal<BaseVecT> normal(0, 0, 1);

        // Flip normals towards the center of the scene or nearest scan pose
        if(m_poseTree)
        {
            vector<size_t> nearestPoseIds;
            m_poseTree->kSearch(queryPoint, 1, nearestPoseIds);
            if(nearestPoseIds.size() == 1)
            {
                auto nearest = this->m_pointBuffer->getPoint(nearestPoseIds[0]);
                normal = p.n;
                if(normal.dot(queryPoint - nearest) < 0)
                {
                    normal = -normal;
                }
            }
            else
            {
                cout << lvr::timestamp << "Could not get nearest scan pose. Defaulting to centroid." << endl;
                normal =  p.n;
                if(normal.dot(queryPoint - m_centroid) < 0)
                {
                    normal = -normal;
                }
            }
        }
        else
        {
            normal =  p.n;
            if(normal.dot(queryPoint - m_centroid) < 0)
            {
                normal = -normal;
            }
        }

        // Save result in normal array
        this->m_pointBuffer->getNormal(i) = normal;
        ++progress;
    }
    cout << endl;

    if(this->m_ki)
    {
        interpolateSurfaceNormals();
    }
}


template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::interpolateSurfaceNormals()
{
    // Create a temporal normal array for the
    vector<Normal<BaseVecT>> tmp(
        this->m_pointBuffer->getNumPoints(),
        Normal<BaseVecT>(0, 0, 1)
    );

    // Create progress output
    string comment = lvr::timestamp.getElapsedTime() + "Interpolating normals ";
    lvr::ProgressBar progress(this->m_pointBuffer->getNumPoints(), comment);

    // Interpolate normals
    #pragma omp parallel for schedule(static)
    for( int i = 0; i < (int)this->m_pointBuffer->getNumPoints(); i++){

        vector<size_t> id;
        vector<float> di;

        this->m_searchTree->kSearch(this->m_pointBuffer->getPoint(i), this->m_ki, id, di);

        Vector<BaseVecT> mean;

        for(int j = 0; j < this->m_ki; j++)
        {
            mean += this->m_pointBuffer->getNormal(id[j])->asVector();
        }
        auto mean_normal = mean.normalized();

        tmp[i] = mean_normal;

        ///todo Try to remove this code. Should improve the results at all.
        for(int j = 0; j < this->m_ki; j++)
        {
            auto n = this->m_pointBuffer->getNormal(id[j]);

            // Only override existing normals if the interpolated
            // normals is significantly different from the initial
            // estimation. This helps to avoid a too smooth normal
            // field
            if(fabs(n->dot(mean_normal.asVector())) > 0.2 )
            {
                this->m_pointBuffer->getNormal(id[j]) = mean_normal;
            }
        }
        ++progress;
    }
    cout << endl;
    cout << lvr::timestamp << "Copying normals..." << endl;

    for(size_t i = 0; i < this->m_pointBuffer->getNumPoints(); i++){
        this->m_pointBuffer->getNormal(i) = tmp[i];
    }
}

template<typename BaseVecT>
bool AdaptiveKSearchSurface<BaseVecT>::boundingBoxOK(float dx, float dy, float dz)
{
    /**
     * @todo Replace magic number here.
     */
    float e = 0.05f;
    if(dx < e * dy) return false;
    else if(dx < e * dz) return false;
    else if(dy < e * dx) return false;
    else if(dy < e * dz) return false;
    else if(dz < e * dx) return false;
    else if(dz < e * dy) return false;
    return true;
}

// template<typename BaseVecT>
// void AdaptiveKSearchSurface<BaseVecT>::getkClosestVertices(const VertexT &v,
//         const size_t &k, vector<VertexT> &nb)
// {
//     vector<int> id;

//     //Allocate ANN point
//     {
//         coord<float> p;
//         p[0] = v[0];
//         p[1] = v[1];
//         p[2] = v[2];

//         //Find nearest tangent plane
//         // m_pointTree.ksearch( p, k, id, 0 );
//         this->m_searchTree->kSearch( p, k, id );
//     }

//     //parse result
//     if ( this->m_colors )
//     {
//         for ( size_t i = 0; i < k; i++ )
//         {
//             VertexT tmp;
//             nb.push_back( tmp );
//             nb[i].x = this->m_points[id[i]][0];
//             nb[i].y = this->m_points[id[i]][1];
//             nb[i].z = this->m_points[id[i]][2];
//     /*      nb[i].r = this->m_colors[id[i]][0];
//             nb[i].g = this->m_colors[id[i]][1];
//             nb[i].b = this->m_colors[id[i]][2]; */
//         }
//     }
//     else
//     {
//         for ( size_t i = 0; i < k; i++ )
//         {
//             VertexT tmp( this->m_points[id[i]][0], this->m_points[id[i]][1],
//                     this->m_points[id[i]][2] );
//             nb.push_back( tmp );
//         }
//     }
// }

// template<typename BaseVecT>
// float AdaptiveKSearchSurface<BaseVecT>::meanDistance(const Plane<BaseVecT> &p,
//         const vector<unsigned long> &id, const int &k)
// {
//     float sum = 0;
//     for(int i = 0; i < k; i++){
//         sum += distance(fromID(id[i]), p);
//     }
//     sum = sum / k;
//     return sum;
// }

// template<typename BaseVecT>
// float AdaptiveKSearchSurface<BaseVecT>::distance(VertexT v, Plane<BaseVecT> p)
// {
//     return fabs((v - p.p) * p.n);
// }

template<typename BaseVecT>
pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>
    AdaptiveKSearchSurface<BaseVecT>::distance(Point<BaseVecT> p) const
{
    int k = this->m_kd;

    vector<size_t> id;
    vector<float> di;

    //Allocate ANN point
    {
        // Find nearest tangent plane
        this->m_searchTree->kSearch( p, k, id, di );
    }

    BaseVecT nearest;
    Vector<BaseVecT> avg_normal;

    for ( int i = 0; i < k; i++ )
    {
        //Get nearest tangent plane
        auto vq = this->m_pointBuffer->getPoint(id[i]);

        //Get normal
        auto n = *this->m_pointBuffer->getNormal(id[i]);

        nearest += vq;
        avg_normal += n.asVector();
    }

    avg_normal /= k;
    nearest /= k;
    auto normal = avg_normal.normalized();

    //Calculate distance
    auto projectedDistance = (p - Point<BaseVecT>(nearest)).dot(normal.asVector());
    auto euklideanDistance = (p - Point<BaseVecT>(nearest)).length();

    return make_pair(projectedDistance, euklideanDistance);
}

// template<typename BaseVecT>
// VertexT AdaptiveKSearchSurface<BaseVecT>::fromID(int i){
//     return VertexT(
//             this->m_points[i][0],
//             this->m_points[i][1],
//             this->m_points[i][2]);
// }

template<typename BaseVecT>
Plane<BaseVecT> AdaptiveKSearchSurface<BaseVecT>::calcPlane(
    const Point<BaseVecT> &queryPoint,
    int k,
    const vector<size_t> &id
)
{
    /**
     * @todo Think of a better way to code this magic number.
     */
    const float epsilon = 100.0;

    // Calculate a least sqaures fit to the given points
    Eigen::Vector3f C;
    Eigen::VectorXf F(k);
    Eigen::MatrixXf B(k, 3);

    for(int j = 0; j < k; j++) {
        auto p = this->m_pointBuffer->getPoint(id[j]);
        F(j)    = p.y;
        B(j, 0) = 1.0f;
        B(j, 1) = p.x;
        B(j, 2) = p.z;
    }

    C = B.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(F);

    // Calculate to vectors in the fitted plane
    auto z1 = C(0) + C(1) * (queryPoint.x + epsilon) + C(2) * queryPoint.z;
    auto z2 = C(0) + C(1) * queryPoint.x + C(2) * (queryPoint.z + epsilon);

    // Calculcate the plane's normal via the cross product
    auto diff1 = Point<BaseVecT>(queryPoint.x + epsilon, z1, queryPoint.z) - queryPoint;
    auto diff2 = Point<BaseVecT>(queryPoint.x, z2, queryPoint.z + epsilon) - queryPoint;

    auto normal = diff1.cross(diff2).normalized();

    if(isnan(normal.getX()) || isnan(normal.getY()) || isnan(normal.getZ()))
    {
        cout << "Warning: Nan-coordinate in plane normal." << endl;
    }

    // Create a plane representation and return the result
    Plane<BaseVecT> p;
    // p.a = C(0);
    // p.b = C(1);
    // p.c = C(2);
    p.n = normal;
    p.p = queryPoint;

    return p;
}

// template<typename BaseVecT>
// const VertexT AdaptiveKSearchSurface<BaseVecT>::operator[]( const size_t& index ) const
// {
//     return VertexT(
//             m_points[index].x, m_points[index].y, m_points[index].z,
//             m_colors[index].r, m_colors[index].g, m_colors[index].b );
// }

//    template<typename BaseVecT>
// size_t AdaptiveKSearchSurface<BaseVecT>::getNumPoints()
// {
//     return m_numPoints;
// }

template<typename BaseVecT>
Plane<BaseVecT> AdaptiveKSearchSurface<BaseVecT>::calcPlaneRANSAC(
    const Point<BaseVecT> &queryPoint,
    int k,
    const vector<size_t> &id,
    bool &ok
)
{

   Plane<BaseVecT> p;

   //representation of best regression plane by point and normal
   Point<BaseVecT> bestPoint;
   Normal<BaseVecT> bestNorm(0, 0, 1);

   float bestdist = numeric_limits<float>::max();
   float dist     = 0;

   int iterations              = 0;
   int nonimproving_iterations = 0;

   //  int max_nonimproving = max(5, k / 2);
   int max_interations  = 10;

   while((nonimproving_iterations < 5) && (iterations < max_interations))
   {
       // randomly choose 3 disjoint points
       int c = 0;

       std::set<unsigned long> ids;
       std::default_random_engine generator;
       std::uniform_int_distribution<unsigned long> distribution(0, id.size() - 1);
       auto number = std::bind(distribution, generator);
       do
       {
           ids.insert(number());
           c++;
           if (c == 20) cout << "Deadlock" << endl;
       }
       while (ids.size() < 3 && c <= 20);

       vector<unsigned long> sample_ids(ids.size());
       std::copy(ids.begin(), ids.end(), sample_ids.begin());

       auto point1 = this->m_pointBuffer->getPoint(sample_ids[0]);
       auto point2 = this->m_pointBuffer->getPoint(sample_ids[1]);
       auto point3 = this->m_pointBuffer->getPoint(sample_ids[2]);

       auto n0 = (point1 - point2).cross(point1 - point3).normalized();

       //compute error to at most 50 other randomly chosen points
       dist = 0;
       int n = min(50, k);
       for(int i = 0; i < n; i++)
       {
           int index = id[rand() % k];
           auto refpoint = this->m_pointBuffer->getPoint(index);
           dist += fabs(refpoint.dot(n0.asVector()) - point1.dot(n0.asVector()));
       }
       if(n != 0) dist /= n;

       //a new optimum is found
       if(dist < bestdist)
       {
           bestdist = dist;

           bestPoint = point1;
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
   // p.a = 0;
   // p.b = 0;
   // p.c = 0;
   p.n = bestNorm;
   p.p = bestPoint;


   return p;
}


// template<typename BaseVecT>
// Plane<BaseVecT> AdaptiveKSearchSurface<BaseVecT>::calcPlaneRANSACfromPoints(const VertexT &queryPoint,
//         const int &k,
//         const vector<VertexT> points,
//         Normal<BaseVecT> c_normal,
//         bool &ok)
// {
//     // the resulting plane
//     Plane<BaseVecT> p;

//     VertexT point1;
//     VertexT point2;
//     VertexT point3;

//     //representation of best regression plane by point and normal
//     VertexT bestpoint;
//     Normal<BaseVecT> bestNorm;

//     float bestdist = numeric_limits<float>::max();
//     float dist     = 0;

//     int iterations              = 0;
//     int nonimproving_iterations = 0;

//     int max_interations = max(10, k / 2);
//     //int max_interations  = 10;

//     bool first_it = true;
//     while((nonimproving_iterations < 5) && (iterations < max_interations))
//     {
//         Normal<BaseVecT> n0;
//         //randomly choose 3 disjoint points
//         int c = 0;
//         do{
//             int index[3];
//             for(int i = 0; i < 3; i++)
//             {
//                 float f = 1.0 * rand() / RAND_MAX;
//                 int r = (int)(f * points.size() - 1);
//                 index[i] = r;
//             }

//             point1 = VertexT(this->m_points[index[0]][0],this->m_points[index[0]][1], this->m_points[index[0]][2]);
//             point2 = VertexT(this->m_points[index[1]][0],this->m_points[index[1]][1], this->m_points[index[1]][2]);
//             point3 = VertexT(this->m_points[index[2]][0],this->m_points[index[2]][1], this->m_points[index[2]][2]);

//             //compute normal of the plane given by the 3 points (look at the end)
//             n0 = (point1 - point2).cross(point1 - point3);
//             n0.normalize();
//             c++;

//             // check if the three points are disjoint
//             if( (point1 != point2) && (point2 != point3) && (point3 != point1) )
//             {
//                 // at first, use interpolated normal
//                 Normal<BaseVecT> check(0.0, 0.0, 0.0);
//                 if(first_it && !(check == c_normal))
//                 {
//                     n0 = c_normal;
//                     n0.normalize();
//                     first_it = false;
//                 }
//                 break;
//             }
//             // Check for deadlock
//             if(c > 50)
//             {
//                 cout << "DL " << k << endl;
//                 ok = false;
//                 return p;
//             }
//         }
//         while(true);

//         //compute error to at most 10 other randomly chosen points
//         dist = 0;
//         int n = min(10,k);
//         for(int i = 0; i < n; i++)
//         {
//             int index = rand() % points.size();
//             VertexT refpoint = VertexT(points[index][0], points[index][1] ,points[index][2]);
//             dist += fabs(refpoint * n0 - point1 * n0);
//         }
//         if(n != 0) dist /= n;

//         //a new optimum is found
//         if(dist < bestdist)
//         {
//             bestdist = dist;
//             bestpoint = point1;
//             bestNorm = n0;

//             nonimproving_iterations = 0;
//         }
//         else
//         {
//             nonimproving_iterations++;
//         }

//         iterations++;
//     } // end while

//     // Save plane parameters
//     p.a = 0;
//     p.b = 0;
//     p.c = 0;
//     p.n = bestNorm;
//     p.p = bestpoint;

//     return p;
// }

// template<typename BaseVecT>
// void AdaptiveKSearchSurface<BaseVecT>::colorizePointCloud(
//       typename AdaptiveKSearchSurface<BaseVecT>::Ptr pcm, const float& sqrtMaxDist,
//       const unsigned char* blankColor)
// {
// //    if( !m_colors )
// //    {
// //        m_colors = color3bArr( new color<uchar>[ m_numPoints ] );
// //    }
// //
// //#pragma omp parallel for schedule(dynamic)
// //    for( size_t i = 0; i < m_numPoints; i++ )
// //    {
// //        std::vector< VertexT > nearestPoint( 1 );
// //
// //        VertexT p( this->getPoint( i ) );
// //        pcm->getkClosestVertices( p, 1, nearestPoint );
// //        if(nearestPoint.size() )
// //        {
// //            if( p.sqrDistance( nearestPoint[0] ) < sqrtMaxDist )
// //            {
// //                m_colors[i][0] = nearestPoint[0].r;
// //                m_colors[i][1] = nearestPoint[0].g;
// //                m_colors[i][2] = nearestPoint[0].b;
// //            }
// //            else if( blankColor )
// //            {
// //                m_colors[i][0] = blankColor[0];
// //                m_colors[i][1] = blankColor[1];
// //                m_colors[i][2] = blankColor[2];
// //            }
// //        }
// //    }
// }




} // namespace lvr
