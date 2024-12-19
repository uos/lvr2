/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include <algorithm>

#include <lvr2/util/Progress.hpp>

#include "lvr2/util/Factories.hpp"
#include "lvr2/util/Logging.hpp"

namespace lvr2
{

template<typename BaseVecT>
AdaptiveKSearchSurface<BaseVecT>::AdaptiveKSearchSurface(
    PointBufferPtr buffer,
    std::string searchTreeName,
    int kn,
    int ki,
    int kd,
    int calcMethod,
    std::string posefile
) :
    PointsetSurface<BaseVecT>(buffer),
    m_searchTreeName(searchTreeName),
    m_calcMethod(calcMethod)
{
    this->setKi(ki);
    this->setKn(kn);
    this->setKd(kd);

    init();

    this->m_searchTree = getSearchTree<BaseVecT>(m_searchTreeName, buffer);

    if(!this->m_searchTree)
    {
       this->m_searchTree = getSearchTree<BaseVecT>("flann", buffer);
       lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] No valid search tree specified (" << searchTreeName << ")." << lvr2::endl;
       lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] Maybe you did not install the required library." << lvr2::endl;
       lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] Defaulting to flann." << lvr2::endl;
    }

    if(posefile != "")
    {
        //panic_unimplemented("posefile handling");
        parseScanPoses(posefile);
    }
}

// TODO: it is possible that this will not work anymore after removing the default contructor. Check this
template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::parseScanPoses(string posefile)
{
    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Parsing scan poses." << lvr2::endl;
    std::ifstream in(posefile.c_str());
    if (!in.good())
    {
      lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] Unable to open scan pose file " << posefile << lvr2::endl;
      return;
    }

    // Read vertex information
    float x, y, z;
    std::vector<BaseVecT> v;
    while(in.good())
    {
        in >> x >> y >> z;
        v.push_back(BaseVecT(x, y, z));
    }

    if(v.size() > 0)
    {
        PointBufferPtr loader (new PointBuffer);
        floatArr points(new float[3 * v.size()]);
        for(size_t i = 0; i < v.size(); i++)
        {
            points[3 * i]       = v[i][0];
            points[3 * i + 1]   = v[i][1];
            points[3 * i + 2]   = v[i][2];
        }

        loader->setPointArray(points, v.size());
        size_t n = v.size();

        lvr2::logout::get() << lvr2::info <<  "[AdaptiveKSearchSurface]  Creating pose search tree(" << m_searchTreeName << ") with "
            << n << " poses." << lvr2::endl;

        this->m_poseTree = getSearchTree<BaseVecT>(m_searchTreeName, loader);

        if( !this->m_poseTree )
        {
            lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] No Valid Searchtree class specified!" << lvr2::endl;
            lvr2::logout::get() << lvr2::warning <<  "[AdaptiveKSearchSurface] Class: " << m_searchTreeName << lvr2::endl;
        }
    }
}

template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::init()
{
    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Dataset statistics: " << lvr2::endl;
    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Num points: " << m_points.numElements() << lvr2::endl;
    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] kn, ki, kd: "<< this->m_kn << ", " << this->m_ki << ", " << this->m_kd << lvr2::endl;
    const auto& min = this->m_boundingBox.getMin(), max = this->m_boundingBox.getMax();
    lvr2::logout::get() 
        << lvr2::info 
        << "[AdaptiveKSearchSurface] BB of points: [" 
        << min.x << ", " << min.y << ", " << min.z << "] - ["
        << max.x << ", " << max.y << ", " << max.z << "]" 
        << lvr2::endl;

    this->m_flipPoint = this->m_boundingBox.getCentroid();
}

template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::calculateSurfaceNormals()
{
    int k_0 = this->m_kn;
    const size_t numPoints = m_points.numElements();

    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Initializing normal array..." << lvr2::endl;

    floatArr normals = floatArr(new float[numPoints * 3]);
    this->m_pointBuffer->setNormalArray(normals, numPoints);

    const int max_threads = omp_get_max_threads();
    const int normal_estimation_threads = max_threads;

    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Estimating " << numPoints << " Surface Normals using " << normal_estimation_threads << " threads ..." << lvr2::endl;
    // Create a monitor counter
    lvr2::Monitor monitor(lvr2::LogLevel::info, "[AdaptiveKSearchSurface] Estimating Normals", numPoints);

    // lvr2::PacmanProgressBar monitor(numPoints / normal_estimation_threads, "[AdaptiveKSearchSurface] Estimating Normals");

    #pragma omp parallel for schedule(dynamic) num_threads(normal_estimation_threads) shared(monitor)
    for(size_t i = 0; i < numPoints; i++)
    {
        // We have to fit these vector to have the
        // correct return values when performing the
        // search on the stann kd tree. So we don't use
        // the template parameter T for di
        std::vector<size_t> id;

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

            id.clear();

            this->m_searchTree->kSearch(m_points[i], k, id);

            // Calculate the bounding box of found point set
            BoundingBox<BaseVecT> bb;
            for (auto& index : id)
            {
                bb.expand(BaseVecT(m_points[index]));
            }

            if(boundingBoxOK(bb))
            {
                break;
            }
        }

        // Create a query point for the current point
        auto queryPoint = m_points[i];

        // Interpolate a plane based on the k-neighborhood
        Plane<BaseVecT> p;
        bool ransac_ok;

        if(m_calcMethod == 1)
        {
            p = calcPlaneRANSAC(queryPoint, id, ransac_ok);
            // Fallback if RANSAC failed
            if(!ransac_ok)
            {
                // compare speed
                p = calcPlane(queryPoint, id);
            }
        }
        else if(m_calcMethod == 2)
        {
            p = calcPlaneIterative(queryPoint, id);
        }
        else if(m_calcMethod == 3)
        {
            p = calcPlaneIPCAExact(queryPoint, id);
        }
        else
        {
            p = calcPlane(queryPoint, id);
        }
        // Get the mean distance to the tangent plane
        //mean_distance = meanDistance(p, id, k);
        auto normal = p.normal;
        bool normalCorrected = false;

        // Flip normals towards the center of the scene or nearest scan pose
        if(m_poseTree)
        {
            std::vector<size_t> nearestPoseIds;
            m_poseTree->kSearch(queryPoint, 1, nearestPoseIds);
            if(nearestPoseIds.size() == 1)
            {
                BaseVecT nearest = m_points[nearestPoseIds[0]];
                if(normal.dot(nearest - queryPoint) < 0)
                {
                    normal = -normal;
                }
                normalCorrected = true;
            }
        }

        if (!normalCorrected)
        {
            if(normal.dot(this->m_flipPoint - queryPoint) < 0)
            {
                normal = -normal;
            }
        }

        // Save result in normal array
        normals[i*3 + 0] = normal.x;
        normals[i*3 + 1] = normal.y;
        normals[i*3 + 2] = normal.z;

        ++monitor;
    }

    monitor.terminate();
   
    if(this->m_ki)
    {
        interpolateSurfaceNormals();
    }
}


template<typename BaseVecT>
void AdaptiveKSearchSurface<BaseVecT>::interpolateSurfaceNormals()
{
    const size_t numPoints     = this->m_pointBuffer->numPoints();
    FloatChannel normals = *(this->m_pointBuffer->getFloatChannel("normals"));
    // Create a temporal normal array for the
    std::vector<Normal<typename BaseVecT::CoordType>> tmp(
        numPoints,
        Normal<typename BaseVecT::CoordType>(0, 0, 1)
    );

    const int max_threads = omp_get_max_threads();
    const int normal_interpolation_threads = max_threads;

    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Interpolating " << numPoints << " Surface Normals using " << normal_interpolation_threads << " threads ..." << lvr2::endl;
    // Create monitor output
    lvr2::Monitor monitor(lvr2::LogLevel::info, "[AdaptiveKSearchSurface] Interpolating normals", numPoints);

    // Interpolate normals
    #pragma omp parallel for schedule(dynamic) num_threads(normal_interpolation_threads) shared(monitor)
    for( size_t i = 0; i < numPoints; i++)
    {
        vector<size_t> id;

        this->m_searchTree->kSearch(m_points[i], this->m_ki, id);

        BaseVecT mean = normals[i];
        for(auto& index : id)
        {
            mean += normals[index];
        }
        tmp[i] = mean.normalized();

        ++monitor;
    }
    monitor.terminate();
    // std::cout << std::endl;

    lvr2::logout::get() << lvr2::info << "[AdaptiveKSearchSurface] Copying normals..." << lvr2::endl;
    for(size_t i = 0; i < numPoints; i++){
        normals[i] = tmp[i];
    }
}

template<typename BaseVecT>
bool AdaptiveKSearchSurface<BaseVecT>::boundingBoxOK(const BoundingBox<BaseVecT>& bb)
{
    float dx = bb.getXSize();
    float dy = bb.getYSize();
    float dz = bb.getZSize();
    /**
     * @todo Replace magic number here.
     */
    float e = 0.05f;
    return dx >= e * dy && dx >= e * dz
        && dy >= e * dx && dy >= e * dz
        && dz >= e * dx && dz >= e * dy;
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
    AdaptiveKSearchSurface<BaseVecT>::distance(BaseVecT p) const
{
    const FloatChannel normals = *(this->m_pointBuffer->getFloatChannel("normals"));
    size_t numPoints     = m_points.numElements();

    vector<size_t> id;

    // Find nearest tangent plane
    this->m_searchTree->kSearch( p, this->m_kd, id );

    if (id.empty())
    {
        auto dist = std::numeric_limits<typename BaseVecT::CoordType>::max();
        return std::make_pair(dist, dist);
    }

    BaseVecT nearest;
    BaseVecT avg_normal;

    for ( auto& index : id )
    {
        //Get nearest tangent plane
        auto vq = m_points[index];

        //Get normal
        auto n = normals[index];

        nearest += vq;
        avg_normal += n;
    }

    avg_normal /= id.size();
    nearest /= id.size();
    auto normal = avg_normal.normalized();

    //Calculate distance
    auto projectedDistance = (p - BaseVecT(nearest)).dot(normal);
    auto euklideanDistance = (p - BaseVecT(nearest)).length();

    return std::make_pair(projectedDistance, euklideanDistance);
    // return make_pair(euklideanDistance, projectedDistance);
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
    const BaseVecT &queryPoint,
    const std::vector<size_t> &id
)
{
    const size_t numPoints     = m_points.numElements();

    /**
     * @todo Think of a better way to code this magic number.
     */
    const float epsilon = 100.0;

    // Calculate a least sqaures fit to the given points
    Eigen::Vector3f C;
    Eigen::VectorXf F(id.size());
    Eigen::MatrixXf B(id.size(), 3);

    for(size_t j = 0; j < id.size(); j++) 
    {
        const BaseVecT p = m_points[id[j]];
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
    auto diff1 = BaseVecT(queryPoint.x + epsilon, z1, queryPoint.z) - queryPoint;
    auto diff2 = BaseVecT(queryPoint.x, z2, queryPoint.z + epsilon) - queryPoint;

    auto normal = diff1.cross(diff2).normalized();

    if(isnan(normal.getX()) || isnan(normal.getY()) || isnan(normal.getZ()))
    {
        lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] Warning: Nan-coordinate in plane normal." << lvr2::endl;
    }

    // Create a plane representation and return the result
    Plane<BaseVecT> p;
    // p.a = C(0);
    // p.b = C(1);
    // p.c = C(2);
    p.normal = normal;
    p.pos = queryPoint;

    return p;
}

template<typename BaseVecT>
Plane<BaseVecT> AdaptiveKSearchSurface<BaseVecT>::calcPlaneIterative(
    const BaseVecT& queryPoint,
    const vector<size_t>& id)
{
    const size_t numPoints     = m_points.numElements();

    Plane<BaseVecT> p;
    Normal<typename BaseVecT::CoordType> normal;

    //x
    float xx = 0.0;
    float xy = 0.0;
    float xz = 0.0;

    //y
    float yy = 0.0;
    float yz = 0.0;

    //z
    float zz = 0.0;

    for(auto& index : id) {
        auto p = m_points[index];

        auto r = p - queryPoint;

        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    //determinante
    float det_x = yy * zz - yz * yz;
    float det_y = xx * zz - xz * xz;
    float det_z = xx * yy - xy * xy;

    float dir_x;
    float dir_y;
    float dir_z;
    // det X biggest
    if( det_x >= det_y && det_x >= det_z){

        if(det_x <= 0.0){
            //not a plane
        }

        dir_x = 1.0;
        dir_y = (xz * yz - xy * zz) / det_x;
        dir_z = (xy * yz - xz * yy) / det_x;
    } //det Y biggest
    else if( det_y >= det_x && det_y >= det_z){

        if(det_y <= 0.0){
            // not a plane
        }

        dir_x = (yz * xz - xy * zz) / det_y;
        dir_y = 1.0;
        dir_z = (xy * xz - yz * xx) / det_y;
    } // det Z biggest
    else{
        if(det_z <= 0.0){
            // not a plane
        }

        dir_x = (yz * xy - xz * yy ) / det_z;
        dir_y = (xz * xy - yz * xx ) / det_z;
        dir_z = 1.0;
    }

    const float invnorm = 1/sqrtf( dir_x * dir_x + dir_y * dir_y + dir_z * dir_z );

    normal.x = dir_x * invnorm;
    normal.y = dir_y * invnorm;
    normal.z = dir_z * invnorm;


    p.normal = normal;
    p.pos = queryPoint;

    return p;
}

template<typename BaseVecT>
Plane<BaseVecT> AdaptiveKSearchSurface<BaseVecT>::calcPlaneIPCAExact(
    const BaseVecT& queryPoint,
    const vector<size_t>& ids)
{
    const Eigen::Vector3f qp(queryPoint.x, queryPoint.y, queryPoint.z);
    Eigen::Matrix3f cov  = Eigen::Matrix3f::Zero();
    Eigen::Vector3f mean(queryPoint.x, queryPoint.y, queryPoint.z);
    double n_meas = 1.0;

    for(size_t i=0; i<ids.size(); i++)
    {
        const size_t index = ids[i];
        const Eigen::Vector3f p = Eigen::Vector3f(m_points[index][0], m_points[index][1], m_points[index][2]);

        const double n_meas_new = n_meas + 1.0;
        double w1 = n_meas /n_meas_new;
        double w2 = 1.0 / n_meas_new;

        // update mean
        const Eigen::Vector3f mean_new = mean * w1 + p * w2;

        // update cov
        const Eigen::Matrix3f P1 = (mean - mean_new) * (mean - mean_new).transpose();
        const Eigen::Matrix3f P2 = (p - mean_new) * (p - mean_new).transpose();
        const Eigen::Matrix3f cov_new = (cov + P1) * w1 + P2 * w2;

        // write
        n_meas = n_meas_new;
        mean = mean_new;
        cov = cov_new;
    }

    Eigen::EigenSolver<Eigen::Matrix3f> es(cov, true);
    

    const auto eigen_vals = es.eigenvalues().real();

    Eigen::Vector3f smallest_eigenvector; 
    if(eigen_vals(0) < eigen_vals(1) && eigen_vals(0) < eigen_vals(2))
    {
        smallest_eigenvector = es.eigenvectors().col(0).real();
    } 
    else if(eigen_vals(1) < eigen_vals(2)) 
    {
        smallest_eigenvector = es.eigenvectors().col(1).real();
    } 
    else 
    {
        smallest_eigenvector = es.eigenvectors().col(2).real();
    }

    // std::cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << std::endl;
    // std::cout << es.eigenvectors() << std::endl;

    // std::cout << "Smallest Vec: " << smallest_eigenvector.transpose() << std::endl;

    Plane<BaseVecT> res;
    res.normal.x = smallest_eigenvector.x();
    res.normal.y = smallest_eigenvector.y();
    res.normal.z = smallest_eigenvector.z();
    res.pos = queryPoint;

    return res;
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
    const BaseVecT &queryPoint,
    const vector<size_t> &id,
    bool &ok
)
{
    FloatChannel normals = *(this->m_pointBuffer->getFloatChannel("normals"));
    size_t numPoints     = m_points.numElements();

   Plane<BaseVecT> p;

   //representation of best regression plane by point and normal
   BaseVecT bestPoint;
   Normal<typename BaseVecT::CoordType> bestNorm(0, 0, 1);

   float bestdist = numeric_limits<float>::max();
   float dist     = 0;

   int iterations              = 0;
   int nonimproving_iterations = 0;

   //  int max_nonimproving = max(5, k / 2);
   int max_interations  = 10;

   std::unordered_set<size_t> ids;
   std::default_random_engine generator;
   std::uniform_int_distribution<size_t> distribution(0, id.size() - 1);
   auto number = std::bind(distribution, generator);

   while((nonimproving_iterations < 5) && (iterations < max_interations))
   {
       // randomly choose 3 disjoint points
        int c = 0;

        ids.clear();
        do
        {
            ids.insert(number());
            c++;
            if (c == 20)
            {
                lvr2::logout::get() << lvr2::warning << "[AdaptiveKSearchSurface] Deadlock" << lvr2::endl;
            }
        } 
        while (ids.size() < 3 && c <= 20);

       auto it = ids.begin();

       BaseVecT point1 = m_points[*it];
       BaseVecT point2 = m_points[*(++it)];
       BaseVecT point3 = m_points[*(++it)];

       auto n0 = (point1 - point2).cross(point1 - point3).normalized();

       //compute error to at most 50 other randomly chosen points
       dist = 0;
       int n = std::min(50, (int)id.size());
       for(int i = 0; i < n; i++)
       {
           int index = id[number()];
           BaseVecT refpoint = m_points[index];
           dist += fabs(refpoint.dot(n0) - point1.dot(n0));
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
   p.normal = bestNorm;
   p.pos = bestPoint;

   return p;
}


// template<typename BaseVecT>
// Plane<BaseVecT> AdaptiveKSearchSurface<BaseVecT>::calcPlaneRANSACfromPoints(const VertexT &queryPoint,
//         const int &k,
//         const vector<VertexT> points,
//         Normal<typename BaseVecT::CoordType> c_normal,
//         bool &ok)
// {
//     // the resulting plane
//     Plane<BaseVecT> p;

//     VertexT point1;
//     VertexT point2;
//     VertexT point3;

//     //representation of best regression plane by point and normal
//     VertexT bestpoint;
//     Normal<typename BaseVecT::CoordType> bestNorm;

//     float bestdist = numeric_limits<float>::max();
//     float dist     = 0;

//     int iterations              = 0;
//     int nonimproving_iterations = 0;

//     int max_interations = max(10, k / 2);
//     //int max_interations  = 10;

//     bool first_it = true;
//     while((nonimproving_iterations < 5) && (iterations < max_interations))
//     {
//         Normal<typename BaseVecT::CoordType> n0;
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
//                 Normal<typename BaseVecT::CoordType> check(0.0, 0.0, 0.0);
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
//                 std::cout << "DL " << k << std::endl;
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




} // namespace lvr2
