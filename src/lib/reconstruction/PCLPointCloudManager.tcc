/**
 * PCLPointCloudManager.cpp
 *
 *  @date 08.08.2011
 *  @author Thomas Wiemann
 */

#include "PCLPointCloudManager.hpp"
#include "io/Timestamp.hpp"

#include <vector>

namespace lssr
{

template<typename VertexT, typename NormalT>
PCLPointCloudManager<VertexT, NormalT>::PCLPointCloudManager(string filename)
{
    typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // Read dat from disk
    this->readFromFile(filename);

    // Parse to PCL point cloud
    cout << timestamp << "Creating PCL point cloud" << endl;
    cloud->resize(this->m_numPoints);
    float x, y, z;
    for(size_t i = 0; i < this->m_numPoints; i++)
    {
        x = this->m_points[i][0];
        y = this->m_points[i][1];
        z = this->m_points[i][2];

        this->m_boundingBox.expand(x, y, z);

        cloud->points[i].x = x;
        cloud->points[i].y = y;
        cloud->points[i].z = z;
    }

    // Estimate normals
    cout << timestamp << "Estimating normals" << endl;

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr m_kdTree (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
    ne.setSearchMethod (m_kdTree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setKSearch(100);

    // Compute the features
    ne.compute (*cloud_normals);
    cout << timestamp << "Normal estimation done" << endl;

    //cout << cloud_normals->points.size() << endl;

}

template<typename VertexT, typename NormalT>
PCLPointCloudManager<VertexT, NormalT>::~PCLPointCloudManager()
{
    // TODO Auto-generated destructor stub
}

template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::getkClosestVertices(const VertexT &v,
         const size_t &k, vector<VertexT> &nb)
{
    std::vector< int > k_indices;
    std::vector< float > k_distances;

    pcl::PointXYZ qp;
    qp.x = v.x;
    qp.y = v.y;
    qp.z = v.z;

    // Query tree
    int res = m_kdTree->nearestKSearch(qp, k, k_indices, k_distances);

    // Parse result
    for(int i = 0; i < res; i++)
    {

    }
}


template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::getkClosestNormals(const VertexT &n,
         const size_t &k, vector<NormalT> &nb)
{

}

template<typename VertexT, typename NormalT>
float PCLPointCloudManager<VertexT, NormalT>::distance(VertexT v)
{

}

} // namespace lssr
