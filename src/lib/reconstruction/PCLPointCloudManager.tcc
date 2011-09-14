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
PCLPointCloudManager<VertexT, NormalT>::PCLPointCloudManager(string filename, int kn, int ki, int kd)
{
    this->m_kn = kn;
    this->m_ki = ki;
    this->m_kd = kd;

    m_pointCloud  = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);

    // Read dat from disk
    this->readFromFile(filename);

    // Parse to PCL point cloud
    cout << timestamp << "Creating PCL point cloud" << endl;
    m_pointCloud->resize(this->m_numPoints);
    float x, y, z;
    for(size_t i = 0; i < this->m_numPoints; i++)
    {
        x = this->m_points[i][0];
        y = this->m_points[i][1];
        z = this->m_points[i][2];

        this->m_boundingBox.expand(x, y, z);

        m_pointCloud->points[i].x = x;
        m_pointCloud->points[i].y = y;
        m_pointCloud->points[i].z = z;
    }

    //cout << cloud_normals->points.size() << endl;

}

template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::calcNormals()
{
    VertexT center = this->m_boundingBox.getCentroid();

    // Estimate normals
    cout << timestamp << "Estimating normals" << endl;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (m_pointCloud);
    ne.setViewPoint(center.x, center.y, center.z);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    m_kdTree = pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
    ne.setSearchMethod (m_kdTree);

    // Output datasets
    m_pointNormals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    //ne.setKSearch(this->m_kn);
    ne.setKSearch(this->m_kn);

    // Compute the features
    ne.compute (*m_pointNormals);
    cout << timestamp << "Normal estimation done" << endl;
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

    // Check number of found neighbours
    if(res != k)
    {
        cout << timestamp << "PCLPointCloudManager::getkClosestVertices() : Warning: Number of found neighbours != k_n." << endl;
    }

    // Parse result
    for(int i = 0; i < res; i++)
    {
        int index = k_indices[i];
        if(this->m_colors != 0)
        	nb.push_back(VertexT(m_pointCloud->points[index].x,
                             m_pointCloud->points[index].y,
                             m_pointCloud->points[index].z,
                             this->m_colors[index][0],
                             this->m_colors[index][1],
                             this->m_colors[index][2]));
        else
        	nb.push_back(VertexT(m_pointCloud->points[index].x,
        	                             m_pointCloud->points[index].y,
        	                             m_pointCloud->points[index].z));
    }
}


template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::getkClosestNormals(const VertexT &n,
         const size_t &k, vector<NormalT> &nb)
{
      std::vector< int > k_indices;
      std::vector< float > k_distances;

      pcl::PointXYZ qp;
      qp.x = n.x;
      qp.y = n.y;
      qp.z = n.z;

      // Query tree
      int res = m_kdTree->nearestKSearch(qp, k, k_indices, k_distances);

      // Check number of found neighbours
      if(res != k)
      {
          cout << timestamp << "PCLPointCloudManager::getkClosestNormals() : Warning: Number of found neighbours != k_n." << endl;
      }

      // Parse result
      for(int i = 0; i < res; i++)
      {
          int index = k_indices[i];
          nb.push_back(NormalT(m_pointNormals->points[index].data_c[0],
                               m_pointNormals->points[index].data_c[1],
                               m_pointNormals->points[index].data_c[2]));
      }
}

template<typename VertexT, typename NormalT>
float PCLPointCloudManager<VertexT, NormalT>::distance(VertexT v)
{
    std::vector< int > k_indices;
    std::vector< float > k_distances;

    pcl::PointXYZ qp;
    qp.x = v.x;
    qp.y = v.y;
    qp.z = v.z;

    // Query tree
    int res = m_kdTree->nearestKSearch(qp, this->m_kd, k_indices, k_distances);

    // Round distance
    float q_distance = 0.0f;
    NormalT n;
    VertexT c;
    for(int i = 0; i < res; i++)
    {
        int ind = k_indices[i];
        NormalT tmp_n = NormalT(
                m_pointNormals->points[ind].normal[0],
                m_pointNormals->points[ind].normal[1],
                m_pointNormals->points[ind].normal[2]
                );

        VertexT tmp_p = VertexT(
                m_pointCloud->points[ind].x,
                m_pointCloud->points[ind].y,
                m_pointCloud->points[ind].z
                );


        //cout << ind << " / " << m_pointNormals->points.size() << " " << tmp << endl;
        n += tmp_n;
        c += tmp_p;

        q_distance += k_distances[i];
    }
    n /= res;
    c /= res;

    return (v - c) * n;

}

} // namespace lssr
