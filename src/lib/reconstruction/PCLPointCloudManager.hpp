/**
 * PCLPointCloudManager.h
 *
 *  @date 08.08.2011
 *  @author Thomas Wiemann
 */

#ifndef PCLPOINTCLOUDMANAGER_H_
#define PCLPOINTCLOUDMANAGER_H_

#include "PointCloudManager.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

namespace lssr
{

template<typename VertexT, typename NormalT>
class PCLPointCloudManager : public PointCloudManager<VertexT, NormalT>
{
public:
    PCLPointCloudManager(string filename);

    virtual ~PCLPointCloudManager();


    /**
     * @brief Returns the k closest neighbor vertices to a given query point
     *
     * @param v         A query vertex
     * @param k         The (max) number of returned closest points to v
     * @param nb        A vector containing the determined closest points
     */
    virtual void getkClosestVertices(const VertexT &v,
            const size_t &k, vector<VertexT> &nb);

    /**
     * @brief Returns the k closest neighbor normals to a given query point
     *
     * @param n         A query vertex
     * @param k         The (max) number of returned closest points to v
     * @param nb        A vector containing the determined closest normals
     */
    virtual void getkClosestNormals(const VertexT &n,
            const size_t &k, vector<NormalT> &nb);

    virtual float distance(VertexT v);

private:
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr m_kdTree;
};

} // namespace lssr

#include "PCLPointCloudManager.tcc"

#endif /* PCLPOINTCLOUDMANAGER_H_ */
