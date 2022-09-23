#include <lvr2/registration/FPFH.hpp>
#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/reconstruction/SearchTreeFlann.hpp>
#include <lvr2/util/Timestamp.hpp>
#include <lvr2/geometry/BaseVector.hpp>

#include <iostream>

namespace lvr2
{

static Eigen::Vector4f ComputePairFeatures(const Eigen::Vector3f &p1,
                                           const Eigen::Vector3f &n1,
                                           const Eigen::Vector3f &p2,
                                           const Eigen::Vector3f &n2) 
{
    Eigen::Vector4f result;
    Eigen::Vector3f dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return Eigen::Vector4f::Zero();
    }
    auto n1_copy = n1;
    auto n2_copy = n2;
    double angle1 = n1_copy.dot(dp2p1) / result(3);
    double angle2 = n2_copy.dot(dp2p1) / result(3);
    if (acos(fabs(angle1)) > acos(fabs(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }
    auto v = dp2p1.cross(n1_copy);
    double v_norm = v.norm();
    if (v_norm == 0.0) {
        return Eigen::Vector4f::Zero();
    }
    v /= v_norm;
    auto w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

FPFHFeaturePtr computeInitialFeatures(const PointBufferPtr pointBuffer, SearchTreePtr<BaseVector<float>> tree, size_t k)
{
    auto feature = Eigen::MatrixXf(33, pointBuffer->numPoints());


    auto points_opt = pointBuffer->getChannel<float>("points");
    auto normals_opt = pointBuffer->getChannel<float>("normals");

    if(points_opt && normals_opt)
    {
        auto points = *points_opt;
        auto normals = *points_opt;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < pointBuffer->numPoints(); i++)
        {
            BaseVector<float> point(points[i][0], points[i][1], points[i][2]);
            BaseVector<float> normal(normals[i][0], normals[i][1], normals[i][2]);

            std::vector<size_t> indices;
            std::vector<float> distances;
            if (tree->kSearch(point, k, indices, distances) > 1)
            {
                Eigen::Vector3f query_point(point.x, point.y, point.z);
                Eigen::Vector3f query_normal(normal.x, normal.y, normal.z);

                // Only compute SPFH feature when a point has neighbors
                double hist_incr = 100.0 / (float)(indices.size() - 1);
                for (size_t k = 1; k < indices.size(); k++)
                {
                    // Skip the point itself, compute histogram
                    size_t current = indices[k];
                    Eigen::Vector3f current_point(
                        points[current][0], 
                        points[current][1], 
                        points[current][2]);

                    Eigen::Vector3f current_normal(
                        normals[current][0], 
                        normals[current][1], 
                        normals[current][2]);

                    auto pf = ComputePairFeatures(query_point, query_normal,
                                                  current_point,
                                                  current_normal);
                    int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
                    if (h_index < 0)
                        h_index = 0;
                    if (h_index >= 11)
                        h_index = 10;
                    feature(h_index, i) += hist_incr;
                    h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
                    if (h_index < 0)
                        h_index = 0;
                    if (h_index >= 11)
                        h_index = 10;
                    feature(h_index + 11, i) += hist_incr;
                    h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
                    if (h_index < 0)
                        h_index = 0;
                    if (h_index >= 11)
                        h_index = 10;
                    feature(h_index + 22, i) += hist_incr;
                }
            }
        }
    }
    return FPFHFeaturePtr(new Eigen::MatrixXf(feature));
}

FPFHFeaturePtr computeFPFHFeatures(const PointBufferPtr pointBuffer, SearchTreePtr<BaseVector<float>> tree, size_t k)
{
    Eigen::MatrixXf feature(33, pointBuffer->numPoints());
    auto spfh = computeInitialFeatures(pointBuffer, tree, k);
    if (spfh == nullptr)
    {
        std::cout << timestamp << "Internal error: SPFH feature vector is nullptr." << std::endl;
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < pointBuffer->numPoints(); i++)
    {
        auto points = pointBuffer->getChannel<float>("points");
        auto normals = pointBuffer->getChannel<float>("normals");

        BaseVector<float> point( (*points)[i][0], (*points)[i][1], (*points)[i][2]);
        std::vector<size_t> indices;
        std::vector<float> distance2;
        if (tree->kSearch(point, k, indices, distance2) > 1)
        {
            double sum[3] = {0.0, 0.0, 0.0};
            for (size_t k = 1; k < indices.size(); k++)
            {
                // skip the point itself
                double dist = distance2[k];
                if (dist == 0.0)
                    continue;
                for (int j = 0; j < 33; j++)
                {
                    double val = (*spfh)(j, indices[k]) / dist;
                    sum[j / 11] += val;
                    feature(j, i) += val;
                }
            }
            for (int j = 0; j < 3; j++)
                if (sum[j] != 0.0)
                    sum[j] = 100.0 / sum[j];
            for (int j = 0; j < 33; j++)
            {
                feature(j, i) *= sum[j / 11];
                // The commented line is the fpfh function in the paper.
                // But according to PCL implementation, it is skipped.
                // Our initial test shows that the full fpfh function in the
                // paper seems to be better than PCL implementation. Further
                // test required.
                feature(j, i) += (*spfh)(j, i);
            }
        }
    }
    return FPFHFeaturePtr(new Eigen::MatrixXf(feature));
}

FPFHFeaturePtr computeFPFHFeatures(const PointBufferPtr pointCloud, size_t k)
{
    if (!pointCloud->hasNormals()) 
    {
        std::cout << timestamp << "FPFH Failed because input point cloud has no normals" << std::endl;
    }

    SearchTreePtr<BaseVector<float>> tree(new SearchTreeFlann<BaseVector<float>>(pointCloud));

    return computeFPFHFeatures(pointCloud, tree, k);

}

} // namespace lvr2
