#include "qttf.hpp"

#include <iostream>

namespace lvr2 {

namespace qttf {

Eigen::Matrix<float, 4, 4, Eigen::RowMajor> getTransformation(
    QTreeWidgetItem* from,
    QTreeWidgetItem* to
)
{
    // default identity
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> ret = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Identity();

    QTreeWidgetItem* it = from;

    while(it != to || it != NULL)
    {
        auto transform_obj = dynamic_cast< Transformable* >(it);
        if(transform_obj)
        {
            ret = ret * transform_obj->getTransform();
        }
        it = it->parent();
    }

    return ret;
}

PointBufferPtr transform(
    PointBufferPtr pc_in,
    const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>& T
)
{
    // generate copy
    PointBufferPtr buffer_ptr = std::make_shared<PointBuffer>(pc_in->clone());
    
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T_trans = T;
    T_trans.transpose();

    // 1) transform points
    const size_t num_points = buffer_ptr->numPoints();
    BaseVector<float>* points_raw = reinterpret_cast<BaseVector<float>* >(
        buffer_ptr->getPointArray().get()
    );

    for(size_t i=0; i<num_points; i++)
    {
        points_raw[i] = T_trans * points_raw[i];
    }

    // 2) transform normals
    if(buffer_ptr->hasNormals())
    {
        Normal<float>* normals_raw = reinterpret_cast<Normal<float>* >(
            buffer_ptr->getNormalArray().get()
        );
        for(size_t i=0; i<num_points; i++)
        {
            normals_raw[i] = T_trans * normals_raw[i];
        }
    }


    return buffer_ptr;
}

PointBufferPtr transform(
    PointBufferPtr pc_in,
    QTreeWidgetItem* from,
    QTreeWidgetItem* to
)
{
    return transform(pc_in, getTransformation(from, to)); 
}

} // namespace qttf

} // namespace lvr2