#pragma once

#include <memory>

#include <Eigen/Dense>

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/geometry/Transformable.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include <QTreeWidgetItem>

namespace lvr2 {

namespace qttf {

Eigen::Matrix<float, 4, 4, Eigen::RowMajor> getTransformation(
    QTreeWidgetItem* from,
    QTreeWidgetItem* to = NULL
);

PointBufferPtr transform(
    PointBufferPtr pc_in,
    const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>& T
);

PointBufferPtr transform(
    PointBufferPtr pc_in,
    QTreeWidgetItem* from,
    QTreeWidgetItem* to = NULL
);

} // namespace qttf

} // namespace lvr2