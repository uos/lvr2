#pragma once

#include <memory>

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/geometry/Transformable.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include <QTreeWidgetItem>

namespace lvr2 {

namespace qttf {

Matrix4<BaseVector<float> > getTransformation(
    QTreeWidgetItem* from,
    QTreeWidgetItem* to = NULL
);

PointBufferPtr transform(
    PointBufferPtr pc_in,
    const Matrix4<BaseVector<float> > T
);

PointBufferPtr transform(
    PointBufferPtr pc_in,
    QTreeWidgetItem* from,
    QTreeWidgetItem* to = NULL
);

} // namespace qttf

} // namespace lvr2