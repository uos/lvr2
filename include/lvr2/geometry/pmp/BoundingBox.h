// Copyright 2013-2021 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see PMP_LICENSE.txt for details.

#pragma once

#include "Types.h"

namespace pmp {

//! Simple class for representing a bounding box.
//! \ingroup core
class BoundingBox
{
public:
    //! Construct infinite/invalid bounding box.
    BoundingBox()
        : min_(Point::Constant(std::numeric_limits<Scalar>::max())),
          max_(Point::Constant(-std::numeric_limits<Scalar>::max()))
    {
    }

    //! Construct from min and max points.
    BoundingBox(const Point& min, const Point& max) : min_(min), max_(max) {}

    //! Add point to the bounding box.
    BoundingBox& operator+=(const Point& p)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (p[i] < min_[i])
                min_[i] = p[i];
            if (p[i] > max_[i])
                max_[i] = p[i];
        }
        return *this;
    }

    //! Add two bounding boxes.
    BoundingBox& operator+=(const BoundingBox& bb)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (bb.min_[i] < min_[i])
                min_[i] = bb.min_[i];
            if (bb.max_[i] > max_[i])
                max_[i] = bb.max_[i];
        }
        return *this;
    }

    //! Get min point.
    Point& min() { return min_; }
    //! Get min point.
    const Point& min() const { return min_; }

    //! Get max point.
    Point& max() { return max_; }
    //! Get max point.
    const Point& max() const { return max_; }

    //! Get center point.
    Point center() const { return 0.5f * (min_ + max_); }

    //! Get index of longest axis (0=x, 1=y, 2=z).
    size_t longest_axis() const
    {
        Point size = max_ - min_;
        return std::max_element(size.data(), size.data() + 3) - size.data();
    }

    //! Get length of longest axis.
    Scalar longest_axis_size() const
    {
        Point size = max_ - min_;
        return *std::max_element(size.data(), size.data() + 3);
    }

    //! Indicate if the bounding box is empty.
    bool is_empty() const
    {
        return (max_[0] < min_[0] || max_[1] < min_[1] || max_[2] < min_[2]);
    }

    //! Get the size of the bounding box.
    Scalar size() const { return is_empty() ? 0.0 : (max_ - min_).norm(); }

    //! Indicate if p is inside the bounding box.
    bool contains(const Point& p)
    {
        for (int i = 0; i < 3; ++i)
            if (p[i] < min_[i] || p[i] > max_[i])
                return false;
        return true;
    }

    //! Indicate if bb overlaps this bounding box.
    bool overlaps(const BoundingBox& bb)
    {
        for (int i = 0; i < 3; ++i)
            if (bb.max_[i] < min_[i] || bb.min_[i] > max_[i])
                return false;
        return true;
    }
    BoundingBox overlap(const BoundingBox& bb) const
    {
        BoundingBox result;
        for (int i = 0; i < 3; ++i)
        {
            result.min_[i] = std::max(min_[i], bb.min_[i]);
            result.max_[i] = std::min(max_[i], bb.max_[i]);
        }
        return result;
    }

    std::array<BoundingBox, 2> split(int axis, float threshold) const
    {
        BoundingBox bb1(*this), bb2(*this);
        bb1.max_[axis] = threshold;
        bb2.min_[axis] = threshold;
        return {bb1, bb2};
    }

private:
    Point min_, max_;
};

// Add + operator for BoundingBox in openMP reductions.
#pragma omp declare reduction(+: BoundingBox : omp_out += omp_in ) \
           initializer(omp_priv(BoundingBox()))

} // namespace pmp
