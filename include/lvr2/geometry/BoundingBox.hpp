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
 * BoundingBox.hpp
 *
 *  @date 22.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#pragma once 

#include <cmath>
#include <ostream>

#include "lvr2/io/LineReader.hpp"

namespace lvr2
{

/**
 * @brief A dynamic bounding box class.
 */
template<typename BaseVecT>
class BoundingBox
{
public:

    using VectorType = BaseVecT;
    /**
     * @brief Default constructor
     */
    BoundingBox();

    /**
     * @brief Constructs a bounding box with from the given vertices
     *
     * @param v1        Lower left corner of the BoundingBox
     * @param v2        Upper right corner of the BoundingBox
     * @return
     */
    template<typename T>
    BoundingBox(T v1, T v2);


    /**
     *
     * @brief Constructs a bounding box for a given point cloud
     *
     * @param plyPath path of the point cloud
     */
    BoundingBox(std::string plyPath);

    /**
     * @brief Expands the bounding box if the given Vector \ref{v} is
     *        outside the current volume
     *
     * @param v         A 3d Vector
     */
    template<typename T>
    inline void expand(T v);

    /**
     * @brief  Calculates the surrounding bounding box of the current
     *         volume and the other given bounding box
     *
     * @param bb        Another bounding box
     */
    inline void expand(const BoundingBox<BaseVecT>& bb);

    /**
     * @brief Returns the radius of the current volume, i.e. the distance
     *        between the centroid and the most distant corner from this
     *        Vector.
     */
    typename BaseVecT::CoordType getRadius() const;

    /**
     * @brief Returns true if the bounding box has been expanded before or
     *        was initialized with a preset size.
     */
    bool isValid() const;

    /**
     * @brief Returns the center Vector of the bounding box.
     */
    BaseVecT getCentroid() const;

    /**
     * @brief check if current volume overlap with a given bounding box
     *
     * @param bb Another bounding box
     * @return true if both boxes overlap
     */
    bool overlap(const BoundingBox<BaseVecT>& bb);

    /**
     * @brief Returns the longest side of the bounding box
     */
    typename BaseVecT::CoordType getLongestSide() const;

    /**
     * @brief Returns the x-size of the bounding box
     */
    typename BaseVecT::CoordType getXSize() const;

    /**
     * @brief Returns the y-size of the bounding box
     */
    typename BaseVecT::CoordType getYSize() const;

    /**
     * @brief Returns the z-size of the bounding box
     */
    typename BaseVecT::CoordType getZSize() const;

    /**
     * @brief Returns the volume of the bounding box
     * @return
     */
    typename BaseVecT::CoordType getVolume() const;

    /**
     * @brief Returns the upper right coordinates
     */
    BaseVecT getMax() const;

    /**
     * @brief Returns the lower left coordinates
     */
    BaseVecT getMin() const;

private:
    /// The lower left Vector of the bounding box
    BaseVecT m_min;

    /// The upper right Vector of the bounding box
    BaseVecT m_max;

    /// The center Vector of the bounding box
    BaseVecT m_centroid;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const BoundingBox<BaseVecT>& bb)
{
    os << "Bounding Box[min: " << bb.getMin() << "             max: " <<  bb.getMax();
    os << "             dimension: " << bb.getXSize() << ", " << bb.getYSize() << ", "
       << bb.getZSize() << "]" << std::endl;
    return os;
}

} // namespace lvr2

#include "lvr2/geometry/BoundingBox.tcc"

