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

/**
 * LBKdTree.hpp
 *
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include <memory>

#include "lvr2/geometry/LBPointArray.hpp"

namespace lvr2
{

/**
 * @brief The LBKdTree class implements a left-balanced array-based index kd-tree.
 *          Left-Balanced: minimum memory
 *          Array-Based: Good for GPU - Usage
 */
class LBKdTree {
public:

    /**
     * @brief Construct a new LBKdTree object. Note that the tree only stores indices into `vertices`, which means
     *        that `vertices` has to be kept in memory and in the sam order.
     * 
     * @param vertices Array of vertices to build the tree from.
     * @param numThreads Number of threads to use for the tree construction. -1 for all cores.
     */
    LBKdTree(const LBPointArray<float>& vertices, int numThreads = -1);

    ~LBKdTree() = default;

    /**
     * @brief Get the kd-tree Values.
     *
     * Values indicate the split value of the kd-tree for all nodes.
     * For leaves, the value is the index of the point in the vertices array.
     */
    std::shared_ptr<LBPointArray<float>> getKdTreeValues()
    {
        return m_values;
    }

    /**
     * @brief Get the kd-tree Splits.
     *
     * Splits indicate the axis that each node splits on.
     *
     * This array is shorter than the values array, because there are no splits for leaves.
     */
    std::shared_ptr<LBPointArray<unsigned char>> getKdTreeSplits()
    {
        return m_splits;
    }

private:

    /**
     * @brief Build the kd-tree.
     * 
     * @param position the index in the values/splits array of the current node.
     * @param indicesStart start of the range of indices to build this node from.
     * @param indicesEnd end of the range of indices to build this node from.
     * @param vertices the vertices to build the tree from.
     */
    void generateKdTreeRecursive(uint position, uint* indicesStart, uint* indicesEnd, const LBPointArray<float>& vertices);

    std::shared_ptr<LBPointArray<unsigned char>> m_splits;
    std::shared_ptr<LBPointArray<float>> m_values;
};

}  /* namespace lvr2 */
