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

#ifndef __LBKDTREE_HPP
#define __LBKDTREE_HPP


#include "lvr2/geometry/LBPointArray.hpp"

#include <ctpl.h>

#include <stdlib.h>
#include <math.h>
#include <list>
#include <unordered_set>

#include <boost/shared_ptr.hpp>

namespace lvr2
{

/**
 * @brief The LBKdTree class implements a left-balanced array-based index kd-tree.
 *          Left-Balanced: minimum memory
 *          Array-Based: Good for GPU - Usage
 */
class LBKdTree {
public:

    LBKdTree( LBPointArray<float>& vertices , int num_threads=8);

    ~LBKdTree();

    void generateKdTree( LBPointArray<float>& vertices );

    boost::shared_ptr<LBPointArray<float> > getKdTreeValues();

    boost::shared_ptr<LBPointArray<unsigned char> > getKdTreeSplits();

private:

    void generateKdTreeArray(LBPointArray<float>& V,
            LBPointArray<unsigned int>* sorted_indices,
            int max_dim);

    boost::shared_ptr<LBPointArray<float> > m_values;

    // split dim 4 dims per split_dim
    boost::shared_ptr<LBPointArray<unsigned char> > m_splits;

    // Static member

    static int st_num_threads;
    static int st_depth_threads;

    static ctpl::thread_pool *pool;

    static void fillCriticalIndices(const LBPointArray<float>& V,
            LBPointArray<unsigned int>& sorted_indices,
            unsigned int current_dim,
            float split_value, unsigned int split_index,
            std::list<unsigned int>& critical_indices_left,
            std::list<unsigned int>& critical_indices_right);

    static void fillCriticalIndicesSet(const LBPointArray<float>& V,
            LBPointArray<unsigned int>& sorted_indices,
            unsigned int current_dim,
            float split_value, unsigned int split_index,
            std::unordered_set<unsigned int>& critical_indices_left,
            std::unordered_set<unsigned int>& critical_indices_right);


    static void generateKdTreeRecursive(int id, LBPointArray<float>& V,
            LBPointArray<unsigned int>* sorted_indices, int current_dim, int max_dim,
            LBPointArray<float> *values, LBPointArray<unsigned char> *splits,
            int size, int max_tree_depth, int position, int current_depth);

    static void test(int id, LBPointArray<float>* sorted_indices);


};

}  /* namespace lvr2 */

#endif // !__LBKDTREE_HPP
