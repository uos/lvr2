#ifndef __LBKDTREE_HPP
#define __LBKDTREE_HPP


#include "lvr/geometry/LBPointArray.hpp"

#include <ctpl.h>

#include <stdlib.h>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include <list>
#include <unordered_set>

namespace lvr
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


}  /* namespace lvr */

#endif // !__LBKDTREE_HPP
