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

#include <stdio.h>

#include <iostream>
#include "lvr2/reconstruction/LBKdTree.hpp"
#include "lvr2/config/lvropenmp.hpp"

namespace lvr2
{

// Static variables

ctpl::thread_pool* LBKdTree::pool;
int LBKdTree::st_num_threads = 8;
int LBKdTree::st_depth_threads = 3;

/// Public

LBKdTree::LBKdTree( LBPointArray<float>& vertices, int num_threads) {
    this->m_values = boost::shared_ptr<LBPointArray<float> >(new LBPointArray<float>);
    this->m_splits = boost::shared_ptr<LBPointArray<unsigned char> >(new LBPointArray<unsigned char>);
    st_num_threads = num_threads;
    st_depth_threads = static_cast<int>(log2(st_num_threads));
    pool = new ctpl::thread_pool(OpenMPConfig::getNumThreads());
    this->generateKdTree(vertices);
}

LBKdTree::~LBKdTree() {
    if(pool)
    {
        delete pool;
    }
}

void LBKdTree::generateKdTree(LBPointArray<float> &vertices) {

    LBPointArray<unsigned int>* indices_sorted =
        (LBPointArray<unsigned int>*)malloc(vertices.dim * sizeof(LBPointArray<unsigned int>) );
    LBPointArray<float>* values_sorted =
        (LBPointArray<float>*)malloc(vertices.dim * sizeof(LBPointArray<float>) );

    for(unsigned int i=0; i< vertices.dim; i++)
    {
        pool->push(generateAndSort<float, unsigned int>, vertices, indices_sorted, values_sorted, i);
        //generateAndSort<float, unsigned int>(0, vertices, indices_sorted, values_sorted, i);
    }

    pool->stop(true);
    delete pool;
    pool = new ctpl::thread_pool(OpenMPConfig::getNumThreads());


    this->generateKdTreeArray(vertices, indices_sorted, vertices.dim);

    for(unsigned int i=0; i<vertices.dim;i++)
    {
        free(values_sorted[i].elements);
    }
}

boost::shared_ptr<LBPointArray<float> > LBKdTree::getKdTreeValues() {
    return this->m_values;
}

boost::shared_ptr<LBPointArray<unsigned char> > LBKdTree::getKdTreeSplits() {
    return this->m_splits;
}

/// Private

void LBKdTree::generateKdTreeArray(LBPointArray<float>& V,
        LBPointArray<unsigned int>* sorted_indices, int max_dim) {

    // DEBUG CHECK

    int first_split_dim = -1;
    float best_deviation = -1.0;
    
    for(int i=0; i<V.dim; i++)
    {
        float deviation = V.elements[static_cast<unsigned int>(
                            sorted_indices[i].elements[sorted_indices[i].width-1]+0.5)* V.dim + i]
                        -  V.elements[static_cast<unsigned int>(
                            sorted_indices[i].elements[i]+0.5)* V.dim + i] ;
        
        if(deviation > best_deviation)
        {
            best_deviation = deviation;
            first_split_dim = i;
        }
    }


    unsigned int size;
    int max_tree_depth;

    max_tree_depth = static_cast<int>( log2f(V.width - 1 ) + 2.0 ) ;

    if (V.width == 1)
    {
        max_tree_depth = 1;
    }

    size = V.width * 2 - 1;

    // std::cout << "size values: " << size << std::endl;
    this->m_values->elements = (float*)malloc(sizeof(float) * size );
    this->m_values->width = size;
    this->m_values->dim = 1;

    unsigned int size_splits = size - V.width;
    // std::cout << "size splits: " << size_splits << std::endl;
    this->m_splits->elements = (unsigned char*)malloc(sizeof(unsigned char) * size_splits );
    this->m_splits->width = size_splits;
    this->m_splits->dim = 1;

    LBPointArray<float>* value_ptr = this->m_values.get();
    LBPointArray<unsigned char>* splits_ptr = this->m_splits.get();
    //start real generate
    generateKdTreeRecursive(0, V, sorted_indices, first_split_dim,
            max_dim, value_ptr, splits_ptr ,size, max_tree_depth, 0, 0);

    pool->stop(true);
    delete pool;
    pool = new ctpl::thread_pool(OpenMPConfig::getNumThreads());
}

void LBKdTree::fillCriticalIndices(const LBPointArray<float>& V,
        LBPointArray<unsigned int>& sorted_indices, unsigned int current_dim,
        float split_value, unsigned int split_index,
        std::list<unsigned int>& critical_indices_left,
        std::list<unsigned int>& critical_indices_right)
{
    critical_indices_left.push_back( sorted_indices.elements[split_index] );

    unsigned int iterator;
    // nach links
    for(iterator = split_index-1;
        iterator < sorted_indices.width
        && V.elements[ sorted_indices.elements[iterator] * V.dim + current_dim] == split_value;
        iterator--)
    {
        critical_indices_left.push_back( sorted_indices.elements[iterator] );
    }

    // nach rechts
    for(iterator = split_index+1;
        iterator < sorted_indices.width
        && V.elements[ sorted_indices.elements[iterator] * V.dim + current_dim] == split_value;
        iterator++)
    {
        critical_indices_right.push_back( sorted_indices.elements[iterator] );
    }

}

void LBKdTree::fillCriticalIndicesSet(const LBPointArray<float>& V,
        LBPointArray<unsigned int>& sorted_indices, unsigned int current_dim,
        float split_value, unsigned int split_index,
        std::unordered_set<unsigned int>& critical_indices_left,
        std::unordered_set<unsigned int>& critical_indices_right)
{
    // push split index to the left
    critical_indices_left.insert(sorted_indices.elements[split_index]);

    unsigned int iterator;
    // to the left
    for(iterator = split_index-1;
        iterator < sorted_indices.width
        && V.elements[ sorted_indices.elements[iterator] * V.dim + current_dim] == split_value;
        iterator--)
    {
        critical_indices_left.insert( sorted_indices.elements[iterator] );
    }

    // to the right
    for(iterator = split_index+1;
        iterator < sorted_indices.width
        && V.elements[ sorted_indices.elements[iterator] * V.dim + current_dim] == split_value;
        iterator++)
    {
        critical_indices_right.insert( sorted_indices.elements[iterator] );
    }

}

void LBKdTree::generateKdTreeRecursive(int id, LBPointArray<float>& V,
        LBPointArray<unsigned int>* sorted_indices, int current_dim, int max_dim,
        LBPointArray<float> *values, LBPointArray<unsigned char> *splits ,
        int size, int max_tree_depth, int position, int current_depth)
{
    int left = position*2+1;
    int right = position*2+2;


    if( sorted_indices[current_dim].width <= 1 )
    {
        values->elements[position] = static_cast<float>(sorted_indices[current_dim].elements[0] );
    } else {
        /// split sorted_indices
        unsigned int indices_size = sorted_indices[current_dim].width;

        unsigned int v = pow( 2, static_cast<int>( log2(indices_size-1) ) );
        unsigned int left_size = indices_size - v/2;

        if( left_size > v )
        {
            left_size = v;
        }

        unsigned int right_size = indices_size - left_size;



        unsigned int split_index = static_cast<unsigned int>(
                sorted_indices[current_dim].elements[left_size-1] + 0.5
            );

        float split_value = V.elements[split_index * V.dim + current_dim ];

        // critical indices
        std::unordered_set<unsigned int> critical_indices_left;
        std::unordered_set<unsigned int> critical_indices_right;

        fillCriticalIndicesSet(V, sorted_indices[current_dim],
                current_dim, split_value, left_size-1, 
                critical_indices_left, critical_indices_right);

        //std::cout << "Split in dimension: " << current_dim << std::endl;
        values->elements[ position ] = split_value;
        splits->elements[ position ] = static_cast<unsigned char>(current_dim);


        LBPointArray<unsigned int> *sorted_indices_left =
            (LBPointArray<unsigned int>*)malloc( 3*sizeof(LBPointArray<unsigned int>) );
        LBPointArray<unsigned int> *sorted_indices_right =
            (LBPointArray<unsigned int>*)malloc( 3*sizeof(LBPointArray<unsigned int>) );

        int next_dim_left = -1;
        int next_dim_right = -1;
        float biggest_deviation_left = -1.0;
        float biggest_deviation_right = -1.0;

        for( int i=0; i<max_dim; i++ )
        {

            sorted_indices_left[i].width = left_size;
            sorted_indices_left[i].dim = 1;
            sorted_indices_left[i].elements = (unsigned int*)malloc( left_size * sizeof(unsigned int) );

            sorted_indices_right[i].width = right_size;
            sorted_indices_right[i].dim = 1;
            sorted_indices_right[i].elements = (unsigned int*)malloc( right_size * sizeof(unsigned int) );

            float deviation_left;
            float deviation_right;


            if( i == current_dim ){
                 splitPointArray<unsigned int>( sorted_indices[i],
                         sorted_indices_left[i],
                         sorted_indices_right[i]);


                 deviation_left = fabs(V.elements[sorted_indices_left[i].elements[left_size - 1] * V.dim + i ]
                                    - V.elements[sorted_indices_left[i].elements[0] * V.dim + i ]   );
                 deviation_right = fabs( V.elements[ sorted_indices_right[i].elements[right_size - 1]  * V.dim + i ]
                                    - V.elements[sorted_indices_right[i].elements[0]  * V.dim + i] );

            } else {
                

                // check for wrong value in sorted indices
                
                splitPointArrayWithValueSet<float, unsigned int>(V, sorted_indices[i],
                        sorted_indices_left[i], sorted_indices_right[i],
                        current_dim, split_value,
                        deviation_left, deviation_right, i,
                        critical_indices_left, critical_indices_right);

            }

            if(deviation_left > biggest_deviation_left )
            {
                biggest_deviation_left = deviation_left;
                next_dim_left = i;
            }

            if(deviation_right > biggest_deviation_right )
            {
                biggest_deviation_right = deviation_right;
                next_dim_right = i;
            }
        }

        //int next_dim = (current_dim+1)%max_dim;

        if(current_depth == st_depth_threads )
        {
            pool->push(generateKdTreeRecursive, V, sorted_indices_left,
                    next_dim_left, max_dim, values, splits, size, max_tree_depth,
                    left, current_depth + 1);

            pool->push(generateKdTreeRecursive, V, sorted_indices_right,
                    next_dim_right, max_dim, values, splits, size, max_tree_depth,
                    right, current_depth +1);
        } else {
            //std::cout<< "left " << current_dim << std::endl;
            generateKdTreeRecursive(0, V, sorted_indices_left, next_dim_left, max_dim,
                    values, splits, size, max_tree_depth, left, current_depth + 1);
            //std::cout << "right " << current_dim << std::endl;
            generateKdTreeRecursive(0, V, sorted_indices_right, next_dim_right, max_dim,
                    values, splits, size, max_tree_depth, right, current_depth +1);
        }

    }

    for(int i=0; i<max_dim; i++) {
        free(sorted_indices[i].elements );
    }
    free(sorted_indices);

}

} /* namespace lvr2 */
