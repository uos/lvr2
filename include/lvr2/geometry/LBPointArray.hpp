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

#ifndef __LBPOINTARRAY_HPP
#define __LBPOINTARRAY_HPP

#include <stdlib.h>
#include <iostream>
#include <limits>
#include <list>
#include <unordered_set>
#include <stdlib.h>

namespace lvr2
{

template<typename T>
struct LBPointArray {
    unsigned int width;
    unsigned int dim;
    T* elements;
};

// static helper methods

template<typename T>
static void mallocPointArray(LBPointArray<T>& m);

template<typename T>
static void generatePointArray(LBPointArray<T>& m, int width, int dim);

template<typename T>
static void generatePointArray(int id, LBPointArray<T>& m, int width, int dim);

template<typename T>
static void fillPointArrayWithSequence( LBPointArray<T>& m);

// Pool function
template<typename T>
static void fillPointArrayWithSequence(int id, LBPointArray<T>& m);

template<typename T>
static void copyVectorInterval(LBPointArray<T>& in, int start, int end, LBPointArray<T>& out);

template<typename T>
static void copyDimensionToPointArray(LBPointArray<T>& in, int dim, LBPointArray<T>& out);

template<typename T>
static void splitPointArray(LBPointArray<T>& I, LBPointArray<T>& I_L, LBPointArray<T>& I_R);

template<typename T, typename U>
static bool checkSortedIndices(const LBPointArray<T>& V,
        const LBPointArray<U>& sorted_indices,
        unsigned int dim,
        int n=0);


template<typename T, typename U>
static void splitPointArrayWithValue(const LBPointArray<T>& V,
        const LBPointArray<U>& I, LBPointArray<U>& I_L, LBPointArray<U>& I_R,
        int current_dim, T value,
        T& deviation_left, T& deviation_right, const unsigned int& orig_dim,
        const std::list<U>& critical_indices_left, const std::list<U>& critical_indices_right);

template<typename T, typename U>
static void splitPointArrayWithValueSet(const LBPointArray<T>& V,
        const LBPointArray<U>& I, LBPointArray<U>& I_L, LBPointArray<U>& I_R,
        int current_dim, T value,
        T& deviation_left, T& deviation_right, const unsigned int& orig_dim,
        const std::unordered_set<U>& critical_indices_left,
        const std::unordered_set<U>& critical_indices_right);


template<typename T>
static unsigned int checkNumberOfBiggerValues(LBPointArray<T>& V, unsigned int dim, T split);

static unsigned int checkNumberOfSmallerEqualValues(LBPointArray<float>& V, unsigned int dim, float split);


// SORT FUNCTIONS THREADED
template<typename T, typename U>
static void mergeHostWithIndices(T* a, U* b, unsigned int i1, unsigned int j1,
        unsigned int i2, unsigned int j2, int limit);

template<typename T, typename U>
static void naturalMergeSort(LBPointArray<T>& in, int dim, LBPointArray<U>& indices, LBPointArray<T>& m, int limit=-1);

template<typename T, typename U>
static void sortByDim(LBPointArray<T>& V, int dim, LBPointArray<U>& indices, LBPointArray<T>& values);

template<typename T, typename U>
static void generateAndSort(int id, LBPointArray<T>& vertices, LBPointArray<U>* indices_sorted,
        LBPointArray<T>* values_sorted, int dim);

} /* namespace lvr2 */

#include "LBPointArray.tcc"

#endif // !__POINTARRAY_HPP
