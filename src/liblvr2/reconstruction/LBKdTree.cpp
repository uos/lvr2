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
 * LBKdTree.cpp
 *
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "lvr2/reconstruction/LBKdTree.hpp"
#include "lvr2/config/lvropenmp.hpp"

#include <math.h>
#include <algorithm>

namespace lvr2
{

LBKdTree::LBKdTree(const LBPointArray<float>& vertices, int numThreads)
{
    if (numThreads <= 0)
    {
        numThreads = OpenMPConfig::getNumThreads();
    }

    this->m_splits.reset(new LBPointArray<unsigned char>(), LBPointArrayDeleter<unsigned char>());
    generatePointArray(*this->m_splits, vertices.width - 1, 1);

    // m_values contains the split value for all nodes + the point index of all leaves
    this->m_values.reset(new LBPointArray<float>(), LBPointArrayDeleter<float>());
    generatePointArray(*this->m_values, this->m_splits->width + vertices.width, 1);

    // leaf values are indices => fill with invalid index to catch empty leafs
    float invalidIndex = vertices.width + 1;
    std::fill_n(this->m_values->elements, this->m_values->width, invalidIndex);

    std::unique_ptr<uint[]> indices(new uint[vertices.width]);
    for (uint i = 0; i < vertices.width; i++)
    {
        indices[i] = i;
    }

    uint* start = indices.get();
    uint* end = indices.get() + vertices.width;

    #pragma omp parallel num_threads(numThreads)
    #pragma omp single
    generateKdTreeRecursive(0, start, end, vertices);
}

void LBKdTree::generateKdTreeRecursive(uint position, uint* indicesStart, uint* indicesEnd, const LBPointArray<float>& vertices)
{
    size_t n = indicesEnd - indicesStart;

    if (n == 0)
    {
        return;
    }
    else if (n == 1)
    {
        this->m_values->elements[position] = *indicesStart;
        return;
    }

    if (position >= this->m_splits->width)
    {
        throw std::runtime_error("LBKdTree::generateKdTreeRecursive: position out of bounds");
    }

    float maxSideLength = -1;
    unsigned char splitDim = 0;
    for (unsigned char dim = 0; dim < vertices.dim; dim++)
    {
        float min = std::numeric_limits<float>::max();
        float max = -std::numeric_limits<float>::max();
        for (uint* i = indicesStart; i < indicesEnd; i++)
        {
            float value = vertices.elements[*i * vertices.dim + dim];
            if (value < min)
            {
                min = value;
            }
            if (value > max)
            {
                max = value;
            }
        }
        float sideLength = max - min;
        if (sideLength > maxSideLength)
        {
            maxSideLength = sideLength;
            splitDim = dim;
        }
    }

    // the largest power of 2 that is less than n
    size_t v = std::pow(2, std::floor(std::log2(n - 1)));

    // assure left-balanced-ness of the tree
    size_t leftN = std::min(n - v / 2, v);

    uint* leftEnd = indicesStart + leftN;
    std::nth_element(indicesStart, leftEnd, indicesEnd, [splitDim, &vertices](auto a, auto b) {
        return vertices.elements[a * vertices.dim + splitDim] < vertices.elements[b * vertices.dim + splitDim];
    });

    float splitValue = vertices.elements[*leftEnd * vertices.dim + splitDim];
    this->m_splits->elements[position] = splitDim;
    this->m_values->elements[position] = splitValue;

    uint leftPosition = position * 2 + 1;
    uint rightPosition = leftPosition + 1;

    if (n > 100)
    {
        #pragma omp task
        generateKdTreeRecursive(leftPosition, indicesStart, leftEnd, vertices);

        #pragma omp task
        generateKdTreeRecursive(rightPosition, leftEnd, indicesEnd, vertices);
    }
    else
    {
        generateKdTreeRecursive(leftPosition, indicesStart, leftEnd, vertices);
        generateKdTreeRecursive(rightPosition, leftEnd, indicesEnd, vertices);
    }
}

} /* namespace lvr2 */
