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
 *  SearchTree.tcc
 *
 *       Created on: 02.01.2012
 *           Author: Florian Otte, Thomas Wiemann
 *
 */

#include "lvr2/io/Timestamp.hpp"

#include <iostream>
using std::cout;
using std::endl;

namespace lvr2 {


template<typename BaseVecT>
int SearchTree<BaseVecT>::kSearch(
    const BaseVecT &qp,
    int neighbours,
    std::vector<size_t>& indices
) const
{
    std::vector<CoordT> distances;
    return this->kSearch(qp, neighbours, indices, distances);
}

// template<typename BaseVecT>
// void SearchTree<BaseVecT>::setKi(int ki)
// {
//     m_ki = ki;
// }


// template<typename BaseVecT>
// void SearchTree<BaseVecT>::setKd(int kd)
// {
//     m_kd = kd;
// }

// template<typename BaseVecT>
// int SearchTree<BaseVecT>::getKi()
// {
//     return m_ki;
// }

// template<typename BaseVecT>
// int SearchTree<BaseVecT>::getKd()
// {
//     return m_kd;
// }

} // namespace
