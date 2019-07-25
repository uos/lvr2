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
 * GraphSlam.hpp
 *
 *  @date July 22, 2019
 *  @author Malte Hillmann
 */
#ifndef GRAPHSLAM_HPP_
#define GRAPHSLAM_HPP_

#include "Scan.hpp"
#include "SlamOptions.hpp"

#include <Eigen/Sparse>

using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using GraphMatrix = Eigen::SparseMatrix<float>;
using GraphVector = Eigen::VectorXf;

namespace lvr2
{

class GraphSlam
{

public:
    GraphSlam(const SlamOptions* options);

    virtual ~GraphSlam() = default;

    void addEdge(int start, int end);

    void doGraphSlam(vector<ScanPtr>& scans, int last);

protected:

    void eulerCovariance(ScanPtr a, ScanPtr b, Matrix6f& outMat, Vector6f& outVec) const;
    void fillEquation(const vector<ScanPtr>& scans, GraphMatrix& mat, GraphVector& vec);

    const SlamOptions*     m_options;

    vector<pair<int, int>> m_graph;
};

} /* namespace lvr2 */

#endif /* GRAPHSLAM_HPP_ */
