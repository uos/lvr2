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
 * GraphSLAM.hpp
 *
 *  @date July 22, 2019
 *  @author Malte Hillmann
 */
#ifndef GRAPHSLAM_HPP_
#define GRAPHSLAM_HPP_

#include "SLAMScanWrapper.hpp"
#include "SLAMOptions.hpp"
#include "KDTree.hpp"

#include <Eigen/SparseCore>

namespace lvr2
{

/**
 * @brief finds Scans that are "close" to a Scan as determined by a Loopclosing strategy
 *
 * @param scans   A vector with all Scans
 * @param scan    The index of the scan
 * @param options The options on how to search
 * @param output  Will be filled with the indices of all close Scans
 *
 * @return true if any Scans were found, false otherwise
 */
bool findCloseScans(const std::vector<SLAMScanPtr>& scans, size_t scan, const SLAMOptions& options, std::vector<size_t>& output);

/**
 * @brief Wrapper class for running GraphSLAM on Scans
 */
class GraphSLAM
{
public:

    using GraphMatrix = Eigen::SparseMatrix<double>;
    using GraphVector = Eigen::VectorXd;
    using Graph = std::vector<std::pair<int, int>>;

    GraphSLAM(const SLAMOptions* options);

    virtual ~GraphSLAM() = default;

    /**
     * @brief runs the GraphSLAM algorithm
     *
     * @param scans The scans to work on
     * @param last  The index of the last Scan to consider. `scans` may be longer, but anything
     *              after `last` will be ignored
     */
    void doGraphSLAM(const std::vector<SLAMScanPtr>& scans, size_t last, const std::vector<bool>& new_scans = std::vector<bool>()) const;

protected:
    /**
     * @brief Creates a graph. An edge between nodes(scans) means posible overlap.
     * @param scans reference to a vector containing the SlamScanPtr
     * @param last number of the last considered scan
     * @param graph Outputs the created graph
     * */
    void createGraph(const std::vector<SLAMScanPtr>& scans, size_t last, Graph& graph) const;

    /**
     * @brief A function to fill the linear system mat * x = vec.
     * @param scans reference to a vector containing the SlamScanPtr
     * @param graph the graph created in the createGraph function
     * @param mat Outputs the GraphMatrix
     * @param vec Outputs the GraphVector
     * */
    void fillEquation(const std::vector<SLAMScanPtr>& scans, const Graph& graph, GraphMatrix& mat, GraphVector& vec) const;
    
    void eulerCovariance(KDTreePtr tree, SLAMScanPtr scan, Matrix6d& outMat, Vector6d& outVec) const;

    const SLAMOptions*     m_options;
};

} /* namespace lvr2 */

#endif /* GRAPHSLAM_HPP_ */
