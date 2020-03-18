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
 * GraphSLAM.cpp
 *
 *  @date July 22, 2019
 *  @author Malte Hillmann
 */
#include "lvr2/registration/GraphSLAM.hpp"

#include <Eigen/SparseCholesky>

#include <math.h>

using namespace std;
using namespace Eigen;

namespace lvr2
{

/**
 * @brief Lists all numbers of scans near to the scan 
 * @param scans reference to a vector containing the SlamScanPtr
 * @param scan number of the current scan
 * @param options SlamOptions struct with all params
 * @param output Returns vector of the scan-numbers which ar defined as "close" 
 * */
bool findCloseScans(const vector<SLAMScanPtr>& scans, size_t scan, const SLAMOptions& options, vector<size_t>& output)
{
    if (scan < options.loopSize)
    {
        return false;
    }

    const SLAMScanPtr& cur = scans[scan];

    // closeLoopPairs not specified => use closeLoopDistance
    if (options.closeLoopPairs < 0)
    {
        double maxDist = std::pow(options.closeLoopDistance, 2);
        Vector3d pos = cur->getPosition();
        for (size_t other = 0; other < scan - options.loopSize; other++)
        {
            if ((scans[other]->getPosition() - pos).squaredNorm() < maxDist)
            {
                output.push_back(other);
            }
        }
    }
    else
    {
        // convert current Scan to KDTree for Pair search
        auto tree = KDTree::create(cur, options.maxLeafSize);

        size_t maxLen = 0;
        for (size_t other = 0; other < scan - options.loopSize; other++)
        {
            maxLen = max(maxLen, scans[other]->numPoints());
        }
        KDTree::Neighbor* neighbors = new KDTree::Neighbor[maxLen];

        for (size_t other = 0; other < scan - options.loopSize; other++)
        {
            size_t count = KDTree::nearestNeighbors(tree, scans[other], neighbors, options.slamMaxDistance);
            if (count >= options.closeLoopPairs)
            {
                output.push_back(other);
            }
        }

        delete[] neighbors;
    }

    return !output.empty();
}


/**
 * Conversion from Pose to Matrix representation in GraphSLAMs internally consistent Coordinate System
 * 
 * @brief Conversion from Pose to Matrix representation
 * */
void EulerToMatrix4(const Vector3d& pos, const Vector3d& theta, Matrix4d& mat);

/**
 * Conversion from Matrix to Pose representation in GraphSLAMs internally consistent Coordinate System
 * 
 * @brief Conversion from Matrix to Pose representation
 * */
void Matrix4ToEuler(const Matrix4d mat, Vector3d& rPosTheta, Vector3d& rPos);

GraphSLAM::GraphSLAM(const SLAMOptions* options)
    : m_options(options)
{
}

void GraphSLAM::doGraphSLAM(const vector<SLAMScanPtr>& scans, size_t last, const std::vector<bool>& new_scans) const
{
    // ignore first scan, keep last scan => n = last - 1 + 1
    size_t n = last;

    Graph graph;
    graph.reserve(n * n / 2);

    GraphMatrix A(6 * n, 6 * n);
    GraphVector B(6 * n);
    GraphVector X(6 * n);

    for (size_t iteration = 0;
            iteration < m_options->slamIterations;
            iteration++)
    {
        cout << "GraphSLAM Iteration " << iteration << " of " << m_options->slamIterations << endl;

        createGraph(scans, last, graph);

        // Construct the linear equation system A * X = B..
        fillEquation(scans, graph, A, B);

        graph.clear();

        X = SimplicialCholesky<GraphMatrix>().compute(A).solve(B);

        double sum_position_diff = 0.0;

        // Start with second Scan
        #pragma omp parallel for reduction(+:sum_position_diff) schedule(static)
        for (size_t i = 1; i <= last; i++)
        {
            if (new_scans.empty() || new_scans.at(i))
            {
                const SLAMScanPtr& scan = scans[i];

                // Now update the Poses
                Matrix6d Ha = Matrix6d::Identity();

                Matrix4d initialPose = scan->pose();
                Vector3d pos, theta;
                Matrix4ToEuler(initialPose, theta, pos);
                if (m_options->verbose)
                {
                    cout << "Start of " << i << ": " << pos.transpose() << ", " << theta.transpose() << endl;
                }

                double ctx, stx, cty, sty;

#ifndef __APPLE__
                sincos(theta.x(), &stx, &ctx);
                sincos(theta.y(), &sty, &cty);
#else
                __sincos(theta.x(), &stx, &ctx);
                __sincos(theta.y(), &sty, &cty);
#endif

                // Fill Ha
                Ha(0, 4) = -pos.z() * ctx + pos.y() * stx;
                Ha(0, 5) = pos.y() * cty * ctx + pos.z() * stx * cty;

                Ha(1, 3) = pos.z();
                Ha(1, 4) = -pos.x() * stx;
                Ha(1, 5) = -pos.x() * ctx * cty + pos.z() * sty;


                Ha(2, 3) = -pos.y();
                Ha(2, 4) = pos.x() * ctx;
                Ha(2, 5) = -pos.x() * cty * stx - pos.y() * sty;

                Ha(3, 5) = sty;

                Ha(4, 4) = stx;
                Ha(4, 5) = ctx * cty;

                Ha(5, 4) = ctx;
                Ha(5, 5) = -stx * cty;

                // Correct pose estimate
                Vector6d result = Ha.inverse() * X.block<6, 1>((i - 1) * 6, 0);

                // Update the Pose
                pos -= result.block<3, 1>(0, 0);
                theta -= result.block<3, 1>(3, 0);
                Matrix4d transform;
                EulerToMatrix4(pos, theta, transform);

                if (m_options->verbose)
                {
                    cout << "End: pos: " << pos.transpose() << "," << endl << "theta: " << theta.transpose() << endl;
                }

                transform = transform * initialPose.inverse();

                scan->transform(transform, m_options->createFrames, FrameUse::GRAPHSLAM);

                sum_position_diff += result.block<3, 1>(0, 0).norm();
            }
        }

        if (m_options->createFrames)
        {
            // add Frames to unused Scans
            scans[0]->addFrame(FrameUse::GRAPHSLAM);
            for (size_t i = last + 1; i < scans.size(); i++)
            {
                scans[i]->addFrame(FrameUse::INVALID);
            }
        }

        cout << "Sum of Position differences = " << sum_position_diff << endl;

        double avg = sum_position_diff / n;
        if (avg < m_options->slamEpsilon)
        {
            break;
        }
    }
}

void GraphSLAM::createGraph(const vector<SLAMScanPtr>& scans, size_t last, Graph& graph) const
{
    graph.clear();

    for (size_t i = 1; i <= last; i++)
    {
        graph.push_back(make_pair(i - 1, i));
    }

    vector<size_t> others;
    for (size_t i = m_options->loopSize; i <= last; i++)
    {
        findCloseScans(scans, i, *m_options, others);

        for (size_t other : others)
        {
            graph.push_back(make_pair(other, i));
        }

        others.clear();
    }
}

void GraphSLAM::fillEquation(const vector<SLAMScanPtr>& scans, const Graph& graph, GraphMatrix& mat, GraphVector& vec) const
{
    // Cache all KDTrees
    map<size_t, KDTreePtr> trees;
    for (size_t i = 0; i < graph.size(); i++)
    {
        size_t a = graph[i].first;
        if (trees.find(a) == trees.end())
        {
            auto tree = KDTree::create(scans[a], m_options->maxLeafSize);
            trees.insert(make_pair(a, tree));
        }
    }

    vector<pair<Matrix6d, Vector6d>> coeff(graph.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < graph.size(); i++)
    {
        int a, b;
        std::tie(a, b) = graph[i];

        KDTreePtr tree  = trees[a];
        SLAMScanPtr scan = scans[b];

        Matrix6d coeffMat;
        Vector6d coeffVec;
        eulerCovariance(tree, scan, coeffMat, coeffVec);

        coeff[i] = make_pair(coeffMat, coeffVec);
    }

    trees.clear();

    map<pair<int, int>, Matrix6d> result;

    mat.setZero();
    vec.setZero();

    for (size_t i = 0; i < graph.size(); i++)
    {
        int a, b;
        std::tie(a, b) = graph[i];

        Matrix6d coeffMat;
        Vector6d coeffVec;
        std::tie(coeffMat, coeffVec) = coeff[i];

        // first scan is not part of Matrix => ignore any a or b of 0

        int offsetA = (a - 1) * 6;
        int offsetB = (b - 1) * 6;

        if (offsetA >= 0)
        {
            vec.block<6, 1>(offsetA, 0) += coeffVec;
            auto key = make_pair(offsetA, offsetA);
            auto found = result.find(key);
            if (found == result.end())
            {
                result.insert(make_pair(key, coeffMat));
            }
            else
            {
                found->second += coeffMat;
            }
        }
        if (offsetB >= 0)
        {
            vec.block<6, 1>(offsetB, 0) -= coeffVec;
            auto key = make_pair(offsetB, offsetB);
            auto found = result.find(key);
            if (found == result.end())
            {
                result.insert(make_pair(key, coeffMat));
            }
            else
            {
                found->second += coeffMat;
            }
        }
        if (offsetA >= 0 && offsetB >= 0)
        {
            auto key = make_pair(offsetA, offsetB);
            auto found = result.find(key);
            if (found == result.end())
            {
                result.insert(make_pair(key, -coeffMat));
            }
            else
            {
                found->second -= coeffMat;
            }

            key = make_pair(offsetB, offsetA);
            found = result.find(key);
            if (found == result.end())
            {
                result.insert(make_pair(key, -coeffMat));
            }
            else
            {
                found->second -= coeffMat;
            }
        }
    }

    vector<Triplet<double>> triplets;
    triplets.reserve(result.size() * 6 * 6);

    int x, y;
    for (auto& e : result)
    {
        tie(x, y) = e.first;
        Matrix6d& m = e.second;
        for (int dx = 0; dx < 6; dx++)
        {
            for (int dy = 0; dy < 6; dy++)
            {
                triplets.push_back(Triplet<double>(x + dx, y + dy, m(dx, dy)));
            }
        }
    }
    mat.setFromTriplets(triplets.begin(), triplets.end());
}

void GraphSLAM::eulerCovariance(KDTreePtr tree, SLAMScanPtr scan, Matrix6d& outMat, Vector6d& outVec) const
{
    size_t n = scan->numPoints();

    KDTree::Neighbor* results = new KDTree::Neighbor[n];

    size_t pairs = KDTree::nearestNeighbors(tree, scan, results, m_options->slamMaxDistance);

    Vector6d mz = Vector6d::Zero();
    Vector3d sum = Vector3d::Zero();
    double xy, yz, xz, ypz, xpz, xpy;
    xy = yz = xz = ypz = xpz = xpy = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        if (results[i] == nullptr)
        {
            continue;
        }

        Vector3d p = scan->point(i).cast<double>();
        Vector3d r = results[i]->cast<double>();

        Vector3d mid = (p + r) / 2.0;
        Vector3d d = r - p;

        double x = mid.x(), y = mid.y(), z = mid.z();

        sum += mid;

        xpy += x * x + y * y;
        xpz += x * x + z * z;
        ypz += y * y + z * z;

        xy += x * y;
        xz += x * z;
        yz += y * z;

        mz.block<3, 1>(0, 0) += d;

        mz(3) += -z * d.y() + y * d.z();
        mz(4) += -y * d.x() + x * d.y();
        mz(5) += z * d.x() - x * d.z();
    }

    Matrix6d mm = Matrix6d::Zero();
    mm(0, 0) = mm(1, 1) = mm(2, 2) = pairs;
    mm(3, 3) = ypz;
    mm(4, 4) = xpy;
    mm(5, 5) = xpz;

    mm(0, 4) = mm(4, 0) = -sum.y();
    mm(0, 5) = mm(5, 0) = sum.z();
    mm(1, 3) = mm(3, 1) = -sum.z();
    mm(1, 4) = mm(4, 1) = sum.x();
    mm(2, 3) = mm(3, 2) = sum.y();
    mm(2, 5) = mm(5, 2) = -sum.x();

    mm(3, 4) = mm(4, 3) = -xz;
    mm(3, 5) = mm(5, 3) = -xy;
    mm(4, 5) = mm(5, 4) = -yz;

    Vector6d d = mm.inverse() * mz;

    double ss = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        if (results[i] == nullptr)
        {
            continue;
        }

        Vector3d p = scan->point(i).cast<double>();
        Vector3d r = results[i]->cast<double>();

        Vector3d mid = (p + r) / 2.0;
        Vector3d delta = r - p;

        ss += pow(delta.x() + (d(0) - mid.y() * d(4) + mid.z() * d(5)), 2.0)
              + pow(delta.y() + (d(1) - mid.z() * d(3) + mid.x() * d(4)), 2.0)
              + pow(delta.z() + (d(2) + mid.y() * d(3) - mid.x() * d(5)), 2.0);
    }

    delete[] results;

    ss = ss / (2.0 * pairs - 3.0);

    ss = 1.0 / ss;

    outMat = mm * ss;
    outVec = mz * ss;
}

void EulerToMatrix4(const Vector3d& pos,
                    const Vector3d& theta,
                    Matrix4d& mat)
{
    double sx = sin(theta[0]);
    double cx = cos(theta[0]);
    double sy = sin(theta[1]);
    double cy = cos(theta[1]);
    double sz = sin(theta[2]);
    double cz = cos(theta[2]);

    mat << cy* cz,
        sx* sy* cz + cx* sz,
        -cx* sy* cz + sx* sz,
        0.0,
        -cy* sz,
        -sx* sy* sz + cx* cz,
        cx* sy* sz + sx* cz,
        0.0,
        sy,
        -sx* cy,
        cx* cy,

        0.0,

        pos[0],
        pos[1],
        pos[2],
        1;

    mat.transposeInPlace();
}

void Matrix4ToEuler(const Matrix4d inputMat,
                    Vector3d& rPosTheta,
                    Vector3d& rPos)
{
    Matrix4d mat = inputMat.transpose();

    double _trX, _trY;

    // Calculate Y-axis angle
    rPosTheta[1] = asin(max(-1.0, min(1.0, mat(2, 0)))); // asin returns nan for any number outside [-1, 1]
    if (mat(0, 0) <= 0.0)
    {
        rPosTheta[1] = M_PI - rPosTheta[1];
    }

    double C = cos(rPosTheta[1]);
    if (fabs( C ) > 0.005)                    // Gimbal lock?
    {
        _trX      =  mat(2, 2) / C;             // No, so get X-axis angle
        _trY      =  -mat(2, 1) / C;
        rPosTheta[0]  = atan2( _trY, _trX );
        _trX      =  mat(0, 0) / C;              // Get Z-axis angle
        _trY      = -mat(1, 0) / C;
        rPosTheta[2]  = atan2( _trY, _trX );
    }
    else                                        // Gimbal lock has occurred
    {
        rPosTheta[0] = 0.0;                       // Set X-axis angle to zero
        _trX      =  mat(1, 1);  //1                // And calculate Z-axis angle
        _trY      =  mat(0, 1);  //2
        rPosTheta[2]  = atan2( _trY, _trX );
    }

    rPos = inputMat.block<3, 1>(0, 3);
}

} /* namespace lvr2 */
