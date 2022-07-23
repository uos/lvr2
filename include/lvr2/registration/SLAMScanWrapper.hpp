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
 * SLAMScanWrapper.hpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#ifndef SLAMSCANWRAPPER_HPP_
#define SLAMSCANWRAPPER_HPP_

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/algorithm/KDTree.hpp"

#include <Eigen/Dense>
#include <vector>

namespace lvr2
{

/**
 * @brief Annotates the use of a Scan when creating an slam6D .frames file
 */
enum class FrameUse
{
    /// The Scan has not been registered yet
    INVALID = 0,
    /// The Scan changed since the last Frame
    UPDATED = 1,
    /// The Scan did not change since the last Frame
    UNUSED = 2,
    /// The Scan was part of a GraphSLAM Iteration
    GRAPHSLAM = 3,
    /// The Scan was part of a Loopclose Iteration
    LOOPCLOSE = 4,
};

/**
 * @brief A Wrapper around Scan to allow for SLAM usage
 */
class SLAMScanWrapper
{
public:
    /**
     * @brief Construct a new SLAMScanWrapper object as a Wrapper around the Scan
     *
     * @param scan The Scan to wrap around
     */
    SLAMScanWrapper(ScanPtr scan);

    virtual ~SLAMScanWrapper() = default;

    /**
     * @brief Access to the Scan that this instance is wrapped around
     *
     * @return ScanPtr The Scan
     */
    ScanPtr innerScan();


    /**
     * @brief Applies a relative Transformation to the Scan
     *
     * @param transform The Transformation
     * @param writeFrame weather or not to add a new animation Frame
     * @param use The FrameUse if writeFrame is set to true
     */
    virtual void transform(const Transformd& transform, bool writeFrame = true, FrameUse use = FrameUse::UPDATED);

    /**
     * @brief Adds a new animation Frame with the current Pose
     *
     * @param use The use of the Frame for coloring purposes
     */
    void addFrame(FrameUse use = FrameUse::UNUSED);


    /**
     * @brief Reduces the Scan using Octree Reduction
     *
     * Does not change the amount of allocated Memory unless trim() is called
     *
     * @param voxelSize
     * @param maxLeafSize
     */
    void reduce(double voxelSize, int maxLeafSize);

    /**
     * @brief Reduces the Scan by removing all Points closer than minDistance to the origin
     *
     * Does not change the amount of allocated Memory unless trim() is called
     *
     * @param minDistance The minimum Distance for a Point to have
     */
    void setMinDistance(double minDistance);

    /**
     * @brief Reduces the Scan by removing all Points farther away than maxDistance to the origin
     *
     * Does not change the amount of allocated Memory unless trim() is called
     *
     * @param maxDistance The maximum Distance for a Point to have
     */
    void setMaxDistance(double maxDistance);

    /**
     * @brief Reduces Memory usage by getting rid of Points removed with reduction Methods
     */
    void trim();


    /**
     * @brief Returns the Point at the specified index in global Coordinates
     *
     * @param index the Index
     * @return Vector3d the Point in global Coordinates
     */
    virtual Vector3d point(size_t index) const;

    /**
     * @brief Returns the Point at the specified index in local Coordinates
     *
     * @param index the Index
     * @return Vector3d the Point in local Coordinates
     */
    const Vector3f& rawPoint(size_t index) const;

    /**
     * @brief Returns the number of Points in the Scan
     *
     * @return size_t the number of Points
     */
    size_t numPoints() const;


    /**
     * @brief Returns the current Pose of the Scan
     *
     * @return const Transformd& the Pose
     */
    const Transformd& pose() const;

    /**
     * @brief Returns the difference between pose() and initialPose()
     *
     * @return const Transformd& the delta Pose
     */
    const Transformd& deltaPose() const;

    /**
     * @brief Returns the initial Pose of the Scan
     *
     * @return const Transformd& the initial Pose
     */
    const Transformd& initialPose() const;

    /**
     * @brief Get the Position of the Scan. Can also be obtained from pose()
     *
     * @return Vector3d the Position
     */
    Vector3d getPosition() const;

    /**
     * @brief Returns the number of Frames generated
     *
     * @return size_t the number of Frames
     */
    size_t frameCount() const;

    /**
     * @brief Returns a Frame consisting of a Pose and a FrameUse
     *
     * @param index the index of the Frame
     * @return const std::pair<Transformd, FrameUse>& the Pose and FrameUse
     */
    const std::pair<Transformd, FrameUse>& frame(size_t index) const;

    /**
     * @brief Writes the Frames to the specified location
     *
     * @param path The path of the file to write to
     */
    void writeFrames(std::string path) const;

    KDTreePtr<Vector3f> createKDTree(size_t maxLeafSize = 20) const;

    /**
     * @brief Finds the nearest neighbors of all points in a Scan using a pre-generated KDTree
     *
     * @param tree          The KDTree to search in
     * @param scan          The Scan to search for
     * @param neighbors     An array to store the results in. neighbors[i] is set to a Pointer to the
     *                      neighbor of points[i] or nullptr if none was found
     * @param maxDistance   The maximum Distance for a Neighbor
     * @param centroid_m    Will be set to the average of all Points in 'neighbors'
     * @param centroid_d    Will be set to the average of all Points in 'points' that have neighbors
     *
     * @return size_t The number of neighbors that were found
     */
    static size_t nearestNeighbors(KDTreePtr<Vector3f> tree, std::shared_ptr<SLAMScanWrapper> scan,
                                   std::vector<Vector3f*>& neighbors, double maxDistance,
                                   Vector3d& centroid_m, Vector3d& centroid_d);

    /**
     * @brief Finds the nearest neighbors of all points in a Scan using a pre-generated KDTree
     *
     * @param tree          The KDTree to search in
     * @param scan          The Scan to search for
     * @param neighbors     An array to store the results in. neighbors[i] is set to a Pointer to the
     *                      neighbor of points[i] or nullptr if none was found
     * @param maxDistance   The maximum Distance for a Neighbor
     *
     * @return size_t The number of neighbors that were found
     */
    static size_t nearestNeighbors(KDTreePtr<Vector3f> tree, std::shared_ptr<SLAMScanWrapper> scan,
                                   std::vector<Vector3f*>& neighbors, double maxDistance);

protected:
    ScanPtr               m_scan;

    std::vector<Vector3f> m_points;
    size_t                m_numPoints;

    Transformd            m_deltaPose;

    std::vector<std::pair<Transformd, FrameUse>> m_frames;
};

using SLAMScanPtr = std::shared_ptr<SLAMScanWrapper>;

} /* namespace lvr2 */

#endif /* SLAMSCANWRAPPER_HPP_ */
