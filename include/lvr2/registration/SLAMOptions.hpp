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
 * SLAMOptions.hpp
 *
 *  @date May 28, 2019
 *  @author Malte Hillmann
 *  @author Timo Osterkamp (tosterkamp@uni-osnabrueck.de)
 * 
 */
#ifndef SLAMOPTIONS_HPP_
#define SLAMOPTIONS_HPP_

namespace lvr2
{

/**
 * @brief A struct to configure SLAMAlign
 */
struct SLAMOptions
{
    // ==================== General Options ======================================================

    /// Use the unmodified Pose of new Scans. If false, apply the relative refinement of previous Scans
    bool    trustPose = false;

    /// Match scans to the combined Pointcloud of all previous Scans instead of just the last Scan
    bool    metascan = false;

    /// Keep track of all previous Transformations of Scans for Animation purposes like 'show' from slam6D
    bool    createFrames = false;

    /// Show more detailed output. Useful for fine-tuning Parameters or debugging
    bool    verbose = false;

    /// Indicates if a HDF file containing the scans should be used
    bool    useHDF = false;

    // ==================== Reduction Options ====================================================

    /// The Voxel size for Octree based reduction
    double  reduction = -1;

    /// Ignore all Points closer than <value> to the origin of a scan
    double  minDistance = -1;

    /// Ignore all Points farther away than <value> from the origin of a scan
    double  maxDistance = -1;

    // ==================== ICP Options ==========================================================

    /// Number of iterations for ICP.
    /// ICP should ideally converge before this number is met, but this number places an upper Bound on calculation time
    int     icpIterations = 100;

    /// The maximum distance between two points during ICP
    double  icpMaxDistance = 25;

    /// The maximum number of Points in a Leaf of a KDTree
    int     maxLeafSize = 20;

    /// The epsilon difference between ICP-errors for the stop criterion of ICP
    double  epsilon = 0.00001;

    // ==================== SLAM Options =========================================================

    /// Use simple Loopclosing
    bool    doLoopClosing = false;

    /// Use complex Loopclosing with GraphSLAM
    bool    doGraphSLAM = false;

    /// The maximum distance between two poses to consider a closed loop or an Edge in the GraphSLAM Graph.
    /// Mutually exclusive to closeLoopPairs
    double  closeLoopDistance = 500;

    /// The minimum pair overlap between two poses to consider a closed loop or an Edge in the GraphSLAM Graph.
    /// Mutually exclusive to closeLoopDistance
    int     closeLoopPairs = -1;

    /// The minimum number of Scans to be considered a Loop to prevent Loopclosing from triggering on adjacent Scans.
    /// Also used in GraphSLAM when considering other Scans for Edges.
    /// For Loopclosing, this value needs to be at least 6, for GraphSLAM at least 1
    int     loopSize = 20;

    /// Number of ICP iterations during Loopclosing and number of GraphSLAM iterations
    int     slamIterations = 50;

    /// The maximum distance between two points during SLAM
    double  slamMaxDistance = 25;

    /// The epsilon difference of SLAM corrections for the stop criterion of SLAM
    double  slamEpsilon = 0.5;

    /// max difference of position (euclidean distance) new and old
    double diffPosition = 50;

    /// max difference of angle (sum of 3 angles) new and old
    double diffAngle = 50;

    /// use scan order as icp order (if false: start with lowest distance)
    bool useScanOrder = true;

    /// rotate this angle around y axis
    double rotate_angle = 0;
};

} /* namespace lvr2 */

#endif /* SLAMOPTIONS_HPP_ */
