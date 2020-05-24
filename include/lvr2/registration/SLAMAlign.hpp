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
 * SLAMAlign.hpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 *  @author Timo Osterkamp (tosterkamp@uni-osnabrueck.de)
 */
#ifndef SLAMALIGN_HPP_
#define SLAMALIGN_HPP_

#include "SLAMScanWrapper.hpp"
#include "SLAMOptions.hpp"
#include "GraphSLAM.hpp"

namespace lvr2
{

/**
 * @brief A class to run SLAM on Scans
 */
class SLAMAlign
{

public:
    /**
     * @brief Creates a new SLAMAlign instance with the given Options and Scans
     *
     * This does not yet register the Scans, it only applies reduction options if specified
     *
     * @param options The Options to use
     * @param scans The Scans to start with
     */
    SLAMAlign(const SLAMOptions& options, const std::vector<SLAMScanPtr>& scans, std::vector<bool> new_scans = std::vector<bool>());

    /**
     * @brief Creates a new SLAMAlign instance with the given Options
     *
     * @param options The Options to use
     */
    SLAMAlign(const SLAMOptions& options = SLAMOptions(), std::vector<bool> new_scans = std::vector<bool>());

    virtual ~SLAMAlign() = default;

    /**
     * @brief Adds a new Scan to the SLAM instance
     *
     * This method will apply any reduction options that are specified
     *
     * @param scan The new Scan
     * @param match true: Immediately call match() with the new Scan added
     */
    void addScan(const SLAMScanPtr& scan, bool match = false);

    /**
     * @brief Adds a new Scan to the SLAM instance
     *
     * This method will apply any reduction options that are specified
     *
     * @param scan The new Scan
     * @param match true: Immediately call match() with the new Scan added
     */
    void addScan(const ScanPtr& scan, bool match = false);

    /**
     * @brief Returns a shared_ptr to a Scan
     *
     * @param index The index of the Scan
     */
    SLAMScanPtr scan(size_t index) const;

    /**
     * @brief Executes SLAM on all current Scans
     *
     * This methods registers any new Scans added since the last call to match()
     * (or the creation of this instance) using Scanmatching and Loopclosing, as specified by
     * the SLAMOptions.
     *
     * Calling this method several times without adding any new Scans has no additional effect
     * after the first call.
     */
    void match();

    /**
     * @brief Indicates that no new Scans will be added
     *
     * This method ensures that all Scans are properly registered, including any Loopclosing
     */
    void finish();

    /**
     * @brief Sets the SLAMOptions struct to the parameter
     *
     * Note that changing options on an active SLAMAlign instance with previously added / matched
     * Scans can cause Undefined Behaviour.
     *
     * @param options The new options
     */
    void setOptions(const SLAMOptions& options);

    /**
     * @brief Returns a reference to the internal SLAMOptions struct
     *
     * This can be used to make changes to specific values within the SLAMOptions without replacing
     * the entire struct.
     *
     * Note that changing options on an active SLAMAlign instance with previously added / matched
     * Scans can cause Undefined Behaviour.
     */
    SLAMOptions& options();

    /**
     * @brief Returns a reference to the internal SLAMOptions struct
     */
    const SLAMOptions& options() const;

protected:

    /// Applies all reductions to the Scan
    void reduceScan(const SLAMScanPtr& scan);

    /// Applies the Transformation to the specified Scan and adds a frame to all other Scans
    void applyTransform(SLAMScanPtr scan, const Matrix4d& transform);

    /// Checks for and executes any loopcloses that occur
    void checkLoopClose(size_t last);

    /// Closes a simple Loop between first and last
    void loopClose(size_t first, size_t last);

    /// Executes GraphSLAM up to and including the specified last Scan
    void graphSLAM(size_t last);

    /// checkLoopClose(size_t last) if the m_icp_graph is in a spezial order
    /**
     * @brief same as checkLoopClose(size_t last) but if the m_icp_graph is in a spezial order
     *
     * Same as checkLoopClose(size_t last) but if the m_icp_graph is in a spezial order. 
     */
    void checkLoopCloseOtherOrder(size_t last);

    /**
     * @brief Create m_icp_graph which defined the order of registrations
     *
     * Create m_icp_graph which defined the order of registrations. The first scan is regarded
     * as registered. Then the scan that is closest to one of the already matched scans is always
     * added. Therefore the scan centers were compared using Euclidean distance.
     */
    void createIcpGraph();

    SLAMOptions              m_options;

    std::vector<SLAMScanPtr> m_scans;

    SLAMScanPtr              m_metascan;

    GraphSLAM                m_graph;
    bool                     m_foundLoop;
    int                      m_loopIndexCount;

    std::vector<bool>        m_new_scans;

    std::vector<std::pair<int, int>> m_icp_graph;
};

} /* namespace lvr2 */

#endif /* SLAMALIGN_HPP_ */
