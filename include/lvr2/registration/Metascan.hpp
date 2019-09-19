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
 * Metascan.hpp
 *
 *  @date Aug 1, 2019
 *  @author Malte Hillmann
 */
#ifndef METASCAN_HPP_
#define METASCAN_HPP_

#include "SLAMScanWrapper.hpp"

namespace lvr2
{

/**
 * @brief Represents several Scans as part of a single Scan
 * 
 * Note that most methods of Scan don't make sense on a Metascan, like reductions or Pose getters.
 */
class Metascan : public SLAMScanWrapper
{
public:
    Metascan();

    virtual ~Metascan() = default;

    virtual void transform(const Transformd& transform, bool writeFrame = true, FrameUse use = FrameUse::UPDATED) override;
    virtual Vector3d point(size_t index) const override;

    void addScan(SLAMScanPtr scan);

protected:
    std::vector<SLAMScanPtr> m_scans;
};

} /* namespace lvr2 */

#endif /* METASCAN_HPP_ */
