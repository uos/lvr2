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
 * MetaScan.hpp
 *
 *  @date May 28, 2019
 *  @author Malte Hillmann
 */
#ifndef METASCAN_HPP_
#define METASCAN_HPP_

#include <Eigen/Dense>
#include "Scan.hpp"

namespace lvr2
{

class MetaScan : public Scan
{
public:
    MetaScan()
	{ }

    const Vector3d& getPoint(size_t index) const override
	{
		for (const ScanPtr& scan : m_scans)
		{
			if (index < scan->count())
			{
				return scan->getPoint(index);
			}
			else
			{
				index -= scan->count();
			}
		}
        throw std::out_of_range("getPoint on MetaScan out of Range");
	}

    Vector3d getPointTransformed(size_t index) const override
	{
		for (const ScanPtr& scan : m_scans)
		{
			if (index < scan->count())
			{
				return scan->getPointTransformed(index);
			}
			else
			{
				index -= scan->count();
			}
		}
        throw std::out_of_range("getPointTransformed on MetaScan out of Range");
	}

	void addScan(ScanPtr scan)
	{
		m_count += scan->count();
		m_scans.push_back(scan);
		m_deltaPose = scan->getDeltaPose();
	}

private:
    std::vector<ScanPtr> m_scans;
};

using ScanPtr = std::shared_ptr<Scan>;

} /* namespace lvr2 */

#endif /* METASCAN_HPP_ */
