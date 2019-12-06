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

#include <iostream>
#include <lvr2/types/Scan.hpp>
#include <lvr2/io/Hdf5IO.hpp>
#include <lvr2/io/hdf5/ChannelIO.hpp>
#include <lvr2/io/hdf5/ArrayIO.hpp>
#include <lvr2/io/hdf5/VariantChannelIO.hpp>
#include <lvr2/io/hdf5/MeshIO.hpp>


#include "LargeScaleReconstruction.hpp"

namespace lvr2
{
    using LSRWriter = lvr2::Hdf5IO<lvr2::hdf5features::ArrayIO,
            lvr2::hdf5features::ChannelIO,
            lvr2::hdf5features::VariantChannelIO,
            lvr2::hdf5features::MeshIO>;

    using Vec = lvr2::BaseVector<float>;

    //TODO: write set Methods for certain variables
    LargeScaleReconstruction::LargeScaleReconstruction(string h5File)
    : m_filePath(h5File), m_voxelSize(10), m_bgVoxelSize(10), m_scale(1), m_chunkSize(20), m_partMethod(1),
    m_Ki(10), m_Kd(5), m_Kn(10), m_useRansac(false), m_extrude(false), m_removeDanglingArtifacts(0), m_cleanContours(0),
    m_fillHoles(0), m_optimizePlanes(false), m_getNormalThreshold(0.85), m_planeIterations(3), m_MinPlaneSize(7), m_SmallRegionThreshold(0),
    m_retesselate(false), m_LineFusionThreshold(0.01)
    {
        std::cout << "Reconstruction Instance generated..." << std::endl;
    }

    int LargeScaleReconstruction::mpiChunkAndReconstruct(std::vector<ScanPtr> scans)
    {
        //do more or less the same stuff as the executable
        return 1;
    }

}