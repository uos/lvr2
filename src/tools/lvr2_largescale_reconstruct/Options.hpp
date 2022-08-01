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
 * Options.cpp
 *
 * @date Nov 21, 2010
 * @author Isaak Mitschke
 * @author Malte Hillmann
 */

#pragma once

#include "lvr2/config/BaseOption.hpp"
#include "lvr2/reconstruction/LargeScaleReconstruction.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace fs = boost::filesystem;

namespace LargeScaleOptions
{

/**
 * @brief A class to parse the program options for the reconstruction
 *           executable.
 */
class Options : public lvr2::BaseOption
{
public:
    /**
     * @brief     Ctor. Parses the command parameters given to the main
     *               function of the program
     */
    Options(int argc, char** argv);
    virtual ~Options() = default;

    bool m_printed = false;
    bool printUsage();

    /// All options from LargeScaleReconstruction
    lvr2::LSROptions m_options;

    /// print help message
    bool m_help;

    /// the input file
    fs::path m_inputFile;

    int m_numThreads = -1;
};

} // namespace LargeScaleOptions
