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
 *
 * @file      WaveformIO.hpp
 * @brief     Read and write pointclouds and Wavform data from .lwf files.
 * @details     Read and write pointclouds and Wavform data from .lwf files.
 * 
 * @author    Thomas Wiemann (twiemann), twiemann@uos.de, Universität Osnabrück
 * @author    Michel Loepmeier (mloepmei), mloepmei@uos.de, Universität Osnabrück
 * @version   111001
 * @date      Created:       2020-06-23
 * @date      Last modified: 2011-10-01 19:49:24
 *
 **/

#ifndef WAVEFORMIO_H_
#define WAVEFORMIO_H_

#include "lvr2/io/BaseIO.hpp"

namespace lvr2
{

/**
 * @brief A import / export class for point cloud and waveform data.
 * LWF fies supported.
 */
class WaveformIO : public BaseIO
{
    public:

        /**
         * \brief Default constructor.
         **/
        WaveformIO() {
            setlocale (LC_ALL, "C");
            m_model.reset();
	}
	~WaveformIO() {};
        virtual ModelPtr read( std::string filename);

        virtual void save( string filename);

};


} // namespace lvr2

#endif /* WAVEFORMIO_H_ */
