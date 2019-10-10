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

/*
 * STLIO.hpp
 *
 *  Created on: Dec 9, 2016
 *      Author: robot
 */

#ifndef INCLUDE_LVR2_IO_STLIO_HPP_
#define INCLUDE_LVR2_IO_STLIO_HPP_

#include "lvr2/io/BaseIO.hpp"

namespace lvr2
{

/****
 * @brief 	Reader / Writer for STL file. Currently only binary STL files
 * 			are supported.
 */
class STLIO : public BaseIO
{
public:
	STLIO();
	virtual ~STLIO();

	virtual void save( string filename );
	virtual void save( ModelPtr model, string filename );
    /**
     * @brief Parse the given file and load supported elements.
     *
     * @param 	filename  The file to read.
     * @return	A new model. If the file could not be parsed, an empty model
     * 			is returned.
     */
    virtual ModelPtr read(string filename );

};

} /* namespace lvr2 */

#endif /* INCLUDE_LVR2_IO_STLIO_HPP_ */
