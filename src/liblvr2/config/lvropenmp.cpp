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
 * lvropenmp.cpp
 *
 *  Created on: 23.01.2015
 *      Author: Thomas Wiemann
 */

#include "lvr2/config/lvropenmp.hpp"

#ifdef LVR2_USE_OPEN_MP
#include <omp.h>
#endif

#include <iostream>
using namespace std;

namespace lvr2
{

bool OpenMPConfig::haveOpenMP()
{
#ifdef LVR2_USE_OPEN_MP
	return true;
#else
	return false;
#endif
}

void OpenMPConfig::setNumThreads(int n)
{
#ifdef LVR2_USE_OPEN_MP
	omp_set_num_threads(n);
#endif
}

void OpenMPConfig::setMaxNumThreads()
{
#ifdef LVR2_USE_OPEN_MP
	omp_set_num_threads(omp_get_num_procs());
#endif
}

int OpenMPConfig::getNumThreads()
{
#ifdef LVR2_USE_OPEN_MP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

} // namespace lvr2


