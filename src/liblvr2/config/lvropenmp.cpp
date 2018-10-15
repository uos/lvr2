/* Copyright (C) 2011 Uni Osnabr√ºck
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * lvropenmp.cpp
 *
 *  Created on: 23.01.2015
 *      Author: Thomas Wiemann
 */

#include <lvr2/config/lvropenmp.hpp>

#ifdef LVR_USE_OPEN_MP
#include <omp.h>
#endif

#include <iostream>
using namespace std;

namespace lvr2
{

bool OpenMPConfig::haveOpenMP()
{
#ifdef LVR_USE_OPEN_MP
	return true;
#else
	return false;
#endif
}

void OpenMPConfig::setNumThreads(int n)
{
#ifdef LVR_USE_OPEN_MP
	omp_set_num_threads(n);
#endif
}

void OpenMPConfig::setMaxNumThreads()
{
#ifdef LVR_USE_OPEN_MP
	omp_set_num_threads(omp_get_num_procs());
#endif
}

int OpenMPConfig::getNumThreads()
{
#ifdef LVR_USE_OPEN_MP
	return omp_get_num_procs();
#else
	return 1;
#endif
}

} // namespace lvr2


