/* Copyright (C) 2011 Uni Osnabrück
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
 * ProgressBar.cpp
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */


#include "io/Progress.hpp"

#include <sstream>
#include <iostream>

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;

namespace lvr
{

ProgressCallbackPtr ProgressBar::m_callBack = 0;

ProgressBar::ProgressBar(size_t max_val, string prefix)
{
	m_prefix = prefix;
	m_maxVal = max_val;
    m_currentVal = 0;
	m_percent = 0;

}

ProgressBar::~ProgressBar()
{

}

void ProgressBar::setProgressCallback(ProgressCallbackPtr ptr)
{
	m_callBack = ptr;
}


void ProgressBar::operator++()
{
    boost::mutex::scoped_lock lock(m_mutex);

    m_currentVal++;
    short difference = (short)((float)m_currentVal/m_maxVal * 100 - m_percent);
    if (difference < 1)
    {
        return;
    }

    while (difference >= 1)
    {
        m_percent++;
        difference--;
        print_bar();

        if(m_callBack)
        {
        	m_callBack(m_percent);
        }
    }

}

void ProgressBar::print_bar()
{
	cout <<  "\r" << m_prefix << " " << m_percent << "%" << flush;
}

ProgressCounter::ProgressCounter(int stepVal, string prefix)
{
	m_prefix = prefix;
	m_stepVal = stepVal;
	m_currentVal = 0;
}

void ProgressCounter::operator++()
{
	boost::mutex::scoped_lock lock(m_mutex);
	m_currentVal++;
	if(m_currentVal % m_stepVal == 0)
	{
		print_progress();
	}
}

void ProgressCounter::print_progress()
{
	cout << "\r" << m_prefix << " " << m_currentVal << flush;
}



} // namespace lvr

