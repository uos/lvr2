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


#include <lvr/io/Progress.hpp>

#include <sstream>
#include <iostream>

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;

namespace lvr2
{

ProgressCallbackPtr ProgressBar::m_progressCallback = 0;
ProgressTitleCallbackPtr ProgressBar::m_titleCallback = 0;

ProgressBar::ProgressBar(size_t max_val, string prefix)
{
	m_prefix = prefix;
	m_maxVal = max_val;
    m_currentVal = 0;
	m_percent = 0;

	if(m_titleCallback)
	{
		// Remove time brackets
		unsigned index;
		index = prefix.find_last_of("]");
		m_titleCallback(prefix.substr(index+1));
	}
}

ProgressBar::~ProgressBar()
{

}

void ProgressBar::setProgressCallback(ProgressCallbackPtr ptr)
{
	m_progressCallback = ptr;
}

void ProgressBar::setProgressTitleCallback(ProgressTitleCallbackPtr ptr)
{
	m_titleCallback = ptr;
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

        if(m_progressCallback)
        {
        	m_progressCallback(m_percent);
        }
    }

}

void ProgressBar::operator+=(size_t n)
{
    boost::mutex::scoped_lock lock(m_mutex);

    m_currentVal+= n;
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

        if(m_progressCallback)
        {
            m_progressCallback(m_percent);
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

PacmanProgressCallbackPtr PacmanProgressBar::m_progressCallback = 0;
PacmanProgressTitleCallbackPtr PacmanProgressBar::m_titleCallback = 0;

PacmanProgressBar::PacmanProgressBar(size_t max_val, string prefix, size_t bar_length)
:
	m_prefix(prefix)
	,m_bar_length(bar_length)
{
	m_maxVal = max_val;
    m_currentVal = 0;
	m_percent = 0;

	if(m_titleCallback)
	{
		// Remove time brackets
		unsigned index;
		index = prefix.find_last_of("]");
		m_titleCallback(prefix.substr(index+1));
	}
}

PacmanProgressBar::~PacmanProgressBar()
{

}

void PacmanProgressBar::setProgressCallback(ProgressCallbackPtr ptr)
{
	m_progressCallback = ptr;
}

void PacmanProgressBar::setProgressTitleCallback(ProgressTitleCallbackPtr ptr)
{
	m_titleCallback = ptr;
}

void PacmanProgressBar::operator++()
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

        if(m_progressCallback)
        {
        	m_progressCallback(m_percent);
        }
    }

}

void PacmanProgressBar::print_bar()
{
	int char_idx = static_cast<float>(m_percent)/100.0 * (m_bar_length);

	cout <<  "\r" << m_prefix << " " << m_percent << "%" << " | ";

	for(size_t i=0; i< char_idx; i++)
	{
		cout << " ";
	}

	if(char_idx % 2 == 0)
	{
		cout << "ᗧ";
	}else{
		cout << "O";
	}

	for(size_t i=char_idx; i < m_bar_length; i++)
	{
		if(i%2 == 0)
		{
			cout << " ";
		} else {
			cout << "•";
		}
	}

    cout << " | " << flush;
}

} // namespace lvr

