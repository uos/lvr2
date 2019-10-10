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
 * ProgressBar.cpp
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */


#include "lvr2/io/Progress.hpp"

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

