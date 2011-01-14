/*
 * ProgressBar.cpp
 *
 *  Created on: 14.01.2011
 *      Author: twiemann
 */

#include "ProgressBar.h"

#include <sstream>
#include <iostream>

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;

ProgressBar::ProgressBar(int max_val, string prefix)
{
	m_prefix = prefix;
	m_maxVal = max_val;
	m_currentVal = 0;
	m_stepSize = max_val / 20;
	m_percent = 0;

	size_t size = 0;
	if(m_prefix.size() < 50)
	{
		size = 50 - m_prefix.size();
	}

	m_fillstring = string(size, ' ');
}

ProgressBar::~ProgressBar()
{

}

void ProgressBar::operator++()
{
	boost::mutex::scoped_lock lock(m_mutex);
	m_currentVal++;
	if(m_currentVal >= m_stepSize)
	{
		m_currentVal = 0;
		print_bar();
	}
}

void ProgressBar::print_bar()
{
	m_percent += 5;

	m_stream << "|";
	cout << "\r";
	cout << m_prefix << m_fillstring << m_stream.str() << " " << m_percent << "%" << flush;
}
