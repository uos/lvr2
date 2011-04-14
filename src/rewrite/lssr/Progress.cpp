/*
 * ProgressBar.cpp
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */

#include "Progress.hpp"

#include <sstream>
#include <iostream>

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;

namespace lssr
{

ProgressBar::ProgressBar(int max_val, string prefix)
{
	m_prefix = prefix;
	m_maxVal = max_val;
	m_currentVal = 0;
	m_stepSize = max_val / 100;
	m_percent = 0;
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
	m_percent += 1;
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

} // namespace lssr

