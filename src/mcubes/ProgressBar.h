/*
 * ProgressBar.h
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */

#ifndef PROGRESSBAR_H_
#define PROGRESSBAR_H_

#include <string>
#include <sstream>
#include <iostream>

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;
using std::string;

#include <boost/thread/mutex.hpp>

class ProgressBar
{
public:
	ProgressBar(int max_val, string prefix = "");
	virtual ~ProgressBar();

	void operator++();
private:

	void print_bar();

	string 			m_prefix;
	size_t			m_maxVal;
	size_t			m_currentVal;
	size_t			m_stepSize;
	boost::mutex 	m_mutex;
	int				m_percent;
	stringstream	m_stream;
	string 			m_fillstring;
};

#endif /* PROGRESSBAR_H_ */
