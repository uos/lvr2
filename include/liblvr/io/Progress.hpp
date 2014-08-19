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

#ifdef __WITH_QT4__
#include <QProgressBar>
#endif

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;
using std::string;

#include <boost/thread/mutex.hpp>

namespace lvr{

/**
 * @brief  	A class to manage progress information output for
 * 			process where the number of performed operations
 * 			is known in advance, e.g. number of loop iterations
 *
 * 	After each iteration the ++-operator should be called. The
 * 	progress information in '%' is automatically printed to stdout
 * 	together with the given prefix string.
 */
class ProgressBar
{
public:

	/**
	 * @brief Ctor.
	 *
	 * @param max_val	The number of performed iterations
	 * @param prefix	The prefix string for progress output
	 */
	ProgressBar(size_t max_val, string prefix = "");

	/**
	 * @brief Increases the counter of performed iterations
	 */
	void operator++();

protected:

	/// Prints the output
	void print_bar();

	/// The prefix string
	string 			m_prefix;

	/// The number of iterations
	size_t			m_maxVal;

        /// The current counter
        size_t	 		m_currentVal;

	/// A mutex object for counter increment (for parallel executions)
	boost::mutex 	m_mutex;

	/// The current progress in percent
	int			m_percent;

	/// A string stream for output generation
	stringstream	m_stream;

	/// A fill string for correct output alignment
	string 			m_fillstring;

#ifdef __WITH_QT4__
	QProgressBar*       m_progressBar;

public:
	void setProgressBar(QProgressBar* progressBar)
	{
	    m_progressBar = progressBar;
	    if(m_progressBar)
	    {
	        m_progressBar->setMinimum(0);
	        m_progressBar->setMaximum(100);
	    }
	}
#endif
};


/**
 * @brief	A progress counter class
 *
 * This class can be used of the number of performed operations is not
 * known in advance (e.g. ASCII file reading). After \ref{m_stepVal} operations
 * the current counter is printed.
 */
class ProgressCounter
{
public:

	/***
	 * @brief CTor.
	 *
	 * @param	stepVal	After m_stepVal operations a new output is generated
	 * @param	prefix	The prefix string for progress output
	 */
	ProgressCounter(int stepVal, string prefix = "");

	/***
	 * @brief	Increase the progress counter
	 */
	void operator++();

protected:

	/// Prints the current state
	void print_progress();

	/// The prefix string
	string 			m_prefix;

	/// The step value for output generation
	size_t			m_stepVal;

	/// The current counter value
	size_t			m_currentVal;

	/// A mutex object for counter increment (for parallel executions)
	boost::mutex 	m_mutex;

	/// A string stream for output generation
	stringstream	m_stream;

	/// A fill string for correct output alignment
	string 			m_fillstring;
};

} // namespace lvr

#endif /* PROGRESSBAR_H_ */
