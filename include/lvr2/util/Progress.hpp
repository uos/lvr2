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
 * ProgressBar.h
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */

#ifndef LVR2_PROGRESSBARQT_H_
#define LVR2_PROGRESSBARQT_H_

#include <string>
#include <sstream>
#include <iostream>

using std::stringstream;
using std::cout;
using std::endl;
using std::flush;
using std::string;
using std::wstring;
using std::wcout;

#include <boost/thread/mutex.hpp>

namespace lvr2
{

/**
 * @brief  	A class to manage progress information output for
 * 			process where the number of performed operations
 * 			is known in advance, e.g. number of loop iterations
 *
 * 	After each iteration the ++-operator should be called. The
 * 	progress information in '%' is automatically printed to stdout
 * 	together with the given prefix string.
 */

typedef void(*ProgressCallbackPtr)(int);
typedef void(*ProgressTitleCallbackPtr)(string);

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

    virtual ~ProgressBar();

    /**
     * @brief Increases the counter of performed iterations
     */
    void operator++();

    /**
     * @brief Increases the counter of performed by \ref n
     */
    void operator+=(size_t n);


    /**
     * @brief 	Registers a callback that is called with the new value
     * 			when the percentage of the progress changed.
     *
     * @param
     */
    static void setProgressCallback(ProgressCallbackPtr);

    /**
     * @brief	Registers a callback that is called when a new progress
     * 			instance is created.
     * @param
     */
    static void setProgressTitleCallback(ProgressTitleCallbackPtr);

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

    static ProgressCallbackPtr 			m_progressCallback;
    static ProgressTitleCallbackPtr		m_titleCallback;
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


/**
 * @brief  	A class to manage progress information output for
 * 			process where the number of performed operations
 * 			is known in advance, e.g. number of loop iterations
 *
 * 	After each iteration the ++-operator should be called. The
 * 	progress information in '%' is automatically printed to stdout
 * 	together with the given prefix string.
 */

typedef void(*PacmanProgressCallbackPtr)(int);
typedef void(*PacmanProgressTitleCallbackPtr)(string);

class PacmanProgressBar
{

public:

	/**
	 * @brief Ctor.
	 *
	 * @param max_val	The number of performed iterations
	 * @param prefix	The prefix string for progress output
	 */
	PacmanProgressBar(size_t max_val, string prefix = "", size_t bar_length = 60 );

	virtual ~PacmanProgressBar();

	/**
	 * @brief Increases the counter of performed iterations
	 */
	void operator++();

	/**
	 * @brief 	Registers a callback that is called with the new value
	 * 			when the percentage of the progress changed.
	 *
	 * @param
	 */
	static void setProgressCallback(ProgressCallbackPtr);

	/**
	 * @brief	Registers a callback that is called when a new progress
	 * 			instance is created.
	 * @param
	 */
	static void setProgressTitleCallback(ProgressTitleCallbackPtr);

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
	int				m_percent;

	// bar length
	int 			m_bar_length;

	/// A string stream for output generation
	stringstream	m_stream;

	/// A fill string for correct output alignment
	string 			m_fillstring;

	static PacmanProgressCallbackPtr 			m_progressCallback;
	static PacmanProgressTitleCallbackPtr		m_titleCallback;
};

} // namespace lvr2

#endif /* PROGRESSBAR_H_ */
