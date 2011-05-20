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

namespace lssr{

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
	ProgressBar(int max_val, string prefix = "");

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
	size_t			m_currentVal;

	/// The step size (in percent) after which an output is generated
	size_t			m_stepSize;

	/// A mutex object for counter increment (for parallel executions)
	boost::mutex 	m_mutex;

	/// The current progress in percent
	int				m_percent;

	/// A string stream for output generation
	stringstream	m_stream;

	/// A fill string for correct output alignment
	string 			m_fillstring;
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

} // namespace lssr

#endif /* PROGRESSBAR_H_ */
