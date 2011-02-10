/*
 * Timestamp.h
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */

#ifndef TIMESTAMP_H_
#define TIMESTAMP_H_

#include <iostream>
#include <string>

using std::ostream;
using std::string;

namespace lssr {

/**
 * @brief	A helper class for automated time stamping. Timing is
 * 			started as soon as an object of this class is created.
 * 			To time some parts of a program, just create a new object
 * 			and use the provided output operator to display the elapsed
 * 			time.
 */
class Timestamp
{
public:

	/**
	 * @brief	Constructor.
	 */
	Timestamp();

	/**
	 * @brief	Returns the current system time in milliseconds
	 */
	unsigned long  getCurrentTimeInMs() const;

	/**
	 * @brief	Returns the milliseconds since object creation
	 */
	unsigned long  getElapsedTimeInMs() const;

	/**
	 * @brief	Returns the current system time in seconds
	 */
	double	 	   getCurrentTimeinS() const;

	/**
	 * @brief	Returns the time since instantiation in seconds
	 */
	double	 	   getElapsedTimeInS() const;

	/**
	 * @brief	Resets the internal timer
	 */
	void     	   resetTimer();

	/**
	 * @brief	Returns a string representation of the current
	 * 			timer value
	 */
	string		   getElapsedTime() const;

private:

	/// The system at object instantiation
	unsigned long 	m_startTime;
};

/// A global time stamp object for program runtime measurement
static Timestamp timestamp;

/// The output operator for Timestamp objects
inline ostream& operator<<(ostream& os, const Timestamp &ts)
{
	os << ts.getElapsedTime();
	return os;
}

} // namespace lssr

#endif /* TIMESTAMP_H_ */
