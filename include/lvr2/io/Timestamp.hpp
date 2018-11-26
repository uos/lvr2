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

namespace lvr2 {

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
	
	void 		   setQuiet(bool quiet) { m_quiet = quiet;};

	/**
	 * @brief	Returns a string representation of the current
	 * 			timer value
	 */
	string		   getElapsedTime() const;
	
	bool		   isQuiet() {return m_quiet;};
	
	ostream&       getNullStream() { return m_nullStream; };
	

private:
	
	class NullBuffer : public std::streambuf
	{
		public:
			int overflow(int c) { return c; }
	};

	/// The system at object instantiation
	unsigned long 	m_startTime;
	bool 			m_quiet;
	NullBuffer 		m_nullBuffer;
	std::ostream    m_nullStream;
};

/// A global time stamp object for program runtime measurement
static Timestamp timestamp;

/// The output operator for Timestamp objects
inline ostream& operator<<(ostream& os, Timestamp &ts)
{
	if(ts.isQuiet())
	{
		return ts.getNullStream();
	}
	else
	{
		os << ts.getElapsedTime();
		return os;
	}
}

} // namespace lvr

#endif /* TIMESTAMP_H_ */
