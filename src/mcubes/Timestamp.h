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

class Timestamp
{
public:
	Timestamp();

	unsigned long  getCurrentTimeInMs() const;
	unsigned long  getElapsedTimeInMs() const;

	double	 	   getCurrentTimeinS() const;
	double	 	   getElapsedTimeInS() const;

	void     	   resetTimer();

	string		   getElapsedTime() const;

private:
	unsigned long 	m_startTime;
};

static Timestamp timestamp;

inline ostream& operator<<(ostream& os, const Timestamp &ts)
{
	os << ts.getElapsedTime();
	return os;
}

#endif /* TIMESTAMP_H_ */
