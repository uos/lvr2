/*
 * Timestamp.cpp
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */

#include "Timestamp.hpp"

#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cstddef>
#include <cstdio>

#include <iostream>
using namespace std;

namespace lssr {

Timestamp::Timestamp()
{
    resetTimer();
}


unsigned long Timestamp::getCurrentTimeInMs() const
{
    static struct timeval tv;
    gettimeofday( &tv, NULL );
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

unsigned long Timestamp::getElapsedTimeInMs() const
{
    return getCurrentTimeInMs() - m_startTime;
}

double Timestamp::getCurrentTimeinS() const
{
    return (double) getCurrentTimeInMs() / 1000.0;
}

double Timestamp::getElapsedTimeInS() const
{
    return (double) getElapsedTimeInMs() / 1000.0;
}

void Timestamp::resetTimer()
{
    m_startTime = getCurrentTimeInMs();
}

string Timestamp::getElapsedTime() const
{
    unsigned long time = getElapsedTimeInMs();
    unsigned long hours =   time / 3600000;
    unsigned long mins  = ( time % 3600000 ) / 60000;
    unsigned long secs  = ( time %   60000 ) /  1000;
    unsigned long msecs =   time %    1000;

    char times[32];
    sprintf( times, "[%02lu:%02lu:%02lu %04lu] ", hours, mins, secs, msecs);

    return string( times );
}

} // namespace lssr



