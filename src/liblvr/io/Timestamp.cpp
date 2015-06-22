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
 * Timestamp.cpp
 *
 *  Created on: 14.01.2011
 *      Author: Thomas Wiemann
 */

#include "io/Timestamp.hpp"

#if defined(_MSC_VER)
#include <time.h>
#include <Windows.h>

const __int64 DELTA_EPOCH_IN_MICROSECS = 11644473600000000;

/* IN UNIX the use of the timezone struct is obsolete;
I don't know why you use it. See http://linux.about.com/od/commands/l/blcmdl2_gettime.htm
But if you want to use this structure to know about GMT(UTC) diffrence from your local time
it will be next: tz_minuteswest is the real diffrence in minutes from GMT(UTC) and a tz_dsttime is a flag
indicates whether daylight is now in use
*/
struct timezone
{
	__int32  tz_minuteswest; /* minutes W of Greenwich */
	bool  tz_dsttime;     /* type of dst correction */
};


int gettimeofday(struct timeval *tv/*in*/, struct timezone *tz/*in*/)
{
	FILETIME ft;
	__int64 tmpres = 0;
	TIME_ZONE_INFORMATION tz_winapi;
	int rez = 0;

	ZeroMemory(&ft, sizeof(ft));
	ZeroMemory(&tz_winapi, sizeof(tz_winapi));

	GetSystemTimeAsFileTime(&ft);

	tmpres = ft.dwHighDateTime;
	tmpres <<= 32;
	tmpres |= ft.dwLowDateTime;

	/*converting file time to unix epoch*/
	tmpres /= 10;  /*convert into microseconds*/
	tmpres -= DELTA_EPOCH_IN_MICROSECS;
	tv->tv_sec = (__int32)(tmpres*0.000001);
	tv->tv_usec = (tmpres % 1000000);


	//_tzset(),don't work properly, so we use GetTimeZoneInformation
	/*
	rez = GetTimeZoneInformation(&tz_winapi);
	tz->tz_dsttime = (rez == 2) ? true : false;
	tz->tz_minuteswest = tz_winapi.Bias + ((rez == 2) ? tz_winapi.DaylightBias : 0); */

	return 0;
}
#else
#include <sys/time.h>
#include <unistd.h>
#endif



#include <cstddef>
#include <cstdlib>
#include <cstdio>

#include <iostream>
using namespace std;

namespace lvr {

Timestamp::Timestamp(): m_nullStream(&m_nullBuffer)
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

    char times[512];

	sprintf( times, "[%02lu:%02lu:%02lu %03lu] ", hours, mins, secs, msecs);
    string result( times );
   
    return result;
}

} // namespace lvr



