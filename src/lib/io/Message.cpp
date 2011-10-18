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


 /**
 *
 * @file      Message.cpp
 * @brief     Class for managing different typed of console output.
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110928
 * @date      09/28/2011 07:30:34 PM
 *
 **/

#include "Message.hpp"
#include "Timestamp.hpp"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace lssr
{

Message::Message( bool color )
    : m_color( color ) {};

ostream& Message::print( const MsgType t, const char * fmt, ... )
{

    char s[1024];
    va_list args;
    va_start( args, fmt );
    vsprintf( s, fmt, args );
    va_end( args );
    return print( t, std::string( s ) );

}


ostream& Message::print( const MsgType t, const std::string s )
{

    return print( t ) << s << ( s[ s.length() - 1 ] == '\n' ? endMsg( false ) : "" );

}


ostream& Message::print( const char* fmt, ... )
{

    char * s;
    va_list args;
    va_start( args, fmt );
    vasprintf( &s, fmt, args );
    va_end( args );
    std::string result( s );
    free( s );
    return print( MSG_TYPE_MESSGAE, result );

}


ostream& Message::print( const std::string fmt )
{

    return print( MSG_TYPE_MESSGAE, fmt );

}


ostream& Message::print( const MsgType t )
{

    ostream * os;
    std::string msg_type( "" );
    switch ( t )
    {
        case MSG_TYPE_WARNING:
            os = &std::cerr; 
            /* msg_type = "[0;33mWarning: "; orange */
            msg_type = std::string( m_color ? "[1;31m" : "" ) + "Warning: "; /* light red */
            break;
        case MSG_TYPE_ERROR:
            os = &std::cerr;
            msg_type = std::string( m_color ? "[0;31m" : "" ) + "Error: "; /* red */
            break;
        case MSG_TYPE_HINT:
            os = &std::cout;
            msg_type = std::string( m_color ? "[1;33m" : "" ) + "Hint: "; /* yellow */
            break;
        case MSG_TYPE_MESSGAE:
            msg_type = std::string( m_color ? "[0;32m" : "" ); /* green */
        default:
            os = &std::cout;
    }

    (*os) << timestamp << msg_type;
    return (*os);

}


std::string Message::endMsg( bool endl )
{

    return endl ? "[0m\n" : "[0m" ;

}


void Message::setColor( bool on )
{

    m_color = on;

}


}

