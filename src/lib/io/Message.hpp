/**
 *
 * @file      Message.hpp
 * @brief     Class for managing different typed of console output.
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110928
 * @date      09/28/2011 07:30:34 PM
 *
 **/


#ifndef LSSR_MESSAGE_H_INCLUDED
#define LSSR_MESSAGE_H_INCLUDED

#include <iostream>
#include <string>

using std::ostream;

namespace lssr
{

enum MsgType
{
    MSG_TYPE_MESSGAE,
    MSG_TYPE_HINT,
    MSG_TYPE_WARNING,
    MSG_TYPE_ERROR,
    MSG_TYPE_NONE
};

class Message
{
    public:
        Message( bool color = true );

        ostream& print( const MsgType t, const char* fmt, ... );
        ostream& print( const MsgType t, const std::string fmt );
        ostream& print( const char* fmt, ... );
        ostream& print( const std::string fmt );
        ostream& print( const MsgType t = MSG_TYPE_MESSGAE );

        std::string endMsg( bool endl = true );
        void setColor( bool on = true );

    private:
        bool m_color;

};

static Message g_msg( false );

} /* namespace lssr */

#endif /* LSSR_MESSAGE_H_INCLUDED */
