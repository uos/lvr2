/**
 *
 * @file      Message.hpp
 * @brief     Class for managing different typed of console output.
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110929
 * @date      Created:       2011-09-28 19:30:34
 * @date      Last modified: 2011-09-29 14:22:37
 *
 **/


#ifndef LSSR_MESSAGE_H_INCLUDED
#define LSSR_MESSAGE_H_INCLUDED

#include <iostream>
#include <string>

using std::ostream;

namespace lssr
{

/**
 * \enum MsgType
 * \brief Specifies the available message types.
 *
 * The MsgType specifies the available message types. The output of the
 * messages are written to stdout or stderr according to rules specified for
 * each different message type.
 */
enum MsgType
{
    MSG_TYPE_MESSGAE,
    MSG_TYPE_HINT,
    MSG_TYPE_WARNING,
    MSG_TYPE_ERROR,
    MSG_TYPE_NONE
};


/**
 * \class Message Message.hpp "io/Message.hpp"
 * \brief A helper class for different types of console output.
 *
 * The Message class is a helper class for different types of messages printed
 * to the console. It is differentiated between messages, hints, warnings and
 * errors. A colorization of the output is supported.
 **/
class Message
{
    public:
        /**
         * \brief Constructor
         * 
         * \param color  Enables/Disables colorization of messages (default: true).
         **/
        Message( bool color = true );


        /**
         * \brief Print message of specified type using a format string.
         *
         * The print method specifies a way to print messages to the console by
         * using a format string and subsequently specifying the replacement
         * parameters. The syntax of the format sting is equal to the syntax
         * used for the printf function.
         * Important: The colorization is automatically disabled at the end of
         * the message if the format string end with a line break. Otherwise
         * the output has to be terminated by the return value of
         * lssr::Message::endMsg afterwards.
         *
         * \param t    Type of the specified message.
         * \param fmt  Format string.
         * \param ...  Replacement parameters.
         * \return     Output stream
         **/
        ostream& print( const MsgType t, const char* fmt, ... );


        /**
         * \brief Print message of specified type given as std::string object.
         *
         * The print method specifies a way to print messages given as
         * std::string object to the console. The output is automatically
         * formated according to the specified message type.
         * Important: The colorization is automatically disabled at the end of
         * the message if the message string end with a line break. Otherwise
         * the output has to be terminated by the return value of
         * lssr::Message::endMsg afterwards.
         *
         * \param t    Type of the specified message.
         * \param fmt  Message string.
         * \return     Output stream
         **/
        ostream& print( const MsgType t, const std::string fmt );


        /**
         * \brief Print message using a format string as normal message.
         *
         * The print method specifies a way to print messages to the console by
         * using a format string and subsequently specifying the replacement
         * parameters. The syntax of the format sting is equal to the syntax
         * used for the printf function.
         * Important: The colorization is automatically disabled at the end of
         * the message if the format string end with a line break. Otherwise
         * the output has to be terminated by the return value of
         * lssr::Message::endMsg afterwards.
         *
         * \param fmt  Format string.
         * \param ...  Replacement parameters.
         * \return     Output stream
         **/
        ostream& print( const char* fmt, ... );


        /**
         * \brief Print message given as std::string object as normal message.
         *
         * The print method specifies a way to print messages given as
         * std::string object to the console.
         * Important: The colorization is automatically disabled at the end of
         * the message if the message string end with a line break. Otherwise
         * the output has to be terminated by the return value of
         * lssr::Message::endMsg afterwards.
         *
         * \param fmt  Message string.
         * \return     Output stream
         **/
        ostream& print( const std::string fmt );


        /**
         * \brief Get output stream for specified message type.
         *
         * The print method without string parameter returns a reference to an
         * output stream according to the specified type of message. 
         * Important: The colorization and other formattings have to be
         * disabled by printing the return value of lssr::Message::endMsg.
         *
         * \param t    Type of the specified message (default: message).
         * \return     Output stream
         **/
        ostream& print( const MsgType t = MSG_TYPE_MESSGAE );

        /**
         * \brief Returns string for end of message.
         *
         * The endMsg method returns a std::string object for marking the end
         * of a message. This string disables all active formattings such as
         * colorizations.
         *
         * \param endl  Specifies if and newline character is also printed.
         * \return      End of message string.
         **/
        std::string endMsg( bool endl = true );


        /**
         * \brief Enables/Disables the colorization of the output.
         *
         * \param on  Enable/Disable colorizaion (default: true).
         **/
        void setColor( bool on = true );

    private:
        /**
         * \brief Flag specifying if colorization is enabled.
         **/
        bool m_color;

};


/**
 * \brief Global Message object for console output.
 **/
static Message g_msg( false );

} /* namespace lssr */

#endif /* LSSR_MESSAGE_H_INCLUDED */
