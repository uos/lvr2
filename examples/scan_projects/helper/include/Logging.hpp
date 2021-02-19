#ifndef LVR2_UTIL_LOGGING_HPP
#define LVR2_UTIL_LOGGING_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "lvr2/io/Timestamp.hpp"

namespace lvr2 {

// enum LoggingLevel {
//         DEBUG,
//         INFO,
//         WARNING,
//         HIGHLIGHT,
//         ERROR
//     };

// template<LoggingLevel L>
// struct Logger  {

//     std::ostringstream stream;

//     template<typename T>
//     Logger<L>& operator<<(T&& value) {
//         stream << value;
//         return *this;
//     }

//     Logger<L>& operator<<(std::ostream& (*os)(std::ostream&)) 
//     {
//         std::cout << "Test: " << stream.str() << os;
//         stream.str(""); // reset the string stream
//         // level = Level::INFO; // reset the level to info
//         return *this;
//     }
// };

struct Logger
{
    enum Level {
        DEBUG,
        INFO,
        WARNING,
        HIGHLIGHT,
        ERROR
    };

    std::ostringstream stream;
    Level level = Level::INFO;
    Level viz = Level::INFO;

    size_t indent = 0;

public:

    void setLoggerLevel(Level level)
    {
        viz = level;
    }

    void tab()
    {
        indent++;
    }

    void deltab()
    {
        if(indent > 0)
        {
            indent--;
        }
    }

    // here you can add parameters to the object, every line log
    Logger& operator()(Level l) {
        level = l;
        return *this;
    }

    // this overload receive the single values to append via <<
    template<typename T>
    Logger& operator<<(T&& value) {
        stream << value;
        return *this;
    }

    std::string getLevelColor() {
        std::string ret;
        if(level == Level::ERROR)
        {
            ret = "\033[1;31m";
        } else if(level == Level::HIGHLIGHT) {
            ret = "\033[1;32m";
        } else if(level == Level::WARNING) {
            ret = "\033[33m";
        } else if(level == Level::INFO) {
            ret = "\033[37m";
        } else if(level == Level::DEBUG) {
            ret = "\033[34m";
        } else {
            ret = "\033[37m";
        }
        return ret;
    }

    std::string getLevelString() {
        std::stringstream ss;
        if(level == Level::ERROR)
        {
            ss << "[ ERROR ] ";
        } else if(level == Level::HIGHLIGHT) {
            ss << "[ HIGH  ] ";
        } else if(level == Level::WARNING) {
            ss << "[ WARN  ] ";
        } else if(level == Level::INFO) {
            ss << "[ INFO  ] ";
        } else if(level == Level::DEBUG) {
            ss << "[ DEBUG ] ";
        } else {
            ss << "[" << (int)level <<  "] ";
        }
        return ss.str();
    }

    std::string getIndent()
    {
        std::stringstream ret;
        for(size_t i=0; i<indent; i++)
        {
            ret << "  ";
        }
        return ret.str();
    }

    // this overload intercept std::endl, to flush the stream and send all to std::cout
    Logger& operator<<(std::ostream& (*os)(std::ostream&)) 
    {
        
        if(level >= viz)
        {

            std::vector<std::string> result;
            std::string bla = stream.str();
            boost::split(result, bla, boost::is_any_of("\n"));


            std::string level_string = getLevelString();
            std::string level_color = getLevelColor();
            std::string indent_string = getIndent();

            for(size_t i=0; i<result.size(); i++)
            {
                if(i != 0)
                {
                    std::cout << "\n";
                }
                std::cout << level_color << timestamp << level_string << indent_string << result[i];
            }

            std::cout << "\033[0m" << os;
        }
        stream.str(""); // reset the string stream
        // level = Level::INFO; // reset the level to info
        return *this;
    }
};

static Logger LOG;

// // static Logger& LOG_ERROR = LOG(Logger::ERROR);
// // static Logger& LOG_HIGH = LOG(Logger::HIGHLIGHT);
// // static Logger& LOG_WARN = LOG(Logger::WARNING);
// // static Logger& LOG_INFO = LOG(Logger::INFO);
// // static Logger& LOG_DEBUG = LOG(Logger::DEBUG);


} // namespace lvr2

#endif // LVR2_UTIL_LOGGING_HPP