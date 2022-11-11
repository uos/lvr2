#ifndef LOGGING
#define LOGGING

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <spdmon/spdmon.hpp>

#include <sstream>
#include <iostream>

namespace lvr2
{

using LogLevel = spdlog::level::level_enum;



class Logger
{
private:

    Logger()
    {
        m_logger = spdlog::stdout_color_mt("lvr2logger");
        m_logger->set_pattern("[%H:%M:%S:%e]%^[%-7l]%$ %v");
        m_level = spdlog::level::info;
    }

public:
    static Logger& get()
    {
        static Logger inst;
        return inst;
    }

    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    void print()
    {
        m_logger->log(m_level, m_buffer.str());
        m_buffer.str("");
        m_buffer.clear();
    }

    void flush()
    {
        m_logger->flush();
    }

    template<typename T>
    void append(const T& text)
    {
        m_buffer << text;
    }

    void setLogLevel(const LogLevel& level)
    {
        m_level = level;
    }

private:
    std::shared_ptr<spdlog::logger> m_logger;
    LogLevel                        m_level;
    std::stringstream               m_buffer;
};

class Monitor
{
public:
    Monitor(const LogLevel& level, const std::string& text, const size_t& max, size_t width = 0)
        : m_monitor(text, max, false, stderr, width), m_prefixText(text)
    {

    }

    inline void operator++()
    {
        ++m_monitor;
    }

    ~Monitor()
    {
        m_monitor.Terminate();
    }

    void terminate()
    {
        m_monitor.Terminate();
    }

private:
     spdmon::Progress m_monitor;
     std::string m_prefixText;
};

struct LoggerEndline{};
struct LoggerError{};
struct LoggerWarning{};
struct LoggerTrace{};
struct LoggerInfo{};
struct LoggerDebug{};

static LoggerEndline endl;
static LoggerError error;
static LoggerWarning warning;
static LoggerTrace trace;
static LoggerInfo info;
static LoggerDebug debug;

using logout = Logger;

template<typename T>
inline Logger& operator<<(Logger& log, const T& s)
{
    log.append(s);
    return log;
}

template<>
inline Logger& operator<<(Logger& log, const LoggerEndline& endl)
{
    log.print();
    return log;
}

inline Logger& operator<<(Logger& log, const LoggerError& err)
{
    log.setLogLevel(LogLevel::err);
    return log;
}

inline Logger& operator<<(Logger& log, const LoggerWarning& warn)
{
    log.setLogLevel(LogLevel::warn);
    return log;
}

inline Logger& operator<<(Logger& log, const LoggerTrace& trace)
{
    log.setLogLevel(LogLevel::trace);
    return log;
}


inline Logger& operator<<(Logger& log, const LoggerDebug& trace)
{
    log.setLogLevel(LogLevel::debug);
    return log;
}

inline Logger& operator<<(Logger& log, const LoggerInfo& info)
{
    log.setLogLevel(LogLevel::info);
    return log;
}


} // namespace lvr2



#endif // LOGGING
