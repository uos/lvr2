#ifndef LOGGING
#define LOGGING

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <spdmon/spdmon.hpp>

#include <sstream>

namespace lvr2
{

using LogLevel = spdlog::level::level_enum;

class LVR2Monitor
{
public:
    LVR2Monitor(const spdlog::level::level_enum& level, const std::string& text, const size_t& max)
        : m_monitor(text, max)
    {

    }

    inline void operator++()
    {
        ++m_monitor;
    }

private:
     spdmon::Progress m_monitor;
};

class Logger
{
public:

    Logger()
    {
        m_logger = spdlog::stdout_color_mt("lvr2logger");
        m_logger->set_pattern("[%H:%M:%S:%e]%^[%-7l]%$ %v");
        m_level = spdlog::level::info;
    }

    void flush()
    {
        m_logger->log(m_level, m_buffer.str());
        m_buffer.str("");
        m_buffer.clear();
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
static Logger log;

template<typename T>
inline Logger& operator<<(Logger& log, const T& s)
{
    log.append(s);
    return log;
}

template<>
inline Logger& operator<<(Logger& log, const LoggerEndline& endl)
{
    log.flush();
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
