#ifndef LOGGING
#define LOGGING

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <spdmon/spdmon.hpp>

#include <sstream>

namespace lvr2
{

class LVR2Monitor
{
public:
    LVR2Monitor(const spdlog::level::level_enum& level, const std::string& text, const size_t& max)
        : m_monitor(spdlog::stdout_color_mt("lvr2alogger"), text, max)
    {

    }

    inline void operator++()
    {
        ++m_monitor;
    }

private:
     spdmon::LoggerProgress m_monitor;
};

using LogLevel = spdlog::level::level_enum;

class Logger
{
public:

    Logger()
    {
        m_logger = spdlog::stdout_color_mt("lvr2logger");
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

static LoggerEndline endl;
static Logger log;

template<typename T>
inline Logger& operator<<(Logger& log, const T& s)
{
    log.append(s);
    return log;
}

inline Logger& operator<<(Logger& log, const LogLevel& l)
{
    log.setLogLevel(l);
    return log;
}

inline Logger& operator<<(Logger& log, const LoggerEndline& endl)
{
    log.flush();
    return log;
}

} // namespace lvr2



#endif // LOGGING
