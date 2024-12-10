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


/**
 * @brief A simple wrapper class using spdmon to provide log output
 *        streams. Implemented as a singleton to produce consistent 
 *        log output
 * 
 */
class Logger
{
private:

    /**
     * @brief Construct a new Logger object
     */
    Logger()
    {
        m_logger = spdlog::stdout_color_mt("lvr2logger");
        m_logger->set_pattern("[%H:%M:%S:%e]%^[%-7l]%$ %v");
        m_level = spdlog::level::info;
    }

public:

    /**
     * @brief Returns the logger instance
     */
    static Logger& get()
    {
        static Logger inst;
        return inst;
    }

    /// Delete for singleton pattern
    Logger(Logger const&) = delete;

    /// Delete for singleton pattern
    void operator=(Logger const&) = delete;

    /// Prints buffer
    void print()
    {
        m_logger->log(m_level, m_buffer.str());
        m_buffer.str("");
        m_buffer.clear();
    }

    /// Flushes the internal buffer
    void flush()
    {
        m_logger->flush();
    }

    /**
     * @brief Appends the serialized (textual) object to 
     *        the internal buffer. Works for all types that
     *        support ostream serialization (i.e., an <<-Operator)
     * 
     * @param token Token to add to the internal buffer
     */
    template<typename T>
    void append(const T& token)
    {
        m_buffer << token;
    }

    /**
     * @brief Sets the current log level. All messages until the 
     *        next call or inserted format token will be logged 
     *        at the set log level
     * 
     * @param level The new log level
     */
    void setLogLevel(const LogLevel& level)
    {
        m_level = level;
    }

private:
    /// SPD mon logger instance
    std::shared_ptr<spdlog::logger> m_logger;

    /// Current log level
    LogLevel                        m_level;

    /// Stringstream buffer 
    std::stringstream               m_buffer;
};

/**
 * @brief A class to monitor progress 
 */
class Monitor
{
public:

    /**
     * @brief Constructs a new Monitor object
     * 
     * @param level     Loglevel to display the progress
     * @param text      Prefix text for progress bar
     * @param max       Number of exspected iterations
     * @param width     Width of the progress bar. Currently buggy, 
     *                  leave it at default!
     */
    Monitor(const LogLevel& level, const std::string& text, const size_t& max, size_t width = 0)
        : m_monitor(text, max, false, stderr, width), m_prefixText(text)
    {

    }

    /// Increment progress by one
    inline void operator++()
    {
        ++m_monitor;
    }

    /// Destructor
    ~Monitor()
    {
        m_monitor.Terminate();
    }

    /**
     * @brief   Removes to progress bar from the terminal. Call this 
     *          function once if the monitor object is still alive and you
     *          want to generate log output in a function.
     */
    inline void terminate()
    {
        m_monitor.Terminate();
    }

private:
     /// @brief SPD mon onject
     spdmon::Progress m_monitor;

     /// @brief Prefix text
     std::string m_prefixText;
};

// Marker structs for log levels
struct LoggerEndline{};
struct LoggerError{};
struct LoggerWarning{};
struct LoggerTrace{};
struct LoggerInfo{};
struct LoggerDebug{};

/// @brief Endline and flush for logger objects
static LoggerEndline endl;

/// @brief Marks error log level for streamed output
static LoggerError error;

/// @brief Marks warning log level for streamed output
static LoggerWarning warning;

/// @brief Marks trace log level for streamed output
static LoggerTrace trace;

/// @brief Marks info log level for streamed output
static LoggerInfo info;

/// @brief Marks debug log level for streamed output
static LoggerDebug debug;

// Alias for logger singleton
using logout = Logger;

/// @brief  Generic output for object the support ostreams
/// @param The logger instance
/// @param s A object to put into the log stream
/// @return The modified logger object
template<typename T>
inline Logger& operator<<(Logger& log, const T& s)
{
    log.append(s);
    return log;
}

/// @brief Spezialization for endl marker
template<>
inline Logger& operator<<(Logger& log, const LoggerEndline& endl)
{
    log.print();
    return log;
}

/// @brief Spezialization for error log level marker
inline Logger& operator<<(Logger& log, const LoggerError& err)
{
    log.setLogLevel(LogLevel::err);
    return log;
}

/// @brief Spezialization for warning log level marker
inline Logger& operator<<(Logger& log, const LoggerWarning& warn)
{
    log.setLogLevel(LogLevel::warn);
    return log;
}

/// @brief Spezialization for trace log level marker
inline Logger& operator<<(Logger& log, const LoggerTrace& trace)
{
    log.setLogLevel(LogLevel::trace);
    return log;
}

/// @brief Spezialization for debug log level marker
inline Logger& operator<<(Logger& log, const LoggerDebug& trace)
{
    log.setLogLevel(LogLevel::debug);
    return log;
}

/// @brief Spezialization for info log level marker
inline Logger& operator<<(Logger& log, const LoggerInfo& info)
{
    log.setLogLevel(LogLevel::info);
    return log;
}


} // namespace lvr2



#endif // LOGGING
