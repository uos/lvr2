#include <lvr2/util/Logging.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdmon/spdmon.hpp>

// This file is compiled with -fvisibility=hidden to prevent spdlog and spdmon symbols
// to be public in the dynamic library
#ifdef LVR2_BUILDING_SHARED
    #define LVR2_API __attribute__ ((visibility ("default")))
#else
    #define LVR2_API
#endif

namespace lvr2
{

LVR2_API Logger::Logger()
{
    m_logger = spdlog::stdout_color_mt("lvr2logger");
    m_logger->set_pattern("[%H:%M:%S:%e]%^[%-7l]%$ %v");
    m_level = LogLevel::info;
}

LVR2_API void Logger::print()
{
    spdlog::level::level_enum level;
    
    switch(m_level)
    {
        case LogLevel::trace: level = spdlog::level::trace; break;
        case LogLevel::debug: level = spdlog::level::debug; break;
        case LogLevel::info: level = spdlog::level::info; break;
        case LogLevel::warning: level = spdlog::level::warn; break;
        case LogLevel::error: level = spdlog::level::err; break;

    }

    m_logger->log(level, m_buffer.str());
    m_buffer.str("");
    m_buffer.clear();
}

LVR2_API void Logger::flush()
{
    m_logger->flush();
}

LVR2_API Monitor::Monitor(const LogLevel& level, const std::string& text, const size_t& max, size_t width)
: m_monitor(std::make_shared<spdmon::Progress>(text, max, false, stderr, width))
, m_prefixText(text)
{
}

LVR2_API void Monitor::terminate()
{
    m_monitor->Terminate();
}

LVR2_API void Monitor::operator++()
{
    ++(*m_monitor);
}

} // namespace lvr2
