/*
 * Example usage code
 */

#include <future>
#include <thread>
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdmon/spdmon.hpp"

using namespace std::chrono_literals;

int main() {
    /*
     * The shortest way to use Progress Monitor. Works with every iterable container
    */
    std::vector<int> vec {0,1,2,3,4,5,6,7,8,9};
    for(auto [logger, val] : spdmon::LogProgress(vec))
    {
        logger->info("Hi info");
        std::this_thread::sleep_for(500ms);
    }

    /*
     * Create logger with many sinks. It can include stdout sinks
     */
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
    sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("logfile", true));
    auto combined_logger = std::make_shared<spdlog::logger>("name", begin(sinks), end(sinks));

    /*
     * Create progress logger. It checks for all stdout sinks and discard them
     * Instead of them it creates custom one, logging to stdout with progress bar
     * and makes it global default logger.
     */
    spdmon::LoggerProgress monitor(combined_logger, "Progress", 40);

    /* We can get newly created logger from progress logger to log new data */
    auto new_logger = monitor.GetLogger();
    new_logger->set_pattern("[thread %t] %+");
    new_logger->set_level(spdlog::level::trace);

    /*
     * We use lambda function to be passed to different logging threads
     */
    auto print_lambda = [&monitor, &new_logger](auto duration, int count) {
        for (int i = 0; i < count; i++) {
            spdlog::trace("Hi trace");
            spdlog::debug("Hi debug");
            spdlog::info("Hi info");
            new_logger->warn("Hi warn");
            new_logger->error("Hi err");
            new_logger->critical("Hi critical");

            ++monitor;

            std::this_thread::sleep_for(duration);
        }
    };

    /*
     * Create few threads which will be logging data through progress logger
     * and global logger
     */
    std::vector<std::thread> threads;
    threads.push_back(std::thread{print_lambda, 250ms, 10});
    threads.push_back(std::thread{print_lambda, 750ms, 10});
    threads.push_back(std::thread{print_lambda, 350ms, 10});
    threads.push_back(std::thread{print_lambda, 500ms, 10});

    /*
     * Wait for threads to finish their jobs
     */
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }

    return 0;
}
