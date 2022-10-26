# spdmon
Progress monitor based on spdlog library. In just one function call visualize your loop progress!
<br>
It is simple, header-only library. Just copy `spdmon.hpp` file into you project and you are ready to go.
<br>
Main idea and algorithm is based on @epruesse [progress monitor](https://github.com/gabime/spdlog/issues/854).

## Usage
```cpp
std::vector<int> vec {0,1,2,3,4,5,6,7,8,9};
for(auto [logger, val] : spdmon::LogProgress(vec))
{
    logger->info("Hi info");
    std::this_thread::sleep_for(500ms);
}
```
```
Progress logger: 100% |████████████████████████████| 250000/250000 [00:00:33 / 00:00:00]
```

It can also be executed with additional spdlog::logger containing many sinks.
<br>
In that case `LoggerProgress` will not include sinks writing logs to `stdout` when creating temporary logger.
<br>
```cpp
std::vector<spdlog::sink_ptr> sinks;
sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("logfile", true));
auto combined_logger = std::make_shared<spdlog::logger>("name", begin(sinks), end(sinks));

spdmon::LoggerProgress monitor(combined_logger, "Progress", 40);
```

## Main features of LoggerProgress:
* Allows you to use default logger to send logs from many threads
* Works only on Linux platform
* Works with latest spdlog library
* Refactor code with Google C++ Style Guide
* Works only with `stdout`
* It is not using `std::recursive_mutex`
* It does not require any modification of spdlog library. Logging sink is based on `spdlog::sinks::ansicolor_sink`

## Benchmarks
Below are some benchmarks done in Ubuntu 64 bit, Intel i5-8250U CPU @ 3.4GHz. I have compared progress monitor to native `spdlog::sinks::ansicolor_sink`. Used [spdlog benchmark code](https://github.com/gabime/spdlog/blob/v1.x/bench/bench.cpp).

### Synchronous mode
```
[info] **************************************************************
[info] Single thread, 250,000 iterations
[info] **************************************************************
[spdlog::ansicolor_stdout_sink_mt]  [info]                Elapsed: 6.93 secs            36,083/sec
[Progress logger]                   [info]                Elapsed: 19.55 secs           12,788/sec
[Progress logger + monitor]         [info]                Elapsed: 27.51 secs            9,086/sec
[info] **************************************************************
[info] 10 threads, competing over the same logger object, 250,000 iterations
[info] **************************************************************
[spdlog::ansicolor_stdout_sink_mt]  [info]                Elapsed: 7.00 secs            35,703/sec
[Progress logger]                   [info]                Elapsed: 21.73 secs           11,507/sec
[Progress logger + monitor]         [info]                Elapsed: 29.70 secs            8,416/sec
```

## Example usage
```cpp
/*
 * Example usage code
*/

#include <thread>
#include <future>
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdmon/spdmon.hpp"

int main()
{
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
    for (int i = 0; i < count; i++)
    {
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

  using namespace std::chrono_literals;

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
  for (auto &t : threads)
  {
    if (t.joinable())
      t.join();
  }

  return 0;
}
```

## Authors
Huge thank you to [Elmar Pruesse](https://github.com/epruesse) for the idea and starting this project!
