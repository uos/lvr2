#include <list>
#include <mutex>
#include <memory>
#include <string>
#include <array>
#include <atomic>
#include <iostream>

#include <spdlog/fmt/bundled/chrono.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <spdlog/details/console_globals.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/details/os.h>
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <spdlog/sinks/ansicolor_sink.h>

/*
 * Check if system is Windows
*/
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#error Progress sink only work on Linux platform
#else
#include <signal.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace spdmon
{
/*
 * This is a short macro for deleting a class's copy constructor and
 * copy assignment operator.
*/
#define SPDMON_DECLARE_NON_COPYABLE(classInst)  \
    classInst(const classInst &other) = delete; \
    classInst &operator=(const classInst &other) = delete;

/*
 * This is a short macro for deleting a class's move constructor and
 * move assignment operator.
*/
#define SPDMON_DECLARE_NON_MOVEABLE(classInst)      \
    classInst(classInst &&other) noexcept = delete; \
    classInst &operator=(classInst &&other) noexcept = delete;

/*
 * This is a short macro for declaring class's default copy constructor and
 * move assignment operator.
*/
#define SPDMON_DECLARE_DEFAULT_COPYABLE(classInst) \
    classInst(const classInst &other) = default;   \
    classInst &operator=(const classInst &other) = default;

    /*
 * Bar symbols coded in unicode used for displaying spdmon
*/
    static const char *kBarSymsUnicode[] = {
        " ", "\xE2\x96\x8F", "\xE2\x96\x8E", "\xE2\x96\x8D", "\xE2\x96\x8C", "\xE2\x96\x8B", "\xE2\x96\x8A", "\xE2\x96\x89",
        "\xE2\x96\x88"};

    /*
 * Bar symbols coded in ascii used for displaying spdmon
*/
    static const char *kBarSymsAscii[] = {" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#"};

    /*
 * Abstract base rendering spdmon bar
 * needs ShowProgress() filled in
*/
    class BaseProgress
    {
    public:
        using clock_t = std::chrono::steady_clock;
        using timepoint_t = clock_t::time_point;
        using duration_t = clock_t::duration;

        explicit BaseProgress(std::string desc = "", unsigned int total = 0, bool ascii = false)
            : total_(total),
              desc_(desc),
              bar_syms_(ascii ? kBarSymsAscii : kBarSymsUnicode),
              nsyms_(ascii ? std::extent<decltype(kBarSymsAscii)>::value
                           : std::extent<decltype(kBarSymsUnicode)>::value)
        {
        }

        virtual ~BaseProgress() = default;

        void Restart(std::string desc = "", unsigned int total = 0)
        {
            n_ = 0;
            total_ = total;
            desc_ = desc;
            ShowProgress();
        }

        unsigned int Count()
        {
            return n_;
        }

        BaseProgress &operator++()
        {
            Update();
            return *this;
        }

        void operator+=(unsigned int n)
        {
            Update(n);
        }

        void SetTotal(unsigned int n)
        {
            total_ = n;
        }

        unsigned int Size()
        {
            return total_;
        }

        void FormatBarTo(fmt::memory_buffer &buf, unsigned int width, float frac)
        {
            if (width == 0)
            {
                return;
            }

            buf.reserve(buf.size() + width * 3);
            auto it = std::back_inserter(buf);

            const auto complete = static_cast<unsigned int>(frac * static_cast<float>(width * nsyms_));
            const size_t full_blocks = complete / nsyms_;
            const size_t frac_block = complete % nsyms_;
            size_t fill_length = width - full_blocks;

            auto append_n = [&](size_t n, size_t idx) {
                size_t len_sym = strlen(bar_syms_[idx]);
                if (len_sym == 1)
                {
                    std::fill_n(it, n, bar_syms_[idx][0]);
                }
                else
                {
                    for (size_t i = 0; i < n; i++)
                        std::copy_n(bar_syms_[idx], len_sym, it);
                }
            };

            append_n(full_blocks, nsyms_ - 1);
            if (frac_block > 0)
            {
                append_n(1, frac_block);
                --fill_length;
            }
            append_n(fill_length, 0);
        }

        void Update(unsigned int n = 1)
        {
            n_ += n;
            if (n_ == total_)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ShowProgress();
                return;
            }
            if (n_ >= last_print_n_ + miniterations_)
            {
                timepoint_t now = clock_t::now();
                duration_t delta_time = now - last_update_;
                if (delta_time > mininterval_)
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    last_update_ = now;
                    miniterations_ = (n_.load() - last_print_n_) * static_cast<unsigned int>(mininterval_ / delta_time);
                    ShowProgress(now);
                }
            }
        }

        void RenderProgress(timepoint_t now, unsigned int width, fmt::memory_buffer &buf)
        {
            last_print_n_ = n_;
            const auto elapsed = now - started_at_;

            if (total_ == 0)
            {
                fmt::format_to(buf, kNoTotalFmt, fmt::arg("desc", desc_),
                               fmt::arg("n", n_),
                               fmt::arg("elapsed", elapsed),
                               fmt::arg("eol", kTermEol));
                return;
            }

            const float frac = static_cast<float>(n_.load()) / static_cast<float>(total_);
            // auto eta = elapsed * (1 / frac - 1);
            const auto remaining = (frac > 0) ? elapsed * (1 / frac - 1) : duration_t(0);

            fmt::memory_buffer right;

            const float percent = frac * 100;

            fmt::format_to(buf, kLbarFmt, fmt::arg("desc", desc_),
                           fmt::arg("frac", percent));
            fmt::format_to(right, kRbarFmt, fmt::arg("n", n_),
                           fmt::arg("total", total_),
                           fmt::arg("elapsed", elapsed),
                           fmt::arg("remaining", remaining),
                           fmt::arg("eol", kTermEol));

            const auto space_for_bar = static_cast<unsigned int>(width - buf.size() - right.size() + kTermEol.size());
            if (space_for_bar > 0)
            {
                FormatBarTo(buf, space_for_bar, frac);
            }
            buf.reserve(buf.size() + right.size());
            std::copy(right.begin(), right.end(), std::back_inserter(buf));
        }

        virtual void ShowProgress(timepoint_t now = clock_t::now()) = 0;

    private:
        std::atomic<unsigned int> n_{0};
        unsigned int last_print_n_{0};
        unsigned int total_{0};
        std::string desc_{""};
        const char **bar_syms_{nullptr};
        unsigned int nsyms_{0};
        timepoint_t started_at_{clock_t::now()};
        timepoint_t last_update_{std::chrono::seconds(0)};
        duration_t mininterval_{std::chrono::milliseconds(10)};
        unsigned int miniterations_{1};
        std::string bar_tpl_{""};
        std::mutex mutex_;

        const std::string kTermEraseLine = "\x1B[0K";
        const std::string kTermEol = "\n";
        const std::string kLbarFmt = "{desc}: {frac:3.0f}% |";
        const std::string kRbarFmt = "| {n}/{total} [{elapsed:%T} / {remaining:%T}]{eol}";
        const std::string kNoTotalFmt = "{desc}: {n} [{elapsed:%T}]{eol}";
    };

    /* Progress monitor writing directly to a file */
    class Progress final : public BaseProgress
    {
    public:
        explicit Progress(std::string desc = "", unsigned int total = 0, bool ascii = false,
                          FILE *file = stderr, unsigned int width = 0)
            : BaseProgress(desc, total, ascii),
              width_(width),
              file_(file)

        {
            if (width_ == 0)
            {
                UpdateTermWidth();
            }
        }

        ~Progress() override = default;

        SPDMON_DECLARE_NON_COPYABLE(Progress)
        SPDMON_DECLARE_NON_MOVEABLE(Progress)

        void UpdateTermWidth()
        {
            int fd = fileno(file_);
            struct winsize size;
            if (ioctl(fd, TIOCGWINSZ, &size) == 0)
            {
                width_ = size.ws_col;
            }
        }

        void ShowProgress(timepoint_t now = clock_t::now()) final
        {
            // called from BaseProgress when it decided the spdmon bar needs
            // to be printed again
            fmt::memory_buffer buf;
            RenderProgress(now, width_, buf);
            std::copy(kTermMoveUp.begin(), kTermMoveUp.end(), std::back_inserter(buf));
            fwrite(buf.data(), 1, buf.size(), file_);
            fflush(file_);
        }

        const std::string kTermMoveUp = "\x1B[A";

    private:
        unsigned int width_;
        FILE *file_;
    };

    class StatusLine;

    class StatusLineRegistry
    {
    public:
        StatusLineRegistry()
        {
            status_lines_.reserve(20);
        }

        virtual ~StatusLineRegistry() = default;

        SPDMON_DECLARE_NON_COPYABLE(StatusLineRegistry)
        SPDMON_DECLARE_NON_MOVEABLE(StatusLineRegistry)

        void AddStatusLine(StatusLine *msg)
        {
            status_lines_.push_back(msg);
        }
        void RemoveStatusLine(StatusLine *msg)
        {
            status_lines_.erase(
                std::remove(status_lines_.begin(), status_lines_.end(), msg),
                status_lines_.end());
        }
        const std::vector<StatusLine *> &GetStatusLines()
        {
            return status_lines_;
        }

        virtual void PrintStatusLineRegistry() = 0;

    private:
        std::vector<StatusLine *> status_lines_ = {};
        int change_since_last_print_{0};
    };

    class StatusLine
    {
    public:
        explicit StatusLine(spdlog::level::level_enum level)
            : level_(level)
        {
        }

        virtual ~StatusLine()
        {
            if (logger_.get() == nullptr)
                return;

            /* on deletion, deregister from all sinks */
            for (auto &sink : logger_->sinks())
            {
                auto ptr = dynamic_cast<StatusLineRegistry *>(sink.get());
                if (ptr)
                {
                    ptr->RemoveStatusLine(this);
                }
            }
        }

        SPDMON_DECLARE_NON_COPYABLE(StatusLine)
        SPDMON_DECLARE_NON_MOVEABLE(StatusLine)

        void RegisterSinks(std::shared_ptr<spdlog::logger> logger)
        {
            logger_ = logger;
            /* on creation, register it with all sinks that understand about us */
            for (auto &sink : logger_->sinks())
            {
                auto ptr = dynamic_cast<StatusLineRegistry *>(sink.get());
                if (ptr)
                {
                    ptr->AddStatusLine(this);
                }
            }
        }

        /* trigger (re)print of this status line */
        void TriggerPrintStatusLines()
        {
            for (const auto &sink : logger_->sinks())
            {
                if (sink->should_log(GetLevel()))
                {
                    auto ptr = dynamic_cast<StatusLineRegistry *>(sink.get());
                    if (ptr)
                    {
                        ptr->PrintStatusLineRegistry();
                    }
                }
            }
        }

        virtual void RenderStatusLine(fmt::memory_buffer &buf, unsigned int width) = 0;

        static const char *MagicFilename()
        {
            static const char *file = "PROGRESS MONITOR";
            return file;
        }

        bool ShouldLog()
        {
            return logger_->should_log(level_);
        }

        spdlog::level::level_enum GetLevel()
        {
            return level_;
        }

        void SendLogMsg(fmt::memory_buffer &buf)
        {
            spdlog::source_loc loc;
            loc.filename = MagicFilename();
            spdlog::details::log_msg msg{loc, logger_->name(), level_, spdlog::string_view_t{buf.data(), buf.size()}};
            for (const auto &sink : logger_->sinks())
            {
                if (sink->should_log(GetLevel()))
                {
                    sink->log(msg);
                }
            }
        }

    private:
        std::shared_ptr<spdlog::logger> logger_{nullptr};
        spdlog::level::level_enum level_{spdlog::level::off};
    };

    class SigwinchMixin
    {
    public:
        SPDMON_DECLARE_NON_COPYABLE(SigwinchMixin)
        SPDMON_DECLARE_NON_MOVEABLE(SigwinchMixin)

    protected:
        explicit SigwinchMixin(bool install)
        {
            if (install)
            {
                InstallHandler();
            }
            Instances().push_back(this);
        }

        virtual ~SigwinchMixin() { Instances().remove(this); }

        inline bool CheckGotSigwinch()
        {
            if (GotSigwinch() != 0)
            {
                GotSigwinch() = 0;
                NotifyInstances();
            }
            if (needs_update_ > 0)
            {
                needs_update_ = 0;
                return true;
            }
            return false;
        }

    private:
        static sig_atomic_t &GotSigwinch()
        {
            static sig_atomic_t flag;
            return flag;
        }

        static std::list<SigwinchMixin *> &Instances()
        {
            static std::list<SigwinchMixin *> list;
            return list;
        }

        static void HandleSigwinch(int) { GotSigwinch() = 1; }

        static void InstallHandler()
        {
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            if (sigaction(SIGWINCH, nullptr, &sa))
            {
                return; // failed
            }
            if (sa.sa_handler == HandleSigwinch)
            {
                return; // already installed
            }
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = HandleSigwinch;
            sigfillset(&sa.sa_mask); // block other signals during handler
            if (sigaction(SIGWINCH, &sa, nullptr))
            {
                return; // failed
            }
        }

        static void NotifyInstances()
        {
            // std::lock_guard<mutex_t> lock(log_sink::mutex_);
            for (auto &instance : Instances())
            {
                instance->needs_update_ = true;
            }
        }

        std::atomic_bool needs_update_{false};
    };

    template <class ConsoleMutex>
    class TerminalSink final : public spdlog::sinks::ansicolor_sink<ConsoleMutex>,
                               public StatusLineRegistry,
                               public SigwinchMixin
    {
    public:
        using log_sink = spdlog::sinks::ansicolor_sink<ConsoleMutex>;
        using mutex_t = typename ConsoleMutex::mutex_t;

        TerminalSink() : log_sink(stdout, spdlog::color_mode::automatic), SigwinchMixin(log_sink::should_color())
        {
            if (log_sink::should_color())
            {
                UpdateTermWidth();
            }
        }

        ~TerminalSink() final = default;

        SPDMON_DECLARE_NON_COPYABLE(TerminalSink)
        SPDMON_DECLARE_NON_MOVEABLE(TerminalSink)

        // normal log message coming in
        void log(const spdlog::details::log_msg &msg) final { Print(&msg); }

        // Update to status coming in
        void PrintStatusLineRegistry() final { Print(nullptr); }

        void Print(const spdlog::details::log_msg *msg)
        {
            if (not log_sink::should_color())
            { // not a tty
                if (msg != nullptr)
                {                        // got a msg
                    log_sink::log(*msg); // pass it upstream
                }
                return;
            }

            if (CheckGotSigwinch())
            {
                UpdateTermWidth();
            }

            // render status lines
            fmt::memory_buffer status_text;
            unsigned int nlines = 0;
            for (const auto line : GetStatusLines())
            {
                if (log_sink::should_log(line->GetLevel()))
                {
                    line->RenderStatusLine(status_text, ncols_);
                    ++nlines;
                }
            }

            // move back up to last log line
            std::lock_guard<mutex_t> lock(this->mutex_);
            for (unsigned int i = 0; i < last_status_lines_; ++i)
            {
                fwrite(kTermMoveUp.data(), 1, kTermMoveUp.size(), stdout);
            }
            last_status_lines_ = nlines;

            // print message if we have one and it's not an update for other sinks
            if (msg != nullptr && msg->source.filename != StatusLine::MagicFilename())
            {
                fwrite(term_clear_line.data(), 1, term_clear_line.size(), stdout);
                log_sink::log(*msg);
            }

            fwrite(status_text.data(), 1, status_text.size(), stdout);
            fflush(stdout);
        }

        void UpdateTermWidth()
        {
            int fd = fileno(stdout);
            struct winsize size;
            if (ioctl(fd, TIOCGWINSZ, &size) == 0)
            {
                // std::lock_guard<mutex_t> lock(log_sink::mutex_);
                std::lock_guard<mutex_t> lock(this->mutex_);
                ncols_ = size.ws_col;
            }
        }

        const std::string kTermMoveUp = "\x1B[A";
        const std::string term_clear_line = "\x1B[K";

    private:
        unsigned int ncols_{60};
        unsigned int last_status_lines_{0};
        mutex_t mutex_;
    };

    using terminal_stdout_sink_mt = TerminalSink<spdlog::details::console_mutex>;
    using terminal_stdout_sink_st = TerminalSink<spdlog::details::console_nullmutex>;
    using terminal_stderr_sink_mt = TerminalSink<spdlog::details::console_mutex>;
    using terminal_stderr_sink_st = TerminalSink<spdlog::details::console_nullmutex>;

    template <typename Factory = spdlog::default_factory>
    inline std::shared_ptr<spdlog::logger> stdout_terminal_mt(const std::string &logger_name)
    {
        return Factory::template create<terminal_stdout_sink_mt>(logger_name);
    }

    template <typename Factory = spdlog::default_factory>
    inline std::shared_ptr<spdlog::logger> stderr_terminal_mt(const std::string &logger_name)
    {
        return Factory::template create<terminal_stderr_sink_mt>(logger_name);
    }

    template <typename Factory = spdlog::default_factory>
    inline std::shared_ptr<spdlog::logger> stdout_terminal_st(const std::string &logger_name)
    {
        return Factory::template create<terminal_stdout_sink_st>(logger_name);
    }
    template <typename Factory = spdlog::default_factory>
    inline std::shared_ptr<spdlog::logger> stderr_terminal_st(const std::string &logger_name)
    {
        return Factory::template create<terminal_stderr_sink_st>(logger_name);
    }

#define IS_ONE_OF_TYPE(sink)                                                    \
    (typeid(*(sink)) == typeid(spdlog::sinks::stdout_sink_st)) ||               \
        (typeid(*(sink)) == typeid(spdlog::sinks::stdout_sink_mt)) ||           \
        (typeid(*(sink)) == typeid(spdlog::sinks::ansicolor_stdout_sink_st)) || \
        (typeid(*(sink)) == typeid(spdlog::sinks::ansicolor_stdout_sink_mt))

    class LoggerProgress final : public BaseProgress, StatusLine
    {
    public:
        explicit LoggerProgress(std::shared_ptr<spdlog::logger> logger,
                                std::string desc = "",
                                unsigned int total = 0,
                                bool ascii = false,
                                unsigned int /*width*/ = 0,
                                spdlog::level::level_enum level = spdlog::level::warn)
            : BaseProgress(desc, total, ascii),
              StatusLine(level)
        {
            auto progress_logger = std::make_shared<spdmon::terminal_stdout_sink_mt>();

            std::vector<spdlog::sink_ptr> sinks;
            for (const auto &sink : logger->sinks())
            {
                if (not(IS_ONE_OF_TYPE(sink)))
                {
                    sinks.push_back(sink);
                }
            }
            sinks.push_back(progress_logger);
            custom_logger_ = std::make_shared<spdlog::logger>(desc, sinks.begin(), sinks.end());

            default_logger_ = spdlog::default_logger();
            spdlog::set_default_logger(custom_logger_);
            StatusLine::RegisterSinks(custom_logger_);
        }

        explicit LoggerProgress(std::string desc = "", unsigned int total = 0, bool ascii = false,
                                unsigned int /*width*/ = 0,
                                spdlog::level::level_enum level = spdlog::level::warn)
            : BaseProgress(desc, total, ascii),
              StatusLine(level)
        {
            default_logger_ = spdlog::default_logger();
            custom_logger_ = spdmon::stdout_terminal_mt("Progress logger");
            spdlog::set_default_logger(custom_logger_);
            StatusLine::RegisterSinks(custom_logger_);
        }

        ~LoggerProgress()
        {
            spdlog::set_default_logger(default_logger_);
        };

        SPDMON_DECLARE_NON_COPYABLE(LoggerProgress)
        SPDMON_DECLARE_NON_MOVEABLE(LoggerProgress)

        void ShowProgress(timepoint_t now = clock_t::now()) final
        {
            if (!ShouldLog())
            {
                // early quit if our logger has loglevel below ourselves
                return;
            }
            const duration_t delta_time = now - last_update_;
            StatusLine::TriggerPrintStatusLines();
            if (delta_time > mininterval_)
            {
                last_update_ = now;

                fmt::memory_buffer buf;
                BaseProgress::RenderProgress(now, 60, buf);
                StatusLine::SendLogMsg(buf);
            }
        }

        void RenderStatusLine(fmt::memory_buffer &buf, unsigned int width) final
        {
            BaseProgress::RenderProgress(clock_t::now(), width, buf);
        }

        std::shared_ptr<spdlog::logger> GetLogger()
        {
            return this->custom_logger_;
        }

    private:
        timepoint_t last_update_{std::chrono::seconds(0)};
        duration_t mininterval_{std::chrono::milliseconds(500)};
        std::shared_ptr<spdlog::logger> default_logger_{nullptr};
        std::shared_ptr<spdlog::logger> custom_logger_{nullptr};
    };
} // namespace spdmon

#include "spdlog/spdlog.h"
#include "spdlog/sinks/ansicolor_sink.h"

#ifdef SPDLOG_FMT_EXTERNAL
#include <fmt/locale.h>
#else
#include "spdlog/fmt/bundled/format.h"
#endif

#include "utils.hpp"

#include <atomic>
#include <cstdlib> // EXIT_FAILURE
#include <memory>
#include <string>
#include <thread>

void bench(int howmany, std::shared_ptr<spdlog::logger> log)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;

    auto start = high_resolution_clock::now();
    for (auto i = 0; i < howmany; ++i)
    {
        log->info("Hello logger: msg number {}", i);
    }

    auto delta = high_resolution_clock::now() - start;
    auto delta_d = duration_cast<duration<double>>(delta).count();

    spdlog::info(
        fmt::format(std::locale("en_US.UTF-8"), "{:<30} Elapsed: {:0.2f} secs {:>16L}/sec", log->name(), delta_d, int(howmany / delta_d)));
    spdlog::drop(log->name());
}

void bench_mt(int howmany, std::shared_ptr<spdlog::logger> log, size_t thread_count)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    auto start = high_resolution_clock::now();
    for (size_t t = 0; t < thread_count; ++t)
    {
        threads.emplace_back([&]() {
            for (int j = 0; j < howmany / static_cast<int>(thread_count); j++)
            {
                log->info("Hello logger: msg number {}", j);
            }
        });
    }

    for (auto &t : threads)
    {
        t.join();
    };

    auto delta = high_resolution_clock::now() - start;
    auto delta_d = duration_cast<duration<double>>(delta).count();
    spdlog::info(
        fmt::format(std::locale("en_US.UTF-8"), "{:<30} Elapsed: {:0.2f} secs {:>16L}/sec", log->name(), delta_d, int(howmany / delta_d)));
    spdlog::drop(log->name());
}

static const size_t file_size = 30 * 1024 * 1024;
static const size_t rotating_files = 5;
static const int max_threads = 1000;

void bench_threaded_logging(size_t threads, int iters)
{
    spdlog::info("**************************************************************");
    spdlog::info(fmt::format(std::locale("en_US.UTF-8"), "Multi threaded: {:L} threads, {:L} messages", threads, iters));
    spdlog::info("**************************************************************");

    {
        auto sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("name", sink);
        bench_mt(iters, std::move(logger), threads);
    }
    {
        spdmon::LoggerProgress monitor("Progress", iters);
        bench_mt(iters, std::move(monitor.GetLogger()), threads);
    }
    {
        spdmon::LoggerProgress monitor("Progress", iters);
        auto log = monitor.GetLogger();
        int howmany = iters;
        int thread_count = threads;

        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;

        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        auto start = high_resolution_clock::now();
        for (size_t t = 0; t < thread_count; ++t)
        {
            threads.emplace_back([&]() {
                for (int j = 0; j < howmany / static_cast<int>(thread_count); j++)
                {
                    log->info("Hello logger: msg number {}", j);
                    ++monitor;
                }
            });
        }

        for (auto &t : threads)
        {
            t.join();
        };

        auto delta = high_resolution_clock::now() - start;
        auto delta_d = duration_cast<duration<double>>(delta).count();
        spdlog::info(
            fmt::format(std::locale("en_US.UTF-8"), "{:<30} Elapsed: {:0.2f} secs {:>16L}/sec", log->name(), delta_d, int(howmany / delta_d)));
        spdlog::drop(log->name());
    }
}

void bench_single_threaded(int iters)
{
    spdlog::info("**************************************************************");
    spdlog::info(fmt::format(std::locale("en_US.UTF-8"), "Single threaded: {} messages", iters));
    spdlog::info("**************************************************************");

    {
        auto sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("name", sink);

        bench(iters, std::move(logger));
    }
    {
        spdmon::LoggerProgress monitor("Progress", iters);
        bench(iters, std::move(monitor.GetLogger()));
    }
    {
        spdmon::LoggerProgress monitor("Progress", iters);
        auto log = monitor.GetLogger();
        int howmany = iters;

        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;

        auto start = high_resolution_clock::now();
        for (auto i = 0; i < howmany; ++i)
        {
            log->info("Hello logger: msg number {}", i);
            ++monitor;
        }

        auto delta = high_resolution_clock::now() - start;
        auto delta_d = duration_cast<duration<double>>(delta).count();

        spdlog::info(
            fmt::format(std::locale("en_US.UTF-8"), "{:<30} Elapsed: {:0.2f} secs {:>16L}/sec", log->name(), delta_d, int(howmany / delta_d)));
        spdlog::drop(log->name());
    }
}

int main(int argc, char *argv[])
{
    spdlog::set_automatic_registration(false);
    spdlog::default_logger()->set_pattern("[%^%l%$] %v");
    int iters = 250000;
    size_t threads = 10;

    bench_single_threaded(iters);
    bench_threaded_logging(threads, iters);

    return EXIT_SUCCESS;
}