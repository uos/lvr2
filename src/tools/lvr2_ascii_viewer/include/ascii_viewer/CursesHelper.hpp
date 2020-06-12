#ifndef CURSES_HELPER_HPP
#define CURSES_HELPER_HPP

#include <ncursesw/ncurses.h>
#include <locale>
#include <iostream>
#include <signal.h>

static mmask_t MOUSE_MOVE = 134217728;
static mmask_t MOUSE_SCROLL_UP = 524288;

static short CURSES_NUM_GRAYS = 200;
static short CURSES_CUSTOM_COLORS_SHIFT = 10;


static void destroy_curses()
{
    endwin();
    // disable mouse movements
    printf("\033[?1003l\n");
}

static void curses_sigint(int sig)
{
    int W = getmaxx(stdscr);
    int H = getmaxy(stdscr);
    mvprintw(H-1, W-1-6, "SIGINT");
    // destroy_curses();
    // exit(0);
}

static void set_num_grays(short num_grays)
{
    CURSES_NUM_GRAYS = num_grays;
}

static void init_grayscale(short num_grays)
{
    set_num_grays(num_grays);
    init_color(COLOR_BLACK, 0, 0, 0);
    
    short step = static_cast<short>(1000 / num_grays); 

    // init colors
    for(short i=0; i < num_grays; i++)
    {
        short color_id = i + CURSES_CUSTOM_COLORS_SHIFT;
        short pair_id  = i+1 + CURSES_CUSTOM_COLORS_SHIFT;

        init_color(color_id, i * step, i * step, i * step);
        init_pair(pair_id, color_id, COLOR_BLACK);
    }
}

static void console_print_grayscale(
    int y, int x,
    wchar_t sign,
    short gray
)
{
    short color_pair_id = gray + CURSES_CUSTOM_COLORS_SHIFT;

    attron(COLOR_PAIR(color_pair_id));
    mvprintw(y, x, "%lc", sign);
    attroff(COLOR_PAIR(color_pair_id));
}

static void console_print_grayscale(
    int y, int x, 
    wchar_t sign,
    float gray)
{
    short gray_short = static_cast<short>(gray * static_cast<float>(CURSES_NUM_GRAYS) );
    console_print_grayscale(y, x, sign, gray_short);
}

static void clear_background()
{
    int W = getmaxx(stdscr);
    int H = getmaxy(stdscr);

    for(int y=0; y<H; y++)
    {
        for(int x=0; x<W; x++)
        {
            wchar_t sign = L'\u2800';
            console_print_grayscale(y, x, sign, 0.0f);
        }
    }
}

static void init_mouse()
{
    mouseinterval(0);
    mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
    printf("\033[?1003h\n");
}

static void init_curses(unsigned int num_grayscale)
{
    signal(SIGINT, curses_sigint);

    initscr();
    clear();
    cbreak();
    keypad(stdscr, TRUE);
    timeout(5);
    
    // test this
    nodelay(stdscr, TRUE);
    // notimeout(stdscr, TRUE);
    // wtimeout(stdscr, 1);
    noecho();
    // mvprintw(0, 0, "Use arrow keys to go up and down, Press enter to select a choice");
    refresh();

    // mouse events
    init_mouse();
    move(0,0);

    start_color();
    
    init_grayscale(num_grayscale);
    std::locale::global(std::locale("en_US.utf8"));
    std::wcout.imbue(std::locale());
    clear_background();
}

static int curses_width()
{
    return getmaxx(stdscr);
}

static int curses_height()
{
    return getmaxy(stdscr);
}


#endif // CURSES_HELPER_HPP