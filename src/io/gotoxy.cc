#include "gotoxy.h"

/*
** gotoxy() Implementation in the *nix environment
**
** Note: I've heard the curses library can be useful 
** when trying to implement the handy DOS-only tools of 
** gotoxy() and clrscr() using move() and initscr().
** Though, there is a way to write your own gotoxy() 
** in the Linux environment. This topic isn't discussed 
** often, so I'd like to bring a few new ideas to the table.
**
** Concept: We will use two ANSI C standard functions 
** to accomplish our task. We will use specific 
** string manuvers, according to the man pages, that 
** will allow us to execute each part of the program.
*/
void gotoxy(int x, int y) {
  char essq[100];		/* String variable to hold the escape sequence */
  char xstr[100];		/* Strings to hold the x and y coordinates */
  char ystr[100];		/* Escape sequences must be built with characters */

  /*
  ** Convert the screen coordinates to strings
  */
  sprintf(xstr, "%d", x);
  sprintf(ystr, "%d", y);

  /*
  ** Build the escape sequence (vertical move)
  */
  essq[0] = '\0';
  strcat(essq, "\033[");
  strcat(essq, ystr);

  /*
  ** Described in man terminfo as vpa=\E[%p1%dd
  ** Vertical position absolute
  */
  strcat(essq, "d");

  /*
  ** Horizontal move
  ** Horizontal position absolute
  */
  strcat(essq, "\033[");
  strcat(essq, xstr);
  /* Described in man terminfo as hpa=\E[%p1%dG */
  strcat(essq, "G");

  /*
  ** Execute the escape sequence
  ** This will move the cursor to x, y
  */
  printf("%s", essq);
}
