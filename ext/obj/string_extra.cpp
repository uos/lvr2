#include "string_extra.h"
#include <string.h>

char strequal(const char *s1, const char *s2)
{
  if(strcmp(s1, s2) == 0)
    {
      return 1;
    }
  return 0;
}

char contains(const char *haystack, const char *needle)
{
  if(strstr(haystack, needle) == NULL)
    {
      return 0;
    }
  return 1;
}

string getPathOf(const char *filename)
{
  if(contains(filename, "/"))
    {
      string fn(filename);
      return fn.substr(0, fn.find_last_of("/"));
    }
  return "";
}
