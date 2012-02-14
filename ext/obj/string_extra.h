#ifndef STRING_EXTRA_H
#define STRING_EXTRA_H

#include <string>

using std::string;

char strequal(const char *s1, const char *s2);
char contains(const char *haystack, const char *needle);
string getPathOf(const char *filename);

#endif
