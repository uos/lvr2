#ifndef SPDMON_VERSION_H_
#define SPDMON_VERSION_H_

#define SPDMON_VERSION "1.0.0"

#define SPDMON_MAJOR_VERSION 1
#define SPDMON_MINOR_VERSION 0
#define SPDMON_PATCH_VERSION 0

#include <iostream>

void PrintAppVersion()
{
    std::cout << "SPDMON_VERSION: ";    
    std::cout.width(16); 
    std::cout << std::right << SPDMON_VERSION << std::endl; 
}

#endif  // SPDMON_VERSION_H_

