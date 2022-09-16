//
// Created by praktikum on 15.09.22.
//

#ifndef LAS_VEGAS_POINTTOVERTEX_H
#define LAS_VEGAS_POINTTOVERTEX_H

#include <iostream>
#include <fstream>
#include <string>


//reads original ply files with point in header
//and creates new .ply with vertex header


//replaces the first instance of toReplace with replaceWith in given string
void replace_first(
        std::string &s,
        std::string const &toReplace,
        std::string const &replaceWith
);
//saves vertex format in destination path
int pointtovertex(std::string src, std::string dest);

#endif //LAS_VEGAS_POINTTOVERTEX_H
