//
// Created by Gruppe teaser on 15.09.22.
//

#ifndef LAS_VEGAS_POINTTOVERTEX_H
#define LAS_VEGAS_POINTTOVERTEX_H

#include <iostream>
#include <fstream>
#include <string>


/**
    * @brief replaces a word in a string
    *
    * @param s, string to replace something in
    * @param toReplace
    * @param replaceWith
    */
void replace_first(
        std::string &s,
        std::string const &toReplace,
        std::string const &replaceWith
);
/**
    * @brief reads original ply files with point in header and creates new .ply with vertex in header
    *
    * @param src, path to ply
    * @param dest, path to new ply
    * @return status int
    */
int pointtovertex(std::string src, std::string dest);

#endif //LAS_VEGAS_POINTTOVERTEX_H
