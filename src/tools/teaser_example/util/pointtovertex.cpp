//
// Created by praktikum on 15.09.22.
//
#include "pointtovertex.h"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>


using namespace std;
//reads original ply files with point in header
//and creates new .ply with vertex header


//replaces the first instance of toReplace with replaceWith in given string
void replace_first(
    std::string &s,
    std::string const &toReplace,
    std::string const &replaceWith
) {
std::size_t pos = s.find(toReplace);
if (pos == std::string::npos) return;
s.replace(pos, toReplace.length(), replaceWith);
}

//saves vertex format in destination path
int pointtovertex(string src, string dest) {

    if(!std::filesystem::exists(dest)) {
        string strReplace = "point";
        string strNew = "vertex";
        ifstream filein(src); //File to read from
        ofstream fileout(dest); //File to write to

        //case file error
        if (!filein || !fileout) {
            cout << "Error opening files!" << endl;
            return 1;
        }


        string strTemp;
        bool found = false;
        std::string line;
        size_t i = 0;
        while (getline(filein, line)) {
            if (i < 10) {
                replace_first(line, "point", "vertex");
            }
            fileout << line << "\n";
            i++;


        }
    }
    return 0;


}
