
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <boost/program_options.hpp>
#include <boost/iostreams/device/mapped_file.hpp>




using namespace std;
size_t countLines(string filename);

int main (int argc , char *argv[]) {
    // get all options
    boost::program_options::options_description desc("Allowed options");
    desc.add_options() ("file", boost::program_options::value<string>()->default_value("noinput"), "Inputfile");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    std::string filePath = vm["file"].as<string>();

    ifstream ifs;
    ifs.open(filePath);
    unsigned long long int line_size = countLines(filePath);



    boost::iostreams::mapped_file_params mfp;
    mfp.new_file_size = 3 * sizeof(float) * line_size;
    mfp.path = "asd.raw";

    boost::iostreams::mapped_file_sink mmf(mfp);
    float * data = (float *)mmf.data();

    for(int i = 0 ; i<line_size*3 ;i+=3)
    {
        ifs >> data[i] >> data[i+1] >> data[i+2];
    }
    mmf.close();
    ifs.close();
}

size_t countLines(string filename)
{
    // Open file for reading
    ifstream in(filename.c_str());

    // Count lines in file
    size_t c = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        c++;
    }
    in.close();
    return c;
}

