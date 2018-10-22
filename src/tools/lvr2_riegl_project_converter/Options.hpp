#include <string>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

class Options {

    private:
        po::variables_map vm;
        po::options_description desc;
        po::positional_options_description pod;

    public:
        Options();

        bool parse(int argc, char **argv);

        std::string getOutputDir() {
            return vm["out"].as<std::string>();
        }

        std::string getInputDir() {
            return vm["input"].as<std::string>();
        }

        bool force_overwrite() {
            return vm.count("force") != 0;
        }

        unsigned int getEndscan() {
            return vm["end"].as<unsigned int>();
        }

        unsigned int getStartscan() {
            return vm["start"].as<unsigned int>();
        }

        unsigned int getReductionFactor() {
            return vm["reduction"].as<unsigned int>();
        }
};
