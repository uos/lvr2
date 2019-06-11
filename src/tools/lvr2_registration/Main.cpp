/**
 * Copyright (c) 2019, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Main.cpp
 *
 *  @date Apr 25, 2019
 *  @author Malte Hillmann
 */

// external includes
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <chrono>

// internal includes
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/io/IOUtils.hpp>
#include <lvr2/registration/SlamAlign.hpp>

using namespace lvr2;
using namespace std;
using boost::filesystem::path;

string format_name(const string& format, int index)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), index) + 1; // Extra space for '\0'
    char buff[size];
    snprintf(buff, size, format.c_str(), index);
    return string(buff);
}

int main(int argc, char** argv)
{
    // =============== parse options ===============
    SlamOptions options;

    int start = -1;
    int end = -1;
    string format = "uos";
    path dir;
    bool help;

    try
    {
        using namespace boost::program_options;

        options_description visible_options("OPTIONS");
        visible_options.add_options()
        ("start,s", value<int>(&start)->default_value(start),
         "The first scan to process.\n"
         "-1 (default): search for first scan")

        ("end,e", value<int>(&end)->default_value(end),
         "The last scan to process.\n"
         "-1 (default): continue until no more scan found")

        ("format,f", value<string>(&format)->default_value(format),
         "The format to use.\n"
         "available formats are listed <somewhere>")

        ("icpIterations,i", value<int>(&options.icpIterations)->default_value(options.icpIterations),
         "Number of iterations for ICP")

        ("icpMaxDistance,d", value<double>(&options.icpMaxDistance)->default_value(options.icpMaxDistance),
         "The maximum distance between two points during ICP")

        ("slamIterations,I", value<int>(&options.slamIterations)->default_value(options.slamIterations),
         "Number of iterations for SLAM")

        ("slamMaxDistance,D", value<double>(&options.slamMaxDistance)->default_value(options.slamMaxDistance),
         "The maximum distance between two points during SLAM")

        ("reduction,r", value<double>(&options.reduction)->default_value(options.reduction),
         "The Voxel size for Voxel based reduction. -1 for no reduction")

        ("min,m", value<double>(&options.minDistance)->default_value(options.minDistance),
         "Ignore all Points closer than <value> to the origin of the scan.\n"
         "-1 (default): No filter")

        ("max,M", value<double>(&options.maxDistance)->default_value(options.maxDistance),
         "Ignore all Points farther away than <value> from the origin of the scan.\n"
         "-1 (default): No filter")

        ("epsilon", value<double>(&options.epsilon)->default_value(options.epsilon),
         "The desired epsilon difference between two error values")

        ("closeLoopDistance,c", value<double>(&options.closeLoopDistance)->default_value(options.closeLoopDistance),
         "The maximum distance between two poses to consider a closed loop")

        ("closeLoopPairs,C", value<int>(&options.closeLoopPairs)->default_value(options.closeLoopPairs),
         "The minimum pair overlap between two poses to consider a closed loop. Pairs are judged using slamMaxDistance.\n"
         "-1 (default): use closeLoopDistance instead")

        ("loop,L", bool_switch(&options.doLoopClosing),
         "Use simple Loop Closing")

        ("graph,G", bool_switch(&options.doGraphSlam),
         "Use complex Loop Closing with GraphSLAM")

        ("trustPose,p", bool_switch(&options.trustPose),
         "Use the unmodified Pose for ICP. Useful for GPS Poses")

        ("metascan", bool_switch(&options.metascan),
         "Match scans to the combined pointcloud of all previous scans")

        ("verbose,v", bool_switch(&options.verbose),
         "Show more detailed output")

        ("help,h", bool_switch(&help),
         "Print this help")
        ;

        options_description hidden_options("hidden_options");
        hidden_options.add_options()
        ("dir", value<path>(&dir))
        ;

        positional_options_description pos;
        pos.add("dir", 1);

        options_description options("options");
        options.add(visible_options).add(hidden_options);

        variables_map variables;
        store(command_line_parser(argc, argv).options(options).positional(pos).run(), variables);
        notify(variables);

        if (help)
        {
            cout << "The Scan Registration Tool" << endl;
            cout << "Usage: " << endl;
            cout << "\tlvr2_registration [OPTIONS] <dir>" << endl;
            cout << endl;
            visible_options.print(cout);
            cout << endl;
            cout << "<dir> is the directory to search scans in" << endl;
            return EXIT_SUCCESS;
        }
        if (variables.count("dir") != 1)
        {
            throw error("Missing <dir> Parameter");
        }
    }
    catch (const boost::program_options::error& ex)
    {
        std::cerr << ex.what() << endl;
        std::cerr << endl;
        std::cerr << "Use '--help' to see the list of possible options" << endl;
        return EXIT_FAILURE;
    }

    // =============== parse format ===============
    size_t split_point;
    if ((split_point = format.find('%')) == string::npos)
    {
        // TODO: map formats
        format = "scan%03i.3d";
    }

    // =============== search scans ===============
    if (start == -1)
    {
        for (int i = 0; i < 100; i++)
        {
            path file = dir / format_name(format, i);
            if (exists(file))
            {
                start = i;
                cout << "First scan: " << file.filename() << endl;
                break;
            }
        }
        if (start == -1)
        {
            cerr << "Could not find a starting scan. are you using the right format?" << endl;
            return EXIT_FAILURE;
        }
    }

    // make sure all scan and pose files are in the directory
    for (int i = start; end == -1 || i <= end; i++)
    {
        path file = dir / format_name(format, i);
        if (!exists(file))
        {
            if (end != -1 || i == start)
            {
                cerr << "Missing scan " << file.filename() << endl;
                return EXIT_FAILURE;
            }
            end = i - 1;
            cout << "Last scan: \"" << format_name(format, end) << '"' << endl;
            break;
        }
        file.replace_extension("pose");
        if (!exists(file))
        {
            cerr << "Missing pose file " << file.filename() << endl;
            return EXIT_FAILURE;
        }
    }

    int count = end - start + 1;

    SlamAlign align(options, count);

    for (int i = 0; i < count; i++)
    {
        path file = dir / format_name(format, start + i);
        auto model = ModelFactory::readModel(file.string());

        file.replace_extension("pose");
        Matrix4d pose = getTransformationFromPose(file);

        auto points = model->m_pointCloud;

        align.setScan(i, make_shared<Scan>(model->m_pointCloud, pose));
    }

    auto start_time = chrono::steady_clock::now();

    align.match();

    auto required_time = chrono::steady_clock::now() - start_time;
    cout << "SLAM finished in " << required_time.count() / 1e9 << " seconds" << endl;

    for (int i = 0; i < count; i++)
    {
        path file = dir / format_name(format, start + i);
        file.replace_extension("frames");

        align.writeFrames(i, file.string());
    }

    return EXIT_SUCCESS;
}
