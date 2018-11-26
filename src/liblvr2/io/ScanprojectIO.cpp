/**
 * Copyright (c) 2018, University Osnabrück
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

#include <cstdio>
#include <iomanip>

#include <boost/filesystem/fstream.hpp>

#include <lvr2/io/ScanprojectIO.hpp>
#include <lvr2/io/UosIO.hpp>


namespace lvr2
{

Scanproject& ScanprojectIO::get_project()
{
    return project;
}

bool ScanprojectIO::exists_and_is_dir(const fs::path &dir, bool silent)
{
    try
    {
        if (fs::exists(dir) && fs::is_directory(dir))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    catch (fs::filesystem_error &e)
    {
        if (!silent)
        {
            std::cout << "[ScanprojectIO] Error: " << e.what() << std::endl;
        }

        return false;
    }
}

bool ScanprojectIO::parse_project(const std::string &dir, bool silent)
{
    project_dir = fs::path(dir);

    if (!exists_and_is_dir(project_dir, silent))
    {
        if (!silent)
        {
            std::cout << "[ScanprojectIO] Error: " << project_dir
                << " doesn't exist or isn't a directory." << std::endl;
        }

        return false;
    }

    project.img_dir   = project_dir / "images";
    project.scans_dir = project_dir / "scans";

    // check if subdirs exists
    if (!exists_and_is_dir(project.img_dir, silent))
    {
        if (!silent)
        {
            std::cout << "[ScanprojectIO] Error: " << project.img_dir
                << " doesn't exist or isn't a directory." << std::endl;
        }

        return false;
    }

    if (!exists_and_is_dir(project.scans_dir, silent))
    {
        if (!silent)
        {
            std::cout << "[ScanprojectIO] Error: " << project.scans_dir 
                << " doesn't exist or isn't a directory." << std::endl;
        }

        return false;
    }

    char buffer[1024];
    unsigned int cur_scan_nr = 1;
    while (true)
    {
        ScanPosition pos;
        std::snprintf(buffer, 1024, "scan%.3u", cur_scan_nr);
        pos.scan_file = project.scans_dir / (buffer + std::string(".3d"));

        if (!fs::exists(pos.scan_file))
        {
            break;
        }

        // is it really sane to use no transformation if .frames file wasn't found?
        pos.transform = Matrix4<Vec>();

        fs::path frames_file(project.scans_dir / (buffer + std::string(".frames")));
        if (!fs::exists(frames_file) || fs::is_empty(frames_file))
        {
            if (!silent)
            {
                std::cout << "[ScanprojectIO] Warning: Frames file " << frames_file
                    << " doesn't exists or is empty. Therefore no transformation is set \
                    for scan position " << cur_scan_nr << "." << std::endl;
            }

            cur_scan_nr++;
            continue;
        }
        else
        {
            std::ifstream frames_fstream(frames_file.string());
            if (!frames_fstream.is_open() || !frames_fstream.good())
            {
                frames_fstream.close();

                if (!silent)
                {
                std::cout << "[ScanprojectIO] Warning: Unable to read frames file "
                    << frames_file << ". Therefore no transformation is set for \
                    scan position " << cur_scan_nr << "." << std::endl;
                }

                cur_scan_nr++;
                continue;
            }

            pos.transform = UosIO().parseFrameFile(frames_fstream);

            frames_fstream.close();
        }

        unsigned int cur_img_nr = 1;
        while (true)
        {
            std::snprintf(buffer, 1024, "scan%.3u_%.2u", cur_scan_nr, cur_img_nr++);
            ImageFile img;
            fs::path img_file = project.img_dir / (std::string(buffer) + ".jpg");
            if (!exists(img_file))
            {
                break;
            }

            img.image_file = img_file;

            fs::path extrinsic_file(project.img_dir / (std::string(buffer) + "_extrinsic.dat"));
            if (!fs::exists(extrinsic_file))
            {
                // TODO inform over that skipping image...
                std::cout << extrinsic_file << std::endl;
                continue;
            }
            img.extrinsic_transform.loadFromFile(extrinsic_file.string());

            fs::path orientation_file(project.img_dir / (std::string(buffer) + "_orientation.dat"));
            if (!fs::exists(orientation_file))
            {
                // TODO inform over that skipping image...
                std::cout << orientation_file << std::endl;
                continue;
            }
            img.orientation_transform.loadFromFile(orientation_file.string());

            fs::path intrinsic_file(project.img_dir / (std::string(buffer) + "_intrinsic.txt"));
            if (!fs::exists(intrinsic_file))
            {
                // TODO inform over that skipping image...
                std::cout << intrinsic_file << std::endl;
                continue;
            }
            load_params_from_file(img.intrinsic_params, intrinsic_file, 4);

            fs::path distortion_file(project.img_dir / (std::string(buffer) + "_distortion.txt"));
            if (!fs::exists(distortion_file))
            {
                // TODO inform over that skipping image...
                std::cout << distortion_file << std::endl;
                continue;
            }

            load_params_from_file(img.distortion_params, distortion_file, 6);

            pos.images.push_back(img);
        }

        project.scans.push_back(pos);

        cur_scan_nr++;
    }

    if (project.scans.empty())
    {
        return false;
    }
    else 
    {
        return true;
    }
}

template<typename ValueType>
bool ScanprojectIO::load_params_from_file(ValueType *buf, const fs::path &src, unsigned int count)
{

    fs::fstream in_stream(src);
    if (!in_stream.is_open())
    {
        in_stream.close();
        return false;
    }

    unsigned int i;
    for (i = 0; i < count && in_stream.good(); i++)
    {
        in_stream >> buf[i];
    }

    if (i < count)
    {
        return false;
    }

    in_stream.close();

    return true;
}

ModelPtr ScanprojectIO::read(std::string dir)
{

    if (!parse_project(dir))
    {
        std::cout << "[ScanprojectIO] Error: The path " << dir
            << " isn't a proper Scanproject. returning an empty model." << std::endl;

        return ModelPtr();
    }

    return UosIO().read(project.scans_dir.string());
}

void ScanprojectIO::save(std::string dir)
{
    std::cout << "[ScanprojectIO]: Error: Saving from ScanProjects \
        currently not supported." << std::endl;
}

} // namespace lvr2 
