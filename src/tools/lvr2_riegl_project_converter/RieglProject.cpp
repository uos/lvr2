#include "RieglProject.hpp"

#include <algorithm>
#include <iostream>
#include <regex>

#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/BaseVector.hpp>

namespace lvr2
{

using Vec = BaseVector<float>;

// @TODO more errorprone against invalid input?
//       And this doesn't belong in here, it should be in Matrix4.
Matrix4<Vec> string2mat4f(const std::string data) {
    Matrix4<Vec> ret;

    int count = 0;
    std::string::size_type n = 0;
    std::string::size_type sz = 0;
    while (count < 16) {
        ret[count++] = std::stof(data.substr(n), &sz);
        n += sz;
        n = data.find_first_not_of(' ', n);
    }

    return ret;
}

std::string get_first_group_regex(std::regex regex_string, std::string data) {
    std::regex regex(regex_string);
    std::smatch sm;
    std::regex_match(data, sm, regex);

    return sm.str(1);
}

RieglProject::RieglProject(std::string dir) {
    m_project_dir    = fs::path(dir);
}

void RieglProject::parse_scanpositions(pt::ptree project_ptree, unsigned int start, unsigned int end) {
    for (auto scanpos_info : project_ptree.get_child("project.scanpositions")) {
        if (scanpos_info.first != "scanposition") continue;

        ScanPosition scanpos;

        std::string scan_name = scanpos_info.second.get<std::string>("name");

        if (!scan_name.length() == 10) {
            std::cout << "[RieglProject] Warning: The scanpos " << scan_name << " is skipped"
                      << " because the name is malformed." << std::endl;

            continue;
        }

        unsigned int scan_pos_nr = std::stoul(scan_name.substr(7));

        // skip scanpositions that aren't in range
        if (scan_pos_nr < start || (end != 0 && scan_pos_nr > end)) {
            continue;
        }

        fs::path scan_dir = m_project_dir / ("SCANS/" + scan_name + "/");

        unsigned int scan_file_size = 0;

        // find the .rxp file for scanpos

        // @TODO What should happen in this case if it is possible?
        //       Currently we take the biggest file but maybe we should merge all scan files.
        for (auto scan_data : scanpos_info.second.get_child("singlescans")) {
            if (scan_data.first != "scan") continue;

            std::string current_scan_filename = scan_data.second.get<std::string>("file");

            // we don't want .mon files
            if (current_scan_filename.find(".mon") != std::string::npos) {
                continue;
            }

            fs::path current_scan_file = scan_dir / ("SINGLESCANS/" + current_scan_filename);

            if (!fs::exists(current_scan_file) || !fs::is_regular_file(current_scan_file)) {
                std::cout << "[RieglProject] Warning: The scan data file '" << current_scan_file
                          << "' doesn't exists or isn't a regular file and there \
                              is gonna be skipped." << std::endl;

                continue;
            }

            unsigned int current_scan_file_size = fs::file_size(current_scan_file);

            if ( current_scan_file_size > scan_file_size) {
                scanpos.scan_file = current_scan_file;
                scan_file_size = current_scan_file_size;
            }
        }

        // @TODO Maybe it should be possible to have scanpositions without scan data but
        //       with images. Then we would have to change this behaviour.
        if (scanpos.scan_file.empty()) {
            std::cout << "[RieglProject] Warning: The scanposition '" << scan_dir
                      << "' has no scan data and therefore we gonna skip this scanposition."
                      << std::endl;

            continue;
        }

        // parse scanpos transformation
        std::string transform = scanpos_info.second.get<std::string>("sop.matrix");
        scanpos.transform = string2mat4f(transform);

        parse_images_per_scanpos(scanpos, scanpos_info.second, project_ptree);

        // @TODO maybe we shouldn't skip scanpositions only because they have no images.
        if (scanpos.images.empty()) {
            std::cout << "[RieglProject] Warning: Scanposition '" << scan_dir
                      << "' has no images and will be skipped." << std::endl;

            continue;
        }

        m_scan_positions.push_back(scanpos);
    }
}

void RieglProject::parse_images_per_scanpos(ScanPosition &scanpos,
                              pt::ptree scanpos_ptree,
                              pt::ptree project_ptree) {

    for (auto img_info : scanpos_ptree.get_child("scanposimages")) {
        if (img_info.first != "scanposimage") continue;

        ImageFile img;

        std::string img_file = img_info.second.get<std::string>("file");
        img.image_file = scanpos.scan_file.parent_path().parent_path() / ("SCANPOSIMAGES/" + img_file);


        if (!fs::exists(img.image_file) && !fs::is_regular_file(img.image_file)) {
            std::cout << "[RieglProject] Warning: Image file '" << img.image_file
                      << "' doesn't exists or is not a regular file and there is skipped."
                      << std::endl;

            continue;
        }



        // get orientation of image
        img.orientation_transform = string2mat4f(img_info.second.get<std::string>("cop.matrix"));



        // @TODO refactor mountcalib and camcalib reference search into one...

        //get extrinsic transformation for image
        std::string mountcalib_ref = img_info.second.get<std::string>("mountcalib_ref.<xmlattr>.noderef");
        mountcalib_ref = mountcalib_ref.substr(mountcalib_ref.find_last_of('/') + 1);

        bool found_mountcalib = false;
        for (auto mountcalib_info : project_ptree.get_child("project.calibrations.mountcalibs")) {
            if (mountcalib_info.first != "mountcalib") continue;
            if (mountcalib_info.second.get<std::string>("<xmlattr>.name") != mountcalib_ref) continue;

            found_mountcalib = true;

            img.extrinsic_transform = string2mat4f(mountcalib_info.second.get<std::string>("matrix"));
        }

        // skip image if no calibration data was found...
        if (!found_mountcalib) {
            std::cout << "[RieglProject] Warning: Extrinsic transformation for image file '"
                      << img.image_file << "' wasn't found and image is gonna be skipped."
                      << std::endl;

            continue;
        }



        //get intrinsic params for image
        std::string camcalib_ref = img_info.second.get<std::string>("camcalib_ref.<xmlattr>.noderef");
        camcalib_ref = camcalib_ref.substr(camcalib_ref.find_last_of('/') + 1);

        bool found_camcalib = false;
        for (auto camcalib_info : project_ptree.get_child("project.calibrations.camcalibs")) {
            if (camcalib_info.first != "camcalib_opencv") continue;
            if (camcalib_info.second.get<std::string>("<xmlattr>.name") != camcalib_ref) continue;

            found_camcalib = true;

            pt::ptree intrinsic_ptree = camcalib_info.second.get_child("internal_opencv");

            img.intrinsic_params[0] = intrinsic_ptree.get<float>("fx");
            img.intrinsic_params[1] = intrinsic_ptree.get<float>("fy");
            img.intrinsic_params[2] = intrinsic_ptree.get<float>("cx");
            img.intrinsic_params[3] = intrinsic_ptree.get<float>("cy");

            img.distortion_params[0] = intrinsic_ptree.get<float>("k1");
            img.distortion_params[1] = intrinsic_ptree.get<float>("k2");
            img.distortion_params[2] = intrinsic_ptree.get<float>("k3");
            img.distortion_params[3] = intrinsic_ptree.get<float>("k4");
            img.distortion_params[4] = intrinsic_ptree.get<float>("p1");
            img.distortion_params[5] = intrinsic_ptree.get<float>("p2");
        }

        // skip image if no calibration data was found...
        if (!found_camcalib) {
            std::cout << "[RieglProject] Warning: Camcalibration for image file '" << img.image_file
                      << "' wasn't found and image is gonna be skipped." << std::endl;

            continue;
        }

        scanpos.images.push_back(img);
    }
}

bool RieglProject::parse_project(unsigned int start, unsigned int end) {
    // check if project path exists
    if (!fs::exists(m_project_dir) || !fs::is_directory(m_project_dir)) {
        std::cout << "[RieglProject] Error: RiSCAN project path '" << m_project_dir
                  << "' doesn't exist or isn't a directory" << std::endl;

        return false;
    }

    // check if project.rsp exists
    fs::path project_file = m_project_dir / "project.rsp";
    if (!fs::exists(project_file) || !fs::is_regular_file(project_file)) {
        std::cout << "[RieglProject] Error: The RiSCAN project file '" << project_file
                  << "' doesn't exist or isn't a file." << std::endl;

        return false;
    }

    // read project.rsp file
    pt::ptree project_ptree;
    pt::read_xml(project_file.string(), project_ptree);

    parse_scanpositions(project_ptree, start, end);

    if (m_scan_positions.empty()) {
        std::cout << "[RieglProject] Error: Unable to parse any scanposition." << std::endl;

        return false;
    }

    return true;
}

std::ostream& operator<<(std::ostream &lhs, const RieglProject &rhs) {
    lhs << "Scan Project dir: " << rhs.m_project_dir << "\n";

    for (const lvr2::ScanPosition &sp : rhs.m_scan_positions) {
        lhs << "\n" << sp;
    }

    return lhs;
}

std::ostream& operator<<(std::ostream &lhs, const ScanPosition &rhs) {
    lhs << "Scan File: " << rhs.scan_file << " "
        << fs::file_size(rhs.scan_file) << "\n"
        << rhs.transform
        << "Images: ";

    for (ImageFile img : rhs.images) {
        lhs << "\n\t" << img.image_file.filename() << " " << fs::file_size(img.image_file) << '\n'
            << "Orientation: " << img.orientation_transform << '\n'
            << "Extrinsic: " << img.extrinsic_transform << '\n';

        lhs << "Intrinsic: ";
        for (int i = 0; i < 4; i++) {
            lhs << img.intrinsic_params[i] << " ";
        }

        lhs << '\n' << "Distortion: ";
        for (int i = 0; i < 6; i++) {
            lhs << img.distortion_params[i] << " ";
        }
    }

    lhs << std::endl;

    return lhs;
}

} // namespace lvr2
