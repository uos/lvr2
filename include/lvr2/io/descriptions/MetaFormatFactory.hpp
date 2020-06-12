#ifndef METAFILEFACTORY_HPP
#define METAFILEFACTORY_HPP

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <fstream>

namespace lvr2
{

void saveMetaInformation(const std::string &outfile, const YAML::Node &node);
YAML::Node loadMetaInformation(const std::string &in);

} // namespace lvr2

#endif