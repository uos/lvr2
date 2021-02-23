#ifndef METAFILEFACTORY_HPP
#define METAFILEFACTORY_HPP

#include "lvr2/io/Timestamp.hpp"

#include <yaml-cpp/yaml.h>
#include "lvr2/io/scanio/yaml/Matrix.hpp"

// TODO: remove this dep
#include "lvr2/util/IOUtils.hpp"

// TODO:  remove this dep
#include "lvr2/util/TransformUtils.hpp"


#include <boost/filesystem.hpp>
#include <fstream>

namespace lvr2
{

bool isMetaFile(const std::string& filename);

void saveMetaInformation(const std::string &outfile, const YAML::Node &node);
YAML::Node loadMetaInformation(const std::string &in);

} // namespace lvr2

#endif