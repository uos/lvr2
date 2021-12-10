#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <boost/optional/optional_io.hpp>

#include <lvr2/io/scanio/MeshSchema.hpp>
#include <lvr2/io/scanio/MeshSchemaDirectory.hpp>
#include <lvr2/io/scanio/MeshSchemaHDF5.hpp>
#include <lvr2/util/Timestamp.hpp>


using namespace lvr2;

bool verify_description(const Description& d, const Description& res)
{
    if (!(d.dataRoot == res.dataRoot))
    {
        std::cerr << timestamp << "Data root mismatch: " << d.dataRoot << " != " << res.dataRoot << std::endl;
        return false;
    }

    if (!(d.data == res.data))
    {
        std::cerr << timestamp << "Data mismatch: " << d.data << " != " << res.data << std::endl;
        return false;
    }

    if (!(d.metaRoot == res.metaRoot))
    {
        std::cerr << timestamp << "metaRoot mismatch: " << d.metaRoot << " != " << res.metaRoot << std::endl;
        return false;
    }

    if (!(d.meta == res.meta))
    {
        std::cerr << timestamp << "meta mismatch: " << d.meta << " != " << res.meta << std::endl;
        return false;
    }

    return true;
}

// Test the Mesh directory schema
void test_mesh_schema_directory(size_t& success_count, size_t& failure_count)
{
    std::cout << timestamp << "Testing Mesh directory schema" << std::endl;

    auto schema = std::make_shared<MeshSchemaDirectory>();

    auto mesh_name          = std::string("mesh1");
    auto channel_name       = std::string("coordinates");
    auto layer_name         = std::string("rgb");
    size_t surface_index    = 0;
    size_t material_index   = 0;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << surface_index;
    auto surface_index_string = sstr.str();

    std::stringstream sstr2;
    sstr2 << std::setfill('0') << std::setw(8) << material_index;
    auto material_index_string = sstr2.str();


    // Test if vertexChannel is correct
    Description d = schema->vertexChannel(mesh_name, channel_name);
    Description expected_res;

    // Expected result
    expected_res.dataRoot   = "meshes/" + mesh_name + "/vertices";
    expected_res.data       = "coordinates.ply";
    expected_res.metaRoot   = "meshes/" + mesh_name + "/vertices";
    expected_res.meta       = "coordinates.yaml";
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;


    // Test if durfaceIndices are correct
    d = schema->surfaceIndices(mesh_name, surface_index);
    
    expected_res.dataRoot   = "meshes/" + mesh_name + "/surfaces/" + surface_index_string;
    expected_res.data       = "indices";
    expected_res.metaRoot   = "meshes/" + mesh_name + "/surfaces/" + surface_index_string;
    expected_res.meta       = "indices.yaml";
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;
    

    // Test if texureCoordinates are correct
    d = schema->textureCoordinates(mesh_name, surface_index);
    
    expected_res.dataRoot   = "meshes/" + mesh_name + "/surfaces/" + surface_index_string;
    expected_res.data       = "texture_coordinates";
    expected_res.metaRoot   = "meshes/" + mesh_name + "/surfaces/" + surface_index_string;
    expected_res.meta       = "texture_coordinates.yaml";
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;

    // Test if texture are correct
    d = schema->texture(mesh_name, material_index, layer_name);

    expected_res.dataRoot   = "meshes/" + mesh_name + "/materials/" + material_index_string + "/textures";
    expected_res.data       = layer_name + ".jpg";
    expected_res.metaRoot   = expected_res.dataRoot;
    expected_res.meta       = layer_name + ".yaml";
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;
}

void test_mesh_schema_hdf5(size_t& success_count, size_t& failure_count)
{
    std::cout << timestamp << "Testing Mesh HDF5 schema" << std::endl;

    auto schema = std::make_shared<MeshSchemaHDF5>();

    auto mesh_name          = std::string("mesh1");
    auto channel_name       = std::string("coordinates");
    auto layer_name         = std::string("rgb");
    size_t surface_index    = 0;
    size_t material_index   = 0;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << surface_index;
    auto surface_index_string = sstr.str();

    std::stringstream sstr2;
    sstr2 << std::setfill('0') << std::setw(8) << material_index;
    auto material_index_string = sstr2.str();


    // Test if vertexChannel is correct
    Description d = schema->vertexChannel(mesh_name, channel_name);
    Description expected_res;

    expected_res.dataRoot   = "meshes/" + mesh_name + "/vertices";
    expected_res.data       = "coordinates";
    expected_res.metaRoot   = expected_res.dataRoot;
    expected_res.meta       = expected_res.data;
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;


    // Test if durfaceIndices are correct
    d = schema->surfaceIndices(mesh_name, surface_index);
    
    expected_res.dataRoot   = "meshes/" + mesh_name + "/surfaces/" + surface_index_string;
    expected_res.data       = "indices";
    expected_res.metaRoot   = expected_res.dataRoot;
    expected_res.meta       = expected_res.data;
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;
    

    // Test if texureCoordinates are correct
    d = schema->textureCoordinates(mesh_name, surface_index);
    
    expected_res.dataRoot   = "meshes/" + mesh_name + "/surfaces/" + surface_index_string;
    expected_res.data       = "texture_coordinates";
    expected_res.metaRoot   = expected_res.dataRoot;
    expected_res.meta       = expected_res.data;
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;

    // Test if texture are correct
    d = schema->texture(mesh_name, material_index, layer_name);

    expected_res.dataRoot   = "meshes/" + mesh_name + "/materials/" + material_index_string + "/textures";
    expected_res.data       = layer_name;
    expected_res.metaRoot   = expected_res.dataRoot;
    expected_res.meta       = expected_res.data;
    // Test if the result is correct
    verify_description(d, expected_res) ? success_count++ : failure_count++;
}

int main()
{
    size_t success_count = 0;
    size_t failure_count = 0;

    ///////////
    // Tests //
    ///////////
    test_mesh_schema_directory(success_count, failure_count);
    test_mesh_schema_hdf5(success_count, failure_count);
    
    // Print summary
    if (failure_count) std::cout << timestamp << "[" << failure_count << "/" << success_count + failure_count << "] tests failed" << std::endl;
    std::cout << timestamp << "[" << success_count << "/" << success_count + failure_count << "] tests passed" << std::endl;

    if (failure_count)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}