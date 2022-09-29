#include <boost/filesystem.hpp>
#include <iostream>
#include <bitset>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Options.hpp"

#include "CudaNormals.cuh"
#include "MortonCodes.cuh"

using namespace lvr2;

void floatToBinary(float f)
{
    char* bits = reinterpret_cast<char*>(&f);
    for(int n = sizeof(f) - 1; n >= 0; n--)
        std::cout << std::bitset<8>(bits[n]);
    std::cout << std::endl;
    
}

int main(int argc, char** argv)
{
    cuda_normals_2::Options opt(argc, argv);

    // Get the model
    ModelPtr model = ModelFactory::readModel(opt.inputFile());

    // Get the points
    PointBufferPtr pbuffer = model->m_pointCloud;
    size_t num_points = model->m_pointCloud->numPoints();
    size_t size =  3 * num_points;

    floatArr points = pbuffer->getPointArray();
    
    float* points_raw = &points[0];

    // Create the normal array
    floatArr normals(new float[size]);

    float* normals_raw = (float*) malloc(sizeof(float) * 3 * num_points);

    // Initialize the normals
    initNormals2(normals_raw, num_points);

    // Set the normal array
    pbuffer->setNormalArray(normals, num_points);

    unsigned long long int* mortonCodes = (unsigned long long int*)
                    malloc(sizeof(unsigned long long int) * num_points);


    float* test_points = (float*) malloc(sizeof(float) * 9);
    test_points[0] = 0.5;
    test_points[1] = 0.25;
    test_points[2] = 0.125;
    test_points[3] = 0.0625;
    test_points[4] = 0.03125;
    test_points[5] = 0.015625;
    test_points[6] = 1.0;
    test_points[7] = 1.0;
    test_points[8] = 1.0;

    // Get the morton codes of the 3D points
    getMortonCodes(mortonCodes, test_points, 3);
    /*
    std::cout << "x: " << points_raw[0] << std::endl;
    std::cout << "y: " << points_raw[1] << std::endl;
    std::cout << "z: " << points_raw[2] << std::endl;
    */
    std::cout << mortonCodes[0] << std::endl;
    /*
    std::cout << "Point 0 x: " << test_points[0] << std:: endl;
    std::cout << "Point 0 y: " << test_points[1] << std:: endl;
    std::cout << "Point 0 z: " << test_points[2] << std:: endl;

    std::cout << "Point 0 x binary: "; 
    floatToBinary(test_points[0]);
    std::cout <<std::endl;
    
    std::cout << "Point 0 y binary: "; 
    floatToBinary(test_points[1]);
    std::cout <<std::endl;
    
    std::cout << "Point 0 z binary: "; 
    floatToBinary(test_points[2]);
    std::cout <<std::endl;
                                             
    std::cout << "x binary: " << std::bitset<64>(4692150331805532160) << std::endl;
    std::cout << "y binary: " << std::bitset<64>(4687375547406221312) << std::endl;
    std::cout << "z binary: " << std::bitset<64>(4682742812607774720) << std::endl;
    */
    std::cout << ("x binary: ");
    floatToBinary(489335.468750);

    std::cout << ("y binary: ");
    floatToBinary(236775.218750);

    std::cout << ("z binary: ");
    floatToBinary(116508.445312);
    
    std::cout << "ULLI: " << std::bitset<64>(4692150331805532160) << std::endl;
    std::cout << "Morton Code: " << std::bitset<64>(mortonCodes[0]) << std::endl;

    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    free(normals_raw);
    free(mortonCodes);
    floatToBinary(1.0f);

    return 0;
}
