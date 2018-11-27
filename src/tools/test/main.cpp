#include <iostream>

#include <lvr2/io/ScanDataManager.hpp>

int main(int argc, char **args)
{
    std::cout << "Hello World" << std::endl;

    lvr2::ScanDataManager man(args[1]);
    std::vector<lvr2::ScanData>& bla = man.getScanData();

    for (auto x : bla)
        std::cout << (x.m_points ? "points" : "no points") << std::endl;

    man.loadPointCloudData(1);

    for (auto x : bla)
        std::cout << (x.m_points ? "points" : "no points") << std::endl;

    std::cout << bla.size() << std::endl;

    return 0;
}
