//
// Created by eiseck on 01.04.16.
//
#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
using namespace std;

int main(int argc, char* argv[])
{

    
    vector<string> files;
    for(int i = 1 ; i<argc ; i++)
    {
        files.push_back(string(argv[i]));
        cout << "using: " << argv[i] << endl;
    }
    ofstream output("out.xyz");

    size_t numFiles = files.size();

    omp_lock_t lock;
    omp_init_lock(&lock);
    cout << "got " << numFiles << " scan files" << endl;
    //#pragma omp parallel for
    for(int i = 0 ; i<numFiles ; i++)
    {
	cout << "finished: " << ((float)i/(float)numFiles)*100.0f << "%" << endl;
        string path = files[i];
        string matPath = path;
        boost::algorithm::replace_last(matPath, "txt", "dat");
        cout << "getting t mat from: " << matPath << endl;
	ifstream matifs(matPath);
        Eigen::Matrix4d transformation;
        double matvalues[16];
        string line_s;
        string line_s_2;
        cout << "reading mat file" << endl;
        while( getline( matifs, line_s ) )
        {
          cout << line_s << endl;
          line_s_2 = line_s;
        }
      cout << "fin, last line:" << endl << line_s_2<< endl;

        std::istringstream mat_ss (line_s_2);
        for(int i = 0 ; i<16;i++)  mat_ss >> matvalues[i];
        transformation  << matvalues[0], matvalues[1], matvalues[2], matvalues[3],
                         matvalues[4], matvalues[5], matvalues[6], matvalues[7],
                        matvalues[8], matvalues[9], matvalues[10], matvalues[11],
                        matvalues[12], matvalues[13], matvalues[14], matvalues[15];



	cout << "Transformmatrix: " << transformation << endl;
        ifstream inputData(path);
        string s;
        int j = 0;
	cout << "opening: " << path << endl;
        unsigned long maxpp = 0;
	while( getline( inputData, s ) )
        {
            if(j==0)
            {
		j++;
            }
            else
            {
                stringstream ss;
                ss.str(s);
                double x,y,z;
                ss >> x >> y >> z;
                Eigen::Vector4d v(x,y,z,1);
		


                Eigen::Vector4d tv = transformation*v;

      //          omp_set_lock(&lock);
		//cout << "######" << endl <<  tv << endl << "------"  << endl;
                output << tv[0] << " " << tv[1] << " " << tv[2] << endl;
      //          omp_unset_lock(&lock);
            }

        }


    }
    return 0;
}
