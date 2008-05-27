#include "show.h"

int main(int argc, char** argv){
 
  if(argc == 2){
    MCShow show(argv[1]);
  } else if (argc == 3){
  } else {
    cout << "usage: show <file> [file2]" << endl;
  }
}
