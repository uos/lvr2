set(NEWMAT_SOURCES
newmat1.cpp 
newmat2.cpp 
newmat3.cpp 
newmat4.cpp 
newmat5.cpp 
newmat6.cpp 
newmat7.cpp 
newmat8.cpp 
newmatex.cpp 
bandmat.cpp 
submat.cpp 
myexcept.cpp 
cholesky.cpp 
evalue.cpp 
fft.cpp 
hholder.cpp 
jacobi.cpp 
newfft.cpp 
sort.cpp 
svd.cpp 
newmatrm.cpp 
newmat9.cpp)


add_library(newmat SHARED ${NEWMAT_SOURCES})
set_target_properties(newmat PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)