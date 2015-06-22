# Microsoft Developer Studio Project File - Name="LASlib" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=LASlib - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "LASlib.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "LASlib.mak" CFG="LASlib - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "LASlib - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "LASlib - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "LASlib - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /I "inc" /I "stl" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=copy Release\LASlib.lib lib\LASlib.lib
# End Special Build Tool

!ELSEIF  "$(CFG)" == "LASlib - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /I "inc" /I "stl" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=copy Debug\LASlib.lib libD\LASlib.lib
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "LASlib - Win32 Release"
# Name "LASlib - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\src\arithmeticdecoder.cpp
# End Source File
# Begin Source File

SOURCE=.\src\arithmeticencoder.cpp
# End Source File
# Begin Source File

SOURCE=.\src\arithmeticmodel.cpp
# End Source File
# Begin Source File

SOURCE=.\src\fopen_compressed.cpp
# End Source File
# Begin Source File

SOURCE=.\src\integercompressor.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasfilter.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasindex.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasinterval.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasquadtree.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreader.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreader_bin.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreader_las.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreader_qfit.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreader_shp.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreader_txt.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreadermerged.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreaditemcompressed_v1.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreaditemcompressed_v2.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreadpoint.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasspatial.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lastransform.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasunzipper.cpp
# End Source File
# Begin Source File

SOURCE=.\src\lasutility.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswaveform13reader.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswaveform13writer.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriteitemcompressed_v1.cpp

!IF  "$(CFG)" == "LASlib - Win32 Release"

!ELSEIF  "$(CFG)" == "LASlib - Win32 Debug"

# ADD CPP /I "..\..\src_full\stl"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\src\laswriteitemcompressed_v2.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswritepoint.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriter.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriter_bin.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriter_las.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriter_qfit.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriter_txt.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laszip.cpp
# End Source File
# Begin Source File

SOURCE=.\src\laszipper.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter ".h"
# Begin Source File

SOURCE=.\src\arithmeticdecoder.hpp
# End Source File
# Begin Source File

SOURCE=.\src\arithmeticencoder.hpp
# End Source File
# Begin Source File

SOURCE=.\src\arithmeticmodel.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\bytestreamin.hpp
# End Source File
# Begin Source File

SOURCE=.\src\bytestreamin_file.hpp
# End Source File
# Begin Source File

SOURCE=.\src\bytestreamin_istream.hpp
# End Source File
# Begin Source File

SOURCE=.\src\bytestreamout.hpp
# End Source File
# Begin Source File

SOURCE=.\src\bytestreamout_file.hpp
# End Source File
# Begin Source File

SOURCE=.\src\bytestreamout_nil.hpp
# End Source File
# Begin Source File

SOURCE=.\src\bytestreamout_ostream.hpp
# End Source File
# Begin Source File

SOURCE=.\src\entropydecoder.hpp
# End Source File
# Begin Source File

SOURCE=.\src\entropyencoder.hpp
# End Source File
# Begin Source File

SOURCE=.\src\integercompressor.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasdefinitions.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasfilter.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasindex.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasinterval.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasquadtree.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreader.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreader_bin.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreader_las.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreader_qfit.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreader_shp.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreader_txt.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasreadermerged.hpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreaditem.hpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreaditemcompressed_v1.hpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreaditemcompressed_v2.hpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreaditemraw.hpp
# End Source File
# Begin Source File

SOURCE=.\src\lasreadpoint.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasspatial.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lastransform.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasunzipper.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\lasutility.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswaveform13reader.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswaveform13writer.hpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriteitem.hpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriteitemcompressed_v1.hpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriteitemcompressed_v2.hpp
# End Source File
# Begin Source File

SOURCE=.\src\laswriteitemraw.hpp
# End Source File
# Begin Source File

SOURCE=.\src\laswritepoint.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswriter.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswriter_bin.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswriter_las.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswriter_qfit.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laswriter_txt.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laszip.hpp
# End Source File
# Begin Source File

SOURCE=.\src\laszip_common_v2.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\laszipper.hpp
# End Source File
# Begin Source File

SOURCE=.\inc\mydefs.hpp
# End Source File
# End Group
# End Target
# End Project
