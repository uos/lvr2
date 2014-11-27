# Originally copied from the KDE project repository:
# http://websvn.kde.org/trunk/KDE/kdeutils/cmake/modules/FindGMP.cmake?view=markup&pathrev=675218

# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
# Copyright (c) 2007, 2008 Francesco Biscani, <bluescarni@gmail.com>

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products 
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------------------

# Try to find the GMP librairies:
# GMP_FOUND - System has GMP lib
# GMP_INCLUDE_DIR - The GMP include directory
# GMP_LIBRARIES - Libraries needed to use GMP
# GMPXX_INCLUDE_DIR - The GMP C++ interface include directory
# GMPXX_LIBRARIES - Libraries needed to use GMP's C++ interface

IF(GMP_INCLUDE_DIR AND GMP_LIBRARIES AND GMPXX_INCLUDE_DIR AND GMPXX_LIBRARIES)
	# Already in cache, be silent
	SET(GMP_FIND_QUIETLY TRUE)
ENDIF(GMP_INCLUDE_DIR AND GMP_LIBRARIES AND GMPXX_INCLUDE_DIR AND GMPXX_LIBRARIES)

FIND_PATH(GMP_INCLUDE_DIR NAMES gmp.h)
FIND_LIBRARY(GMP_LIBRARIES NAMES gmp)
FIND_PATH(GMPXX_INCLUDE_DIR NAMES gmpxx.h)
FIND_LIBRARY(GMPXX_LIBRARIES NAMES gmpxx)

IF(GMP_INCLUDE_DIR AND GMP_LIBRARIES AND GMPXX_INCLUDE_DIR AND GMPXX_LIBRARIES)
	SET(GMP_FOUND TRUE)
ENDIF(GMP_INCLUDE_DIR AND GMP_LIBRARIES AND GMPXX_INCLUDE_DIR AND GMPXX_LIBRARIES)

IF(GMP_FOUND)
	IF(NOT GMP_FIND_QUIETLY)
		MESSAGE(STATUS "Found GMP: ${GMP_LIBRARIES}, ${GMPXX_LIBRARIES}")
	ENDIF(NOT GMP_FIND_QUIETLY)
ELSE(GMP_FOUND)
	IF(GMP_FIND_REQUIRED)
		MESSAGE(FATAL_ERROR "Could NOT find GMP")
	ENDIF(GMP_FIND_REQUIRED)
ENDIF(GMP_FOUND)

MARK_AS_ADVANCED(GMP_INCLUDE_DIR GMP_LIBRARIES GMPXX_INCLUDE_DIR GMPXX_LIBRARIES)

