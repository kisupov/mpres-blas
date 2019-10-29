# Try to find the GNU Multiple Precision Arithmetic Library (GMP)
# See http://gmplib.org/

message("starting search GMP")
if (GMP_INCLUDES AND GMP_LIBRARY)
    set(GMP_FIND_QUIETLY TRUE)
endif (GMP_INCLUDES AND GMP_LIBRARY)

find_path(GMP_INCLUDES
        NAMES
        gmp.h
        PATHS
        $ENV{GMPDIR}
        ${INCLUDE_INSTALL_DIR}
        )

find_library(GMP_LIBRARY gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GMP DEFAULT_MSG
        GMP_INCLUDES GMP_LIBRARY)
if(GMP_LIBRARY AND GMP_INCLUDES)
    set(GMP_FOUND TRUE)
    message(STATUS "GMP FOUND - ${GMP_INCLUDES}")
endif()

mark_as_advanced(GMP_INCLUDES GMP_LIBRARY)