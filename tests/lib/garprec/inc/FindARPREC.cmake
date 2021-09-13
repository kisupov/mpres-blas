# Try to find the GNU Multiple Precision Arithmetic Library (GMP)
# See http://gmplib.org/

message("starting search ARPREC")

find_path(ARPREC_INCLUDES
        NAMES
        c_mp.h
        PATHS
        /usr/local/include/arprec
        )

find_library(ARPREC_LIBRARY arprec PATHS $ENV{ARPRECDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ARPREC DEFAULT_MSG
        ARPREC_INCLUDES ARPREC_LIBRARY)
if (ARPREC_LIBRARY AND ARPREC_INCLUDES)
    set(ARPREC_FOUND TRUE)
    message(STATUS "ARPREC FOUND - ${ARPREC_INCLUDES}")
endif ()

mark_as_advanced(ARPREC_INCLUDES ARPREC_LIBRARY)