# Try to find the GNU Multiple Precision Arithmetic Library (GMP)
# See http://gmplib.org/

message("starting search GARPREC")

find_path(GARPREC_INCLUDES
        NAMES
        libgarprec.a
        PATHS
        ~/CLionProjects/mpres/lib/garprec/inc/biild/
        )

find_library(GARPREC_LIBRARY garprec PATHS $ENV{GARPRECDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GARPREC DEFAULT_MSG
        GARPREC_INCLUDES GARPREC_LIBRARY)
if (GARPREC_LIBRARY AND GARPREC_INCLUDES)
    set(GARPREC_FOUND TRUE)
    message(STATUS "GARPREC FOUND - ${GARPREC_INCLUDES}")
endif ()

mark_as_advanced(GARPREC_INCLUDES GARPREC_LIBRARY)