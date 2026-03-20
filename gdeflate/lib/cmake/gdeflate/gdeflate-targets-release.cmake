#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gdeflate::gdeflate" for configuration "Release"
set_property(TARGET gdeflate::gdeflate APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gdeflate::gdeflate PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgdeflate.so.1.0.0"
  IMPORTED_SONAME_RELEASE "libgdeflate.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS gdeflate::gdeflate )
list(APPEND _IMPORT_CHECK_FILES_FOR_gdeflate::gdeflate "${_IMPORT_PREFIX}/lib/libgdeflate.so.1.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
