# ======================================================================
#  larsimdnn main build file
#
#  cd .../path/to/build/directory
#  source .../path/to/larsimdnn/ups/setup_for_development <-d|-p>
#  buildtool [-I /install/path] -bt[i] -jN
# ======================================================================


cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

find_package(cetmodules REQUIRED)
project(larsimdnn VERSION 10.01.01 LANGUAGES CXX)

include(CetCMakeEnv)
cet_cmake_env()

cet_set_compiler_flags(DIAGS CAUTIOUS
  WERROR
  NO_UNDEFINED
  EXTRA_FLAGS -pedantic -Wno-unused-local-typedefs
)

cet_report_compiler_flags(REPORT_THRESHOLD VERBOSE)

# needed because we call cet_make_plugin_builder
cet_cmake_module_directories(Modules BINARY)

find_package(larfinder REQUIRED EXPORT)

find_package(larsim REQUIRED EXPORT)
find_package(larevt REQUIRED EXPORT)
find_package(lardata REQUIRED EXPORT)
find_package(larcore REQUIRED EXPORT)
find_package(lardataobj REQUIRED EXPORT)
find_package(larcorealg REQUIRED EXPORT)
find_package(larcoreobj REQUIRED EXPORT)

find_package(Eigen3 REQUIRED EXPORT)
find_package(TensorFlow 2.6.0 QUIET EXPORT)

# macros for dictionary and simple_plugin
include(BuildPlugins)

# source
# enable when code is added
add_subdirectory(larsimdnn)

# ups - table and config files

# packaging utility
cet_cmake_config()
