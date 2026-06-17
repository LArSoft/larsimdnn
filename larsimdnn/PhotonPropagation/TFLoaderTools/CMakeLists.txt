cet_make_library(LIBRARY_NAME TFLoaderTool PUBLIC
  SOURCE TFLoader.cc
  LIBRARIES PRIVATE
  TensorFlow::cc
  TensorFlow::framework
)

cet_write_plugin_builder(lar::TFLoaderTool art::tool Modules
  INSTALL_BUILDER
  LIBRARIES CONDITIONAL larsimdnn::TFLoaderTool)

include(lar::TFLoaderTool)
cet_build_plugin(TFLoaderMLP lar::TFLoaderTool
  LIBRARIES PRIVATE
  fhiclcpp::fhiclcpp
  TensorFlow::framework
)

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
    # GCC 12.5 (and above) aggressively/spuriously warns about array-bounds issues.
    target_compile_options(TFLoaderTool
                         PRIVATE "-Wno-error=sign-compare;")
    target_compile_options(larsimdnn_PhotonPropagation_TFLoaderTools_TFLoaderMLP_tool
                         PRIVATE "-Wno-error=sign-compare;")
  endif()

install_headers()
install_fhicl()
install_source()
