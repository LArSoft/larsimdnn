if (TensorFlow_FOUND)
  cet_build_plugin(PDFastSimANN art::EDProducer
    LIBRARIES PRIVATE
    TensorFlow::cc
    TensorFlow::framework
    larsim::PhotonPropagation
    larcore::ServiceUtil
    larcorealg::Geometry
    larcoreobj::geo_vectors
    lardataobj::Simulation
    nurandom::RandomUtils_NuRandomService_service
    art::Framework_Principal
    art_plugin_support::toolMaker
    canvas::canvas
    messagefacility::MF_MessageLogger
    fhiclcpp::types
    fhiclcpp::fhiclcpp
    cetlib_except::cetlib_except
    CLHEP::Random
  )
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
    # GCC 12.5 (and above) aggressively/spuriously warns about array-bounds issues.
    target_compile_options(larsimdnn_PhotonPropagation_PDFastSimANN_module
                         PRIVATE "-Wno-error=sign-compare")
  endif()
  add_subdirectory(TFLoaderTools)
endif()

install_headers()
install_source()

