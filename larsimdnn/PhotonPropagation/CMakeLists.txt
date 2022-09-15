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
  
  add_subdirectory(TFLoaderTools)
endif()

install_headers()
install_fhicl()
install_source()
install_scripts()
