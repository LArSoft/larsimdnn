////////////////////////////////////////////////////////////////////////
// Class:       PDFastSimANN
// Plugin Type: producer
// File:        PDFastSimANN_module.cc
// Description:
// - acts on sim::SimEnergyDeposit from LArG4Main,
// - simulate the OpDet response to optical photons
// Input: 'sim::SimEnergyDeposit'
// Output: 'sim::OpDetBacktrackerRecord'
//Fast simulation of propagating the photons created from SimEnergyDeposits.

//This module does a fast simulation of propagating the photons created from SimEnergyDeposits,
//This simulation is done using the graph trained by artificial neural network,
//which gives the visibilities of each optical channel with respect to scinitllation vertex in the TPC volume,
//to avoid propagating single photons using Geant4.
//At the end of this module a collection of the propagated photons either as
//'sim::OpDetBacktrackerRecord' are placed into the art event.

//The steps this module takes are:
//  - to take number of photon and the vertex information from 'sim::SimEnergyDeposits',
//  - use the visibilities to determine the amount of visible photons at each optical channel,
//  - visible photons: the number of photons times the visibility at the middle of the Geant4 step for a given optical channel.
//  - other photon information is got from 'sim::SimEnergyDeposits'
//  - add 'sim::OpDetBacktrackerRecord' to event
// Aug. 20, 2020 by Mu Wei
////////////////////////////////////////////////////////////////////////

// Art libraries
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art/Utilities/make_tool.h"
#include "canvas/Utilities/Exception.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

// LArSoft libraries
#include "larcore/CoreUtils/ServiceUtil.h"
#include "larcore/Geometry/Geometry.h"
#include "larcorealg/CoreUtils/counter.h"
#include "larcorealg/CoreUtils/enumerate.h"
#include "larcorealg/Geometry/BoxBoundedGeo.h"
#include "larcorealg/Geometry/OpDetGeo.h"
#include "larcoreobj/SimpleTypesAndConstants/geo_vectors.h"
#include "lardataobj/Simulation/OpDetBacktrackerRecord.h"
#include "lardataobj/Simulation/SimEnergyDeposit.h"
#include "lardataobj/Simulation/SimPhotons.h"
#include "larsim/IonizationScintillation/ISTPC.h"
#include "larsim/PhotonPropagation/ScintTimeTools/ScintTime.h"
#include "larsim/Simulation/LArG4Parameters.h"
#include "larsimdnn/PhotonPropagation/TFLoaderTools/TFLoader.h"

// Random number engine
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "nurandom/RandomUtils/NuRandomService.h"

namespace phot {
  class PDFastSimANN : public art::EDProducer {
  public:
    explicit PDFastSimANN(fhicl::ParameterSet const&);
    void beginJob() override;
    void endJob() override;
    void produce(art::Event&) override;

  private:
    const bool fDoSlowComponent;
    const bool fUseLitePhotons;
    art::InputTag simTag;
    std::unique_ptr<ScintTime> fScintTime; //Tool to retrive timinig of scintillation
    std::unique_ptr<TFLoader>
      fTFGenerator; //Tool to predict the hit pattern based on TensorFlow network
    CLHEP::HepRandomEngine& fPhotonEngine;
    CLHEP::HepRandomEngine& fScintTimeEngine;
    std::map<int, int> PDChannelToSOCMap; //Where each OpChan is.
    int nOpChannels;                      //Number of optical detector

    void AddOpDetBTR(std::vector<sim::OpDetBacktrackerRecord>& opbtr,
                     std::map<int, int>& ChannelMap,
                     sim::OpDetBacktrackerRecord btr);
  };

  //......................................................................
  PDFastSimANN::PDFastSimANN(fhicl::ParameterSet const& pset)
    : art::EDProducer{pset}
    , fDoSlowComponent(pset.get<bool>("DoSlowComponent", true))
    , fUseLitePhotons(art::ServiceHandle<sim::LArG4Parameters const>()->UseLitePhotons())
    , simTag{pset.get<art::InputTag>("SimulationLabel")}
    , fScintTime{art::make_tool<ScintTime>(pset.get<fhicl::ParameterSet>("ScintTimeTool"))}
    , fTFGenerator{art::make_tool<TFLoader>(pset.get<fhicl::ParameterSet>("TFLoaderTool"))}
    , fPhotonEngine(art::ServiceHandle<rndm::NuRandomService>()->registerAndSeedEngine(
        createEngine(0, "HepJamesRandom", "photon"),
        "HepJamesRandom",
        "photon",
        pset,
        "SeedPhoton"))
    , fScintTimeEngine(art::ServiceHandle<rndm::NuRandomService>()->registerAndSeedEngine(
        createEngine(0, "HepJamesRandom", "scinttime"),
        "HepJamesRandom",
        "scinttime",
        pset,
        "SeedScintTime"))
  {
    std::cout << "PDFastSimANN Module Construct" << std::endl;

    if (fUseLitePhotons) {
      std::cout << "Use Lite Photon." << std::endl;
      produces<std::vector<sim::SimPhotonsLite>>();
      produces<std::vector<sim::OpDetBacktrackerRecord>>();
    }
    else {
      std::cout << "Use Sim Photon." << std::endl;
      produces<std::vector<sim::SimPhotons>>();
    }
  }

  //......................................................................
  void PDFastSimANN::beginJob()
  {
    std::cout << "PDFastSimANN beginJob." << std::endl;

    fTFGenerator->Initialization();

    art::ServiceHandle<geo::Geometry const> geo;
    nOpChannels = int(geo->Cryostat().NOpDet());
    std::cout << "Number of optical detectors: " << nOpChannels << std::endl;

    return;
  }

  //......................................................................
  void PDFastSimANN::endJob()
  {
    std::cout << "PDFastSimANN endJob." << std::endl;
    fTFGenerator->CloseSession();

    return;
  }

  //......................................................................
  void PDFastSimANN::produce(art::Event& event)
  {
    std::cout << "PDFastSimANN Module Producer..." << std::endl;

    CLHEP::RandPoissonQ randpoisphot{fPhotonEngine};

    std::unique_ptr<std::vector<sim::SimPhotons>> phot(new std::vector<sim::SimPhotons>);
    std::unique_ptr<std::vector<sim::SimPhotonsLite>> phlit(new std::vector<sim::SimPhotonsLite>);
    std::unique_ptr<std::vector<sim::OpDetBacktrackerRecord>> opbtr(
      new std::vector<sim::OpDetBacktrackerRecord>);

    auto& photonCollection(*phot);
    auto& photonLiteCollection(*phlit);

    photonCollection.resize(nOpChannels);
    photonLiteCollection.resize(nOpChannels);

    for (int i = 0; i < nOpChannels; i++) {
      photonCollection[i].fOpChannel = i;
      photonLiteCollection[i].OpChannel = i;
    }

    art::Handle<std::vector<sim::SimEnergyDeposit>> edepHandle;
    if (!event.getByLabel(simTag, edepHandle)) {
      std::cout << "PDFastSimANN Module Cannot getByLabel: " << simTag << std::endl;
      return;
    }

    art::ServiceHandle<geo::Geometry> geom;
    auto const& edeps = edepHandle;

    int num_points = edeps->size();
    float vis_scale = 1.0;

    // Prepare input positions for batch inference
    std::vector<std::array<double, 3>> positions;
    positions.reserve(num_points);

    for (auto const& edepi : *edeps) {
      positions.push_back({edepi.MidPointX(), edepi.MidPointY(), edepi.MidPointZ()});
    }

    // Run batch prediction
    auto VisibilityBatch = fTFGenerator->PredictBatch(positions);

    if (VisibilityBatch.size() != size_t(num_points)) {
      std::cout << "PDFastSimANN: Visibility batch size mismatch." << std::endl;
      return;
    }

    // Loop over energy deposits
    for (size_t idx = 0; idx < edeps->size(); ++idx) {
      auto const& edepi = edeps->at(idx);
      auto const& Visibilities = VisibilityBatch[idx];

      if (int(Visibilities.size()) != nOpChannels) {
        std::cout << "PDFastSimANN get channels from graph " << Visibilities.size()
                  << " is not the same as from geometry: " << nOpChannels << std::endl;
        continue;
      }

      int trackID = edepi.TrackID();
      int nphot_fast = edepi.NumFPhotons();
      int nphot_slow = edepi.NumSPhotons();
      double edeposit = edepi.Energy() / edepi.NumPhotons();

      double pos[3] = {edepi.MidPointX(), edepi.MidPointY(), edepi.MidPointZ()};

      for (int channel = 0; channel < nOpChannels; ++channel) {
        auto visibleFraction = Visibilities[channel] * vis_scale;

        if (visibleFraction == 0.0) { continue; }

        if (fUseLitePhotons) {
          sim::OpDetBacktrackerRecord tmpbtr(channel);

          if (nphot_fast > 0) {
            auto n = static_cast<int>(randpoisphot.fire(nphot_fast * visibleFraction));
            for (long i = 0; i < n; ++i) {
              fScintTime->GenScintTime(true, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              ++photonLiteCollection[channel].DetectedPhotons[time];
              tmpbtr.AddScintillationPhotons(trackID, time, 1, pos, edeposit);
            }
          }

          if ((nphot_slow > 0) && fDoSlowComponent) {
            auto n = static_cast<int>(randpoisphot.fire(nphot_slow * visibleFraction));
            for (long i = 0; i < n; ++i) {
              fScintTime->GenScintTime(false, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              ++photonLiteCollection[channel].DetectedPhotons[time];
              tmpbtr.AddScintillationPhotons(trackID, time, 1, pos, edeposit);
            }
          }

          AddOpDetBTR(*opbtr, PDChannelToSOCMap, tmpbtr);
        }
        else {
          sim::OnePhoton photon;
          photon.SetInSD = false;
          photon.InitialPosition = {edepi.MidPointX(), edepi.MidPointY(), edepi.MidPointZ()};
          photon.Energy = 9.7e-6;

          if (nphot_fast > 0) {
            auto n = static_cast<int>(randpoisphot.fire(nphot_fast * visibleFraction));
            if (n > 0) {
              fScintTime->GenScintTime(true, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              photon.Time = time;
              photonCollection[channel].insert(photonCollection[channel].end(), n, photon);
            }
          }

          if ((nphot_slow > 0) && fDoSlowComponent) {
            auto n = static_cast<int>(randpoisphot.fire(nphot_slow * visibleFraction));
            if (n > 0) {
              fScintTime->GenScintTime(false, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              photon.Time = time;
              photonCollection[channel].insert(photonCollection[channel].end(), n, photon);
            }
          }
        }
      }
    }

    std::cout << "PDFastSimANN module produced " << num_points << " images..." << std::endl;
    PDChannelToSOCMap.clear();

    if (fUseLitePhotons) {
      event.put(move(phlit));
      event.put(move(opbtr));
    }
    else {
      event.put(move(phot));
    }

    return;
  }

  //......................................................................
  void PDFastSimANN::AddOpDetBTR(std::vector<sim::OpDetBacktrackerRecord>& opbtr,
                                 std::map<int, int>& ChannelMap,
                                 sim::OpDetBacktrackerRecord btr)
  {
    int iChan = btr.OpDetNum();
    std::map<int, int>::iterator channelPosition = ChannelMap.find(iChan);

    if (channelPosition == ChannelMap.end()) {
      ChannelMap[iChan] = opbtr.size();
      opbtr.emplace_back(std::move(btr));
    }
    else {
      unsigned int idtest = channelPosition->second;
      auto const& timePDclockSDPsMap = btr.timePDclockSDPsMap();

      for (auto const& timePDclockSDP : timePDclockSDPsMap) {
        for (auto const& sdp : timePDclockSDP.second) {
          double xyz[3] = {sdp.x, sdp.y, sdp.z};
          opbtr.at(idtest).AddScintillationPhotons(
            sdp.trackID, timePDclockSDP.first, sdp.numPhotons, xyz, sdp.energy);
        }
      }
    }

    return;
  }
} // namespace

DEFINE_ART_MODULE(phot::PDFastSimANN)
