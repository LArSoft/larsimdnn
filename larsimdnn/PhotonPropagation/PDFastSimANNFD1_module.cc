////////////////////////////////////////////////////////////////////////
// Class:       PDFastSimANNFD1
// Plugin Type: producer
// File:        PDFastSimANNFD1_module.cc
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
//
//Modified by Shu, May 28, 2025
//This module works specifically for FD1 (Horitonzal drift; 6000 channels)
//
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
  class PDFastSimANNFD1 : public art::EDProducer {
  public:
    explicit PDFastSimANNFD1(fhicl::ParameterSet const&);
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
  PDFastSimANNFD1::PDFastSimANNFD1(fhicl::ParameterSet const& pset)
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
    std::cout << "PDFastSimANNFD1 Module Construct" << std::endl;

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
  void PDFastSimANNFD1::beginJob()
  {
    std::cout << "PDFastSimANNFD1 beginJob." << std::endl;

    fTFGenerator->Initialization();

    art::ServiceHandle<geo::Geometry const> geo;
    nOpChannels = int(geo->Cryostat().NOpDet());
    std::cout << "Number of optical detectors: " << nOpChannels << std::endl;

    return;
  }

  //......................................................................
  void PDFastSimANNFD1::endJob()
  {
    std::cout << "PDFastSimANNFD1 endJob." << std::endl;
    fTFGenerator->CloseSession();

    return;
  }

  //......................................................................
  void PDFastSimANNFD1::produce(art::Event& event)
  {
    std::cout << "PDFastSimANNFD1 Module Producer..." << std::endl;

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
      std::cout << "PDFastSimANNFD1 Module Cannot getByLabel: " << simTag << std::endl;
      return;
    }

    art::ServiceHandle<geo::Geometry> geom;
    auto const& edeps = edepHandle;

    int num_points = 0;
    float vis_scale = 1.0; // to scale the visibility fraction, for test only;
    for (auto const& edepi : *edeps) {
      num_points++;
      int trackID = edepi.TrackID();
      int nphot = edepi.NumPhotons();
      double edeposit = edepi.Energy() / nphot;
      double pos[3] = {edepi.MidPointX(), edepi.MidPointY(), edepi.MidPointZ()};

      int nphot_fast = edepi.NumFPhotons();
      int nphot_slow = edepi.NumSPhotons();


      
      //To expand responses to Full space, 20250514---
      std::vector<double> pars = {pos[0], pos[1], pos[2]};
      std::vector<double> Visibilities;
      
      if (pos[0] > 364 && pos[0] < 726) {
          // Case 1: 364 < pos[0] < 726
          fTFGenerator->Predict(pars);
          Visibilities = fTFGenerator->GetPrediction();
          Visibilities.insert(Visibilities.end(), 4000, 0.0);
      }

      else if (pos[0] > -364 && pos[0] < 364) {
        // Case 2: -364 < pos[0] < 364
        pars[0] = 726 - std::abs(pos[0]);
        fTFGenerator->Predict(pars);
        std::vector<double> prediction = fTFGenerator->GetPrediction();

        // Sanity check
        if (prediction.size() != 2000) {
          std::cerr << "Unexpected prediction size: " << prediction.size() << std::endl;
        }

        // Split the prediction into two halves
        std::vector<double> second_half(prediction.begin() + 1000, prediction.end());
        std::vector<double> first_half(prediction.begin(), prediction.begin() + 1000);

        // Build Visibilities: 2000 zeros + second_half + first_half + 2000 zeros
        Visibilities.clear();
        Visibilities.insert(Visibilities.end(), 2000, 0.0);
        Visibilities.insert(Visibilities.end(), second_half.begin(), second_half.end());
        Visibilities.insert(Visibilities.end(), first_half.begin(), first_half.end());
        Visibilities.insert(Visibilities.end(), 2000, 0.0);
      }

      else if (pos[0] > -726 && pos[0] < -364) {
          // Case 3: -726 < pos[0] < -364
          pars[0] = std::abs(pos[0]);
          fTFGenerator->Predict(pars);
          Visibilities = fTFGenerator->GetPrediction();
          Visibilities.insert(Visibilities.begin(), 4000, 0.0);
      }
      else {
          // Optional: Handle invalid pos[0] case (outside [-726, 726])
          Visibilities = std::vector<double>(6000, 0.0);
      }
      
//      std::cout<<"\nSingle vertex prediction: "<<std::endl;
//      std::cout<<"Adding 4000 zeros successfully!!!===========\n"<<std::endl;




      if (int(Visibilities.size()) != nOpChannels) {
        std::cout << "PDFastSimANNFD1 get channels from graph " << Visibilities.size()
                  << " is not the same as from geometry: " << nOpChannels << std::endl;
        break;
      }

      for (int channel = 0; channel < int(Visibilities.size()); ++channel) {
        auto visibleFraction =
          Visibilities[channel] * vis_scale; // to scale the visibility fraction;

        if (visibleFraction == 0.0) {
          continue; //vertex is not visible at this optical channel.
        }

        if (fUseLitePhotons) {
          sim::OpDetBacktrackerRecord tmpbtr(channel);
          if (nphot_fast > 0) {
            //random number, poisson distribution, mean: the amount of photons visible at this channel
            auto n = static_cast<int>(randpoisphot.fire(nphot_fast * visibleFraction));
            for (long i = 0; i < n; ++i) {
              //calculates the time at which the photon was produced
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
            //random number, poisson distribution, mean: the amount of photons visible at this channel
            auto n = static_cast<int>(randpoisphot.fire(nphot_fast * visibleFraction));
            if (n > 0) {
              fScintTime->GenScintTime(true, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              photon.Time = time;
              // add n copies of sim::OnePhoton photon to the photon collection for a given OpChannel
              photonCollection[channel].insert(photonCollection[channel].end(), n, photon);
            }
          }
          if ((nphot_slow > 0) && fDoSlowComponent) {
            //random number, poisson distribution, mean: the amount of photons visible at this channel
            auto n = static_cast<int>(randpoisphot.fire(nphot_slow * visibleFraction));
            if (n > 0) {
              fScintTime->GenScintTime(false, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              photon.Time = time;
              // add n copies of sim::OnePhoton photon to the photon collection for a given OpChannel
              photonCollection[channel].insert(photonCollection[channel].end(), n, photon);
            }
          }
        }
      }
    }
    std::cout << "PDFastSimANNFD1 module produced " << num_points << " images..." << std::endl;
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
  void PDFastSimANNFD1::AddOpDetBTR(std::vector<sim::OpDetBacktrackerRecord>& opbtr,
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

DEFINE_ART_MODULE(phot::PDFastSimANNFD1)
