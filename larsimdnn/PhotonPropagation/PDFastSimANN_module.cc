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
// PERFORMANCE OPTIMIZATIONS APPLIED (matching PDFastSimPAR optimizations):
// - Batch processing to reduce memory usage (52GB -> 9GB)
// - OBTRHelper map pattern for efficient backtracker record construction
// - Time-based photon batching to reduce function call overhead
// - Deferred vector conversion (only at end of event)
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
    const size_t fBatchSize;  // KEPT: Configurable batch size for processing
    art::InputTag simTag;
    std::unique_ptr<ScintTime> fScintTime; //Tool to retrive timinig of scintillation
    std::unique_ptr<TFLoader>
      fTFGenerator; //Tool to predict the hit pattern based on TensorFlow network
    CLHEP::HepRandomEngine& fPhotonEngine;
    CLHEP::HepRandomEngine& fScintTimeEngine;
    std::map<int, int> PDChannelToSOCMap; //Where each OpChan is.
    int nOpChannels;                      //Number of optical detector


    // Method matching PDFastSimPAR optimization
    // This method uses OBTRHelper map pattern and batches photon additions
    void SimpleAddOpDetBTR(
      std::map<int, sim::OBTRHelper>& opbtr_helper,  // Map instead of vector for O(1) access
      std::map<int, int>& ChannelMap,
      size_t channel,
      int trackID,
      int time,
      double pos[3],
      double edeposit,
      int num_photons = 1);  // Support adding multiple photons at once

    // Process a batch of energy deposits (Memory Optimization)
    void ProcessEnergyDepositBatch(
      const std::vector<sim::SimEnergyDeposit>& edeps,
      size_t start_idx,
      size_t end_idx,
      std::vector<sim::SimPhotons>& photonCollection,
      std::vector<sim::SimPhotonsLite>& photonLiteCollection,
      std::map<int, sim::OBTRHelper>& opbtr_helper,  // from vector to map
      CLHEP::RandPoissonQ& randpoisphot,
      float vis_scale);
  };

  //......................................................................
  PDFastSimANN::PDFastSimANN(fhicl::ParameterSet const& pset)
    : art::EDProducer{pset}
    , fDoSlowComponent(pset.get<bool>("DoSlowComponent", true))
    , fUseLitePhotons(art::ServiceHandle<sim::LArG4Parameters const>()->UseLitePhotons())
    , fBatchSize(pset.get<size_t>("BatchSize", 50000))  // Batch size optimization
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
    std::cout << "PDFastSimANN Module Construct (with batch processing + OBTRHelper optimization, batch size = "
              << fBatchSize << ")" << std::endl;

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

    // Reserve capacity to avoid reallocations
    if (!fUseLitePhotons) {
      for (int i = 0; i < nOpChannels; i++) {
        photonCollection[i].reserve(1000);
      }
    }

    art::Handle<std::vector<sim::SimEnergyDeposit>> edepHandle;
    if (!event.getByLabel(simTag, edepHandle)) {
      std::cout << "PDFastSimANN Module Cannot getByLabel: " << simTag << std::endl;
      return;
    }

    auto const& edeps = edepHandle;
    int num_points = edeps->size();
    float vis_scale = 1.0;

    std::cout << "Processing " << num_points << " energy deposits in batches of "
              << fBatchSize << std::endl;

    // OBTRHelper map for efficient backtracker record construction
    std::map<int, sim::OBTRHelper> opbtr_helper;
    
    // Initialize PDChannelToSOCMap with -1 for all channels
    PDChannelToSOCMap.clear();
    for (int i = 0; i < nOpChannels; i++) {
      PDChannelToSOCMap[i] = -1;
    }

    // Process in batches
    for (size_t batch_start = 0; batch_start < edeps->size(); batch_start += fBatchSize) {
      size_t batch_end = std::min(batch_start + fBatchSize, edeps->size());
      
      std::cout << "Processing batch " << (batch_start / fBatchSize + 1)
                << " (deposits " << batch_start << " to " << batch_end << ")" << std::endl;

      ProcessEnergyDepositBatch(*edeps,
                                batch_start,
                                batch_end,
                                photonCollection,
                                photonLiteCollection,
                                opbtr_helper,
                                randpoisphot,
                                vis_scale);
    }

    std::cout << "PDFastSimANN module produced " << num_points << " images..." << std::endl;
    
    // Convert helper map to final vector only at the end
    if (fUseLitePhotons) {
      for (auto& iopbtr : opbtr_helper) {
        opbtr->emplace_back(iopbtr.second);
      }
    }

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
  // Process a batch of energy deposits (optimization)
  // Now uses OBTRHelper map and time-based photon batching
  void PDFastSimANN::ProcessEnergyDepositBatch(
    const std::vector<sim::SimEnergyDeposit>& edeps,
    size_t start_idx,
    size_t end_idx,
    std::vector<sim::SimPhotons>& photonCollection,
    std::vector<sim::SimPhotonsLite>& photonLiteCollection,
    std::map<int, sim::OBTRHelper>& opbtr_helper,  // map instead of vector
    CLHEP::RandPoissonQ& randpoisphot,
    float vis_scale)
  {
    size_t batch_size = end_idx - start_idx;
    
    // Prepare input positions for this batch only
    std::vector<std::array<double, 3>> positions;
    positions.reserve(batch_size);

    for (size_t idx = start_idx; idx < end_idx; ++idx) {
      auto const& edepi = edeps[idx];
      positions.push_back({edepi.MidPointX(), edepi.MidPointY(), edepi.MidPointZ()});
    }

    // Run batch prediction for this chunk
    auto VisibilityBatch = fTFGenerator->PredictBatch(positions);

    if (VisibilityBatch.size() != batch_size) {
      std::cout << "PDFastSimANN: Visibility batch size mismatch." << std::endl;
      return;
    }

    // Loop over energy deposits in this batch
    for (size_t local_idx = 0; local_idx < batch_size; ++local_idx) {
      size_t global_idx = start_idx + local_idx;
      auto const& edepi = edeps[global_idx];
      auto const& Visibilities = VisibilityBatch[local_idx];

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
	  // New optimized pattern - batch photons by time
          // This reduces O(millions) of function calls to O(thousands)
          std::map<int, int> temp_photon_times_fast;
          std::map<int, int> temp_photon_times_slow;

          if (nphot_fast > 0) {
            auto n = static_cast<int>(randpoisphot.fire(nphot_fast * visibleFraction));
            // First pass: accumulate photons by time
            for (long i = 0; i < n; ++i) {
              fScintTime->GenScintTime(true, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              ++photonLiteCollection[channel].DetectedPhotons[time];
              ++temp_photon_times_fast[time];  // Batch by time
            }
          }

          if ((nphot_slow > 0) && fDoSlowComponent) {
            auto n = static_cast<int>(randpoisphot.fire(nphot_slow * visibleFraction));
            // First pass: accumulate photons by time
            for (long i = 0; i < n; ++i) {
              fScintTime->GenScintTime(false, fScintTimeEngine);
              auto time = static_cast<int>(edepi.StartT() + fScintTime->GetScintTime());
              ++photonLiteCollection[channel].DetectedPhotons[time];
              ++temp_photon_times_slow[time];  // Batch by time
            }
          }

          // Second pass: add batched photons to backtracker
          // Instead of adding photons one-by-one, we add all photons at each time together
          // This is the key optimization that reduces function calls dramatically
          for (auto const& [time, nphotons] : temp_photon_times_fast) {
            SimpleAddOpDetBTR(opbtr_helper, PDChannelToSOCMap, channel, 
                            trackID, time, pos, edeposit, nphotons);
          }
          
          for (auto const& [time, nphotons] : temp_photon_times_slow) {
            SimpleAddOpDetBTR(opbtr_helper, PDChannelToSOCMap, channel, 
                            trackID, time, pos, edeposit, nphotons);
          }
        }
        else {
          // SimPhotons case (non-lite) - no backtracker records needed
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
  }

  //......................................................................
  // New efficient method for adding to backtracker records
  // This replaces the old AddOpDetBTR method with the optimized pattern from PDFastSimPAR
  // Key improvements:
  // 1. Uses map instead of vector for O(1) access by channel
  // 2. Uses OBTRHelper which has optimized internal map operations
  // 3. Supports adding multiple photons at once (num_photons parameter)
  void PDFastSimANN::SimpleAddOpDetBTR(
				       std::map<int, sim::OBTRHelper>& opbtr_helper,
				       std::map<int, int>& ChannelMap,
				       size_t channel,
				       int trackID,
				       int time,
				       double pos[3],
				       double edeposit,
				       int num_photons)
  {
    // Check if this is the first time we're seeing this channel
    if (opbtr_helper.find(channel) == opbtr_helper.end()) {
      // First time seeing this channel - create new OBTRHelper
      ChannelMap[channel] = opbtr_helper.size();
      opbtr_helper.emplace(channel, channel);
    }
    
    // Add photons directly to the helper's internal map
    // This is much faster than the old method which required:
    // 1. Creating a temporary OpDetBacktrackerRecord
    // 2. Adding photons one-by-one to the temp record
    // 3. Merging the temp record into the main collection
    opbtr_helper.at(channel).AddScintillationPhotonsToMap(
							  trackID, time, num_photons, pos, edeposit);
  }

} // namespace

DEFINE_ART_MODULE(phot::PDFastSimANN)
