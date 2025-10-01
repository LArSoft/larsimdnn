////////////////////////////////////////////////////////////////////////
// Class:       TFLoaderMLP
// Plugin Type: tool
// File:        TFLoaderMLP_tool.cc TFLoaderMLP.h
// Aug. 20, 2022 by Mu Wei (wmu@fnal.gov)
////////////////////////////////////////////////////////////////////////
#ifndef TFLoaderMLP_H
#define TFLoaderMLP_H

#include <array>
#include <string>
#include <vector>

#include "art/Utilities/ToolMacros.h"
#include "cetlib/search_path.h"
#include "fhiclcpp/ParameterSet.h"
#include "larsimdnn/PhotonPropagation/TFLoaderTools/TFLoader.h"

#include "tensorflow/core/platform/status.h"

// Forward declaration instead of full include
namespace tensorflow {
  struct SavedModelBundleLite;
}

namespace phot {
  class TFLoaderMLP : public TFLoader {
  public:
    explicit TFLoaderMLP(fhicl::ParameterSet const& pset);
    void Initialization();
    void CloseSession();

    void Predict(std::vector<double> pars) override;
    std::vector<std::vector<double>> PredictBatch(
      const std::vector<std::array<double, 3>>& positions) override;

    std::vector<double> GetPrediction() const override { return prediction; }

  private:
    std::string ModelName;               //Full path to the model (.pb) file;
    std::vector<std::string> InputsName; //Name of the input layer;
    std::string OutputName;              //Name of the output layer;

    tensorflow::SavedModelBundleLite* modelbundle;

    tensorflow::Status status; // ✔️ Fully correct
  };
}
#endif
