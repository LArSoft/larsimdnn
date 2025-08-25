////////////////////////////////////////////////////////////////////////
// Class:       TFLoader
// Plugin Type: tool
// File:        TFLoader.cc TFLoader.h
// Aug. 20, 2022 by Mu Wei (wmu@fnal.gov)
////////////////////////////////////////////////////////////////////////
#ifndef TFLoader_H
#define TFLoader_H

#include <array> // ✔️ Add for std::array
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

namespace phot {

  class TFLoader {
  public:
    TFLoader();
    virtual ~TFLoader() = default;

    // Initialization and shutdown
    virtual void Initialization() = 0;
    virtual void CloseSession() = 0;

    // Single-point prediction
    virtual void Predict(std::vector<double> pars) = 0;

    virtual std::vector<double> GetPrediction() const = 0;

    // Batch prediction (new!)
    virtual std::vector<std::vector<double>> PredictBatch(
      const std::vector<std::array<double, 3>>& positions) = 0;

  protected:
    std::vector<double> prediction;
  };

} // namespace phot
#endif
