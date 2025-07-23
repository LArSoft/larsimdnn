////////////////////////////////////////////////////////////////////////
// Class:       TFLoaderMLP
// Plugin Type: tool
// File:        TFLoaderMLP_tool.cc TFLoaderMLP.h
// Aug. 20, 2022 by Mu Wei (wmu@fnal.gov)
////////////////////////////////////////////////////////////////////////
#include "larsimdnn/PhotonPropagation/TFLoaderTools/TFLoaderMLP.h"

namespace phot {
  //......................................................................
  TFLoaderMLP::TFLoaderMLP(fhicl::ParameterSet const& pset)
    : ModelName{pset.get<std::string>("ModelName")}
    , InputsName{pset.get<std::vector<std::string>>("InputsName")}
    , OutputName{pset.get<std::string>("OutputName")}
  {}

  //......................................................................
  void TFLoaderMLP::Initialization()
  {
    int num_input = int(InputsName.size());
    if (num_input != 3) {
      std::cout << "Input name error! exit!" << std::endl;
      return;
    }
    std::string GraphFileWithPath;
    cet::search_path sp("FW_SEARCH_PATH");
    if (!sp.find_file(ModelName, GraphFileWithPath)) {
      throw cet::exception("TFLoaderMLP")
        << "In larrecodnn:phot::TFLoaderMLP: Failed to load SavedModel in : " << sp.to_string()
        << "\n";
    }
    std::cout << "larrecodnn:phot::TFLoaderMLP Loading TF Model from: " << GraphFileWithPath
              << ", Input Layer: ";
    for (int i = 0; i < num_input; ++i) {
      std::cout << InputsName[i] << " ";
    }
    std::cout << ", Output Layer: " << OutputName << "\n";

    //Load SavedModel
    modelbundle = new tensorflow::SavedModelBundleLite();

    status = tensorflow::LoadSavedModel(tensorflow::SessionOptions(),
                                        tensorflow::RunOptions(),
                                        GraphFileWithPath,
                                        {tensorflow::kSavedModelTagServe},
                                        modelbundle);

    //Initialize a tensorflow session
    //        status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
      throw cet::exception("TFLoaderMLP")
        << "In larrecodnn:phot::TFLoaderMLP: Failed to load SavedModel, status: "
        << status.ToString() << std::endl;
    }

    std::cout << "TF SavedModel loaded successfully." << std::endl;
    return;
  }

  //......................................................................
  void TFLoaderMLP::CloseSession()
  {
    if (status.ok()) {
      std::cout << "Close TF session." << std::endl;
      //            session->Close();
    }

    delete modelbundle;

    //        delete session;
    return;
  }

  //......................................................................
  void TFLoaderMLP::Predict(std::vector<double> pars)
  {
    //std::cout << "TFLoader MLP:: Predicting... " << std::endl;
    int num_input = int(pars.size());
    if (num_input != 3) {
      std::cout << "Input parameter error! exit!" << std::endl;
      return;
    }
    //Clean prediction
    std::vector<double>().swap(prediction);

    //Define inputs
    tensorflow::Tensor pos_x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1}));
    tensorflow::Tensor pos_y(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1}));
    tensorflow::Tensor pos_z(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1}));
    auto dst_x = pos_x.flat<float>().data();
    auto dst_y = pos_y.flat<float>().data();
    auto dst_z = pos_z.flat<float>().data();
    copy_n(pars.begin(), 1, dst_x);
    copy_n(pars.begin() + 1, 1, dst_y);
    copy_n(pars.begin() + 2, 1, dst_z);
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {InputsName[0], pos_x}, {InputsName[1], pos_y}, {InputsName[2], pos_z}};
    //Define outps
    std::vector<tensorflow::Tensor> outputs;

    //Run the session
    status = modelbundle->GetSession()->Run(inputs, {OutputName}, {}, &outputs);
    //        status = session->Run(inputs, {OutputName}, {}, &outputs);
    if (!status.ok()) {
      std::cout << status.ToString() << std::endl;
      return;
    }

    //Grab the outputs
    unsigned int pdr = outputs[0].shape().dim_size(1);
    //std::cout << "TFLoader MLP::Num of optical channels: " << pdr << std::endl;

    for (unsigned int i = 0; i < pdr; i++) {
      double value = outputs[0].flat<float>()(i);
      //std::cout << value << ", ";
      prediction.push_back(value);
    }
    //std::cout << std::endl;
    return;
  }

  //New batch prediction function---
  std::vector<std::vector<double>> TFLoaderMLP::PredictBatch(
    const std::vector<std::array<double, 3>>& positions)
  {
    int batch_size = positions.size();
    if (batch_size == 0) {
      std::cout << "TFLoaderMLP::PredictBatch: Empty input positions!" << std::endl;
      return {};
    }

    // Define input tensors with shape (batch_size, 1)
    tensorflow::Tensor pos_x(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, 1}));
    tensorflow::Tensor pos_y(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, 1}));
    tensorflow::Tensor pos_z(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, 1}));

    auto dst_x = pos_x.flat<float>().data();
    auto dst_y = pos_y.flat<float>().data();
    auto dst_z = pos_z.flat<float>().data();

    // Fill input tensors
    for (int i = 0; i < batch_size; ++i) {
      dst_x[i] = static_cast<float>(positions[i][0]);
      dst_y[i] = static_cast<float>(positions[i][1]);
      dst_z[i] = static_cast<float>(positions[i][2]);
    }

    // Prepare input pair list
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {InputsName[0], pos_x}, {InputsName[1], pos_y}, {InputsName[2], pos_z}};

    // Outputs
    std::vector<tensorflow::Tensor> outputs;

    // Run TensorFlow session
    status = modelbundle->GetSession()->Run(inputs, {OutputName}, {}, &outputs);
    if (!status.ok()) {
      std::cerr << "TFLoaderMLP::PredictBatch Error: " << status.ToString() << std::endl;
      return {};
    }

    // Parse output
    const tensorflow::Tensor& output_tensor = outputs[0];

    // Check output shape: should be (batch_size, num_channels)
    if (output_tensor.dims() != 2) {
      std::cerr << "TFLoaderMLP::PredictBatch Error: Output tensor has wrong shape." << std::endl;
      return {};
    }

    int output_batch_size = output_tensor.dim_size(0);
    int num_channels = output_tensor.dim_size(1);

    if (output_batch_size != batch_size) {
      std::cerr << "TFLoaderMLP::PredictBatch Error: Output batch size mismatch." << std::endl;
      return {};
    }

    // Prepare the output vector
    std::vector<std::vector<double>> predictions;
    predictions.resize(batch_size);

    auto output_data = output_tensor.flat<float>().data();

    for (int i = 0; i < batch_size; ++i) {
      predictions[i].reserve(num_channels);
      for (int j = 0; j < num_channels; ++j) {
        predictions[i].push_back(static_cast<double>(output_data[i * num_channels + j]));
      }
    }

    return predictions;
  }
}
DEFINE_ART_CLASS_TOOL(phot::TFLoaderMLP)
