#ifndef __FACENET_TF_H__
#define __FACENET_TF_H__


#include <iostream>
#include <fstream>
#include <string>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include <opencv2/opencv.hpp>

#include "Classifier.hpp"

namespace Facenet {

    template<typename Classifier_t>
    class FacenetClassifier {
    private:
        tensorflow::GraphDef graph_def;
        tensorflow::Session *session;

        const std::string input_layer = "input:0";
        const std::string phase_train_layer = "phase_train:0";
        const std::string output_layer = "embeddings:0";

    public:
        Classifier<Classifier_t> classifier;

        explicit FacenetClassifier(const std::string &model_path, const std::string &classifier_path);

        ~FacenetClassifier();

        std::pair<std::vector<std::string>, std::vector<int>>
        parse_images_path(const std::string &images_directory_path, int depth);

        tensorflow::Tensor create_input_tensor(const cv::Mat &image);

        tensorflow::Tensor create_phase_tensor();

        cv::Mat run(tensorflow::Tensor &input_tensor, tensorflow::Tensor &phase_tensor);

        void preprocess_input_mat(cv::Mat &image);

        void train(const cv::Mat &samples, const std::vector<int> &labels);

    };

}

#endif

