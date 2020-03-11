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

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace cv;

class Label {
public:
    int class_number;
    string class_name;
};

template<typename Classifier_t = cv::ml::KNearest>
class FacenetClassifier {
private:
    vector<Label> class_labels;
    string operation, model_path;
    tensorflow::GraphDef graph_def;
    tensorflow::Session *session;
    fstream labels_file;

public:
    Classifier<Classifier_t> classifier;

    int batch_size;

    FacenetClassifier(string operation, string model);

    std::pair<std::vector<std::string>, std::vector<int>> parse_images_path(string images_directory_path, int depth);

    Tensor create_input_tensor(const cv::Mat &image);

    Tensor create_phase_tensor();

    cv::Mat run(Tensor &input_tensor, Tensor &phase_tensor);

    void preprocess_input_mat(cv::Mat &image);

    void save_labels(const std::string &file);

    void load_labels(const std::string &file);

    void train(const cv::Mat &samples, const std::vector<int> &labels);

};

#endif

