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
    vector<int> output_labels, training_labels;
    string operation, model_path, labels_file_path;
    tensorflow::GraphDef graph_def;
    tensorflow::Session *session;
    Tensor input_tensor, phase_tensor;
    fstream labels_file;
    Mat mat_training_tensors;

public:
    Classifier<Classifier_t> classifier;
    vector<Mat> input_images;
    Mat output_mat;
    vector<string> input_files;

    int batch_size;

    FacenetClassifier(string operation, string model, string labels_file_path);

    long parse_images_path(string images_directory_path, int depth);

    void load_input_images(long, long);

    void create_input_tensor(long, long);

    void create_phase_tensor();

    void run(long, long);

    void release_batch_images(long, long);

    void preprocess_input_mat(long, long);

    void save_labels();

    void load_labels();

    void set_input_images(std::vector<Mat>, std::vector<string>);

    void clear_input_images();

    void train();

};

#endif

