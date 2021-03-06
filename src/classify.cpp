#include <iostream>
#include <string>
#include "facenet_tf.h"

using namespace std;

const cv::Size image_size = cv::Size(160, 160);

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage:\n"
                "./facenet_classify <TRAIN|CLASSIFY>  <Path/To/TensorFlowModel> <Path/To/Image/Directory/Structure> <Path/To/Classifier/Model>\n"
                "Directory structure should be <class_id>/<image_files>\n"
                "Face Images Should be 160x160\n" << endl;
        return 1;
    }
    string operation, images_path, model_path, classifier_model_path, labels_file_path;
    operation = string(argv[1]);
    model_path = string(argv[2]);
    images_path = string(argv[3]);
    classifier_model_path = string(argv[4]);

    Facenet::FacenetClassifier<cv::ml::ANN_MLP> classifier(model_path, classifier_model_path);

    cv::Mat results;

    auto input_files = classifier.parse_images_path(images_path, 0);
    for (const auto &file : input_files.first) {
        cv::Mat image = cv::imread(file);
        if (image.empty()) {
            cerr << "Cannot load image from " << file << endl;
        } else if (image.size() != image_size) {
            cerr << "Image " << file << " has different size than " << image_size << ", resizing it" << endl;
            cv::resize(image, image, image_size);
        }
        classifier.preprocess_input_mat(image);
        tensorflow::Tensor input_tensor = classifier.create_input_tensor(image);
        tensorflow::Tensor phase_tensor = classifier.create_phase_tensor();
        cv::Mat output = classifier.run(input_tensor, phase_tensor);
        if (results.empty()) {
            results = output;
        } else {
            cv::vconcat(results, output, results);
        }
    }

    if (operation == "TRAIN") {
        classifier.train(results, input_files.second);
        classifier.classifier.save(classifier_model_path);
        cout << "Model trained successfully and saved to " << classifier_model_path << endl;
    } else if (operation == "CLASSIFY") {
        if (!classifier.classifier.ok) {
            LOG(ERROR) << "Classifier is not opened" << endl;
            return EXIT_FAILURE;
        }
        for (int i = 0; i < input_files.first.size(); i++) {
            cv::Mat input_mat;
            results.row(i).convertTo(input_mat, CV_32F);
            cout << "Predicted: " << classifier.classifier.predict(input_mat) << ", Actual: " << input_files.second[i]
                 << endl;
        }
    }

    return EXIT_SUCCESS;
}

