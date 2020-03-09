#include <iostream>
#include <string>
#include "facenet_tf.h"

#ifndef DEBUG
#define DEBUG cout << __FUNCTION__ << ":" << __LINE__ << endl;
#endif
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage:\n"
                "./facenet_classify <TRAIN|CLASSIFY>  <Path/To/TensorFlowModel> <Path/To/Image/Directory/Structure> <Path/To/Classifier/Model> <Classifier/Class/Labels>\n"
                "Directory structure should be <class_name>/<image_files>\n"
                "Face Images Should be 160x160\n" << endl;
        return 1;
    }
    string operation, images_path, model_path, classifier_model_path, labels_file_path;
    operation = string(argv[1]);
    model_path = string(argv[2]);
    images_path = string(argv[3]);
    classifier_model_path = string(argv[4]);
    labels_file_path = string(argv[5]);
    long input_files_count;
    int i;
    long end_index;

    FacenetClassifier<cv::ml::KNearest> classifier(operation, model_path, labels_file_path);
    classifier.batch_size = 1000;

    input_files_count = classifier.parse_images_path(images_path, 0);

    for (i = 0; i < input_files_count; i += classifier.batch_size) {
        cout << "Processing Images: " << i << " to " << i + classifier.batch_size - 1 << endl;
        if ((i + classifier.batch_size) > input_files_count)
            end_index = input_files_count;
        else
            end_index = i + classifier.batch_size;
        classifier.load_input_images(i, end_index);
        classifier.preprocess_input_mat(i, end_index);
        classifier.create_input_tensor(i, end_index);
        classifier.create_phase_tensor();
        classifier.run(i, end_index);
        classifier.release_batch_images(i, end_index);
    }

    if (operation == "TRAIN") {
        classifier.train();
        classifier.classifier.save(classifier_model_path);
        classifier.save_labels();
    } else if (operation == "CLASSIFY") {
        classifier.classifier.load(classifier_model_path);
        classifier.load_labels();
        for (int i = 0; i < classifier.input_images.size(); i++) {
            Mat input_mat;
            Mat result;
            classifier.output_mat.row(i).convertTo(input_mat, CV_32F);
            cout << classifier.classifier.predict(input_mat) << " " << classifier.input_files[i] << endl;
        }
    }

    return 0;
}

