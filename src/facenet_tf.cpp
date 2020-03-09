#include "facenet_tf.h"
#include <dirent.h>
#include <cstring>

template<>
FacenetClassifier<>::FacenetClassifier(string operation, string model_path,
                                       string labels_file_path) {
    this->operation = operation;
    this->model_path = model_path;
    this->labels_file_path = labels_file_path;
    ReadBinaryProto(tensorflow::Env::Default(), model_path.c_str(), &graph_def);
    tensorflow::SessionOptions options;
    tensorflow::NewSession(options, &session);
    session->Create(graph_def);
}

template<>
void FacenetClassifier<>::save_labels() {
    cout << "Saving Labels To " << labels_file_path << endl;
    labels_file.open(labels_file_path, fstream::out);

    for (Label label: class_labels) {
        labels_file << label.class_number << " " << label.class_name << endl;
    }
    labels_file.close();
    cout << "Done Saving labels" << endl;
}

template<>
void FacenetClassifier<>::load_labels() {
    class_labels.clear();
    labels_file.open(labels_file_path, fstream::in);
    int count = 0;
    while (true) {
        if (labels_file.eof())
            break;
        Label label;
        labels_file >> label.class_number >> label.class_name;
        class_labels.push_back(label);
        count++;
    }
    labels_file.close();
    cout << class_labels.size() << endl;
}

template<>
void FacenetClassifier<>::preprocess_input_mat(long start_index, long end_index) {
    long i;
    for (i = start_index; i < end_index; i++) {
        Mat &image = input_images[i];
        //mean and std
        cv::Mat temp = image.reshape(1, image.rows * 3);
        cv::Mat mean3;
        cv::Mat stddev3;
        cv::meanStdDev(temp, mean3, stddev3);

        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);
        cv::Mat image2;
        // cvtColor(image, image, cv::COLOR_BGR2GRAY);
        image.convertTo(image2, CV_64FC1);
        image = image2;
        image = image - cv::Scalar(mean_pxl, mean_pxl, mean_pxl);
        image = image / stddev_pxl;
    }
}

template<>
long FacenetClassifier<>::parse_images_path(string directory_path, int depth) {
    cout << "Parsing Directory: " << directory_path << endl;
    DIR *dir;
    struct dirent *entry;
    static int class_count = -1;
    string class_name;
    string file_name, file_path;
    if ((dir = opendir(directory_path.c_str())) != NULL) {

        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                class_name = string(entry->d_name);
                class_count++;
                parse_images_path(directory_path + "/" + class_name, depth + 1);
                Label label;
                label.class_number = class_count;
                label.class_name = class_name;
                class_labels.push_back(label);
            } else if (entry->d_type != DT_DIR) {
                file_name = string(entry->d_name);
                file_path = directory_path + "/" + file_name;
                cout << file_path << ":" << class_count << endl;
                output_labels.push_back(class_count); //For Training
                training_labels.push_back(class_count);
                input_files.push_back(file_path); //For Classification
            }
        }
        closedir(dir);
    }
    return input_files.size();
}

template<>
void FacenetClassifier<>::load_input_images(long start_index, long end_index) {
    long i;
    for (i = start_index; i < end_index; i++) {
        cout << "Loading Image: " << input_files[i] << endl;
        Mat image = imread(input_files[i].c_str());
        if (image.empty() || image.rows != 160 || image.cols != 160)
            cout << "Ignoring Image " + input_files[i] << endl;
        input_images.push_back(image);
    }
}

template<>
void FacenetClassifier<>::create_input_tensor(long start_index, long end_index) {
    cout << "Using " << input_images.size() << " images" << endl;
    cout << "Start Index:" << start_index << " End Index:" << end_index << endl;
    Tensor input_tensor(DT_FLOAT, TensorShape({(int) (end_index - start_index), 160, 160, 3}));
    // get pointer to memory for that Tensor
    float *p = input_tensor.flat<float>().data();
    int i;

    for (i = 0; i < (end_index - start_index); i++) {
        // create a "fake" cv::Mat from it

        Mat camera_image(160, 160, CV_32FC3, p + i * 160 * 160 * 3);
        input_images[i + start_index].convertTo(camera_image, CV_32FC3);
    }
    cout << input_tensor.DebugString() << endl;
    this->input_tensor = Tensor(input_tensor);
}

template<>
void FacenetClassifier<>::create_phase_tensor() {
    tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_tensor.scalar<bool>()() = false;
    this->phase_tensor = Tensor(phase_tensor);
}

template<>
void FacenetClassifier<>::run(long start_index, long end_index) {
    string input_layer = "input:0";
    string phase_train_layer = "phase_train:0";
    string output_layer = "embeddings:0";
    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::pair<string, tensorflow::Tensor>> feed_dict = {
            {input_layer,       input_tensor},
            {phase_train_layer, phase_tensor},
    };
    tensorflow::SessionOptions options;
    tensorflow::NewSession(options, &session);
    session->Create(graph_def);
    cout << "Input Tensor: " << input_tensor.DebugString() << endl;
    Status run_status = session->Run(feed_dict, {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status << "\n";
        return;
    }
    cout << "Output: " << outputs[0].DebugString() << endl;

    float *p = outputs[0].flat<float>().data();
    Mat output_mat;
    for (int i = start_index; i < end_index; i++) {
        Mat mat_row(cv::Size(128, 1), CV_32F, p + (i - start_index) * 128, Mat::AUTO_STEP);
        mat_training_tensors.push_back(mat_row);
        if (output_mat.empty())
            output_mat = mat_row;
        else
            vconcat(output_mat, mat_row, output_mat);
    }
    if (this->output_mat.empty())
        this->output_mat = output_mat.clone();
    else
        vconcat(this->output_mat, output_mat.clone(), this->output_mat);
    output_mat.release();
    cout << "Output Mat: " << this->output_mat.size() << endl;
    session->Close();
    delete session;
}

template<>
void FacenetClassifier<>::set_input_images(std::vector<Mat> input, std::vector<string> input_roi) {
    input_images = input;
    input_files = input_roi;
}

template<>
void FacenetClassifier<>::clear_input_images() {
    input_images.clear();
    input_files.clear();
}

template<>
void FacenetClassifier<>::release_batch_images(long start_index, long end_index) {
    long i;
    for (i = start_index; i < end_index; i++) {
        input_images[i].release();
    }
}

template<>
void FacenetClassifier<>::train() {
    classifier.train(mat_training_tensors, training_labels);
}

template
class Classifier<cv::ml::SVM>;

template
class Classifier<cv::ml::KNearest>;

template
class Classifier<cv::ml::ANN_MLP>;