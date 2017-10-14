#ifndef __FACENET_TF_H__
#define __FACENET_TF_H__
#include <iostream>
#include <fstream>
#include <string>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace cv;
using namespace ml;

class Label {
	public:
		int class_number;
		string class_name;
};

class FacenetClassifier {
	private:
		vector<Mat> input_images;
		vector<Label> class_labels;
		vector<int> output_labels;
		vector<string> input_files;
		string operation, model_path, svm_model_path, labels_file_path;
		tensorflow::GraphDef graph_def;
		tensorflow::Session *session;
		Tensor input_tensor, phase_tensor;
		Mat output_mat;
		Ptr<SVM> svm;
		fstream labels_file;
	public:
		FacenetClassifier (string operation, string model, string svm_model_path, string labels_file_path);
		void parse_images_path (string images_directory_path, int depth);
		void create_input_tensor ();
		void create_phase_tensor ();
		void run ();
		void save_svm ();
		void load_svm ();
		void save_labels ();
		void load_labels ();
		void predict_labels ();
};
#endif

