#include <iostream>
#include <string>
#include "facenet_tf.h"
#ifndef DEBUG
#define DEBUG cout << __FUNCTION__ << ":" << __LINE__ << endl;
#endif
using namespace std;

int main (int argc, char *argv[]) {
	if (argc < 7) {
		cout << "Usage:\n"
						"./facenet_classify <TRAIN|CLASSIFY>  <Path/To/TensorFlowModel> <Path/To/Image/Directory/Structure> <Path/To/Classifier/Model> <Classifier/Class/Labels> <SVM|KNN|MLP>\n"
						"Directory structure should be <class_name>/<image_files>\n"
						"Face Images Should be 160x160\n" << endl;
		return 1;
	}
	string operation, images_path, model_path, classifier_model_path, labels_file_path, ml_type;
	operation = string (argv[1]);
	model_path = string (argv[2]);
	images_path = string (argv[3]);
	classifier_model_path = string (argv[4]);
	labels_file_path = string (argv[5]);
	ml_type = string (argv[6]); //SVM or KNN or MLP
	long input_files_count;
	int i;
	long end_index;
	
	FacenetClassifier classifier = FacenetClassifier (operation, model_path, classifier_model_path, labels_file_path);
	classifier.batch_size = 1000;
	
	input_files_count = classifier.parse_images_path (images_path, 0);
	
	for (i = 0; i < input_files_count; i += classifier.batch_size) {
		cout << "Processing Images: " << i << " to " << i + classifier.batch_size - 1 << endl;
		if ((i + classifier.batch_size) > input_files_count)
			end_index = input_files_count;
		else	
			end_index = i + classifier.batch_size;
		classifier.load_input_images (i, end_index);
		classifier.preprocess_input_mat (i, end_index);
		classifier.create_input_tensor (i, end_index);
		classifier.create_phase_tensor ();
		classifier.run (i, end_index);
		classifier.release_batch_images (i, end_index);
	}
	if (ml_type == "SVM") {
		if (operation == "TRAIN") {
			classifier.save_svm ();
			classifier.save_labels ();
		}
		else if (operation == "CLASSIFY") {	
			classifier.load_svm ();
			classifier.load_labels ();
			classifier.predict_labels ();
		}
	}
	else if (ml_type == "KNN") {
		if (operation == "TRAIN") {
			classifier.save_knn ();
			classifier.save_labels ();
		}
		else if (operation == "CLASSIFY") {
			classifier.load_labels ();
			classifier.load_knn ();
			classifier.predict_knn_labels ();
		}
	}
	else if (ml_type == "MLP") {
		if (operation == "TRAIN") {
			classifier.save_mlp ();
			classifier.save_labels ();
		}
		else if (operation == "CLASSIFY") {
			classifier.load_labels ();
			classifier.load_mlp ();
			classifier.predict_mlp_labels ();
		}
	}
	return 0;
}

