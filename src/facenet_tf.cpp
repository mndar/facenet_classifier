#include "facenet_tf.h"
#include <dirent.h>
#include <string.h>
FacenetClassifier::FacenetClassifier (string operation, string model_path, string svm_model_path, string labels_file_path) {
	this->operation = operation;
	this->model_path = model_path;
	this->svm_model_path = svm_model_path;
	this->labels_file_path = labels_file_path;
	ReadBinaryProto(tensorflow::Env::Default(), model_path.c_str(), &graph_def);
	tensorflow::SessionOptions options;
	tensorflow::NewSession (options, &session);
	session->Create (graph_def);
	
}

void FacenetClassifier::save_labels () {
	cout << "Saving Labels To " << labels_file_path << endl;
	labels_file.open (labels_file_path, fstream::out);

	for (Label label: class_labels) {
		labels_file << label.class_number << " " << label.class_name << endl;
	}
	labels_file.close ();
	cout << "Done Saving labels" << endl;
}

void FacenetClassifier::load_labels () {
	labels_file.open (labels_file_path, fstream::in);
	int count = 0;
	while (true) {
		if (labels_file.eof())
			break;
		Label label;
		labels_file >> label.class_number >> label.class_name;
		class_labels.push_back (label);
		count++;
	}
	labels_file.close ();
}

void FacenetClassifier::parse_images_path (string directory_path, int depth) {
	cout << "Parsing Directory: " << directory_path << endl;
	DIR *dir;
	struct dirent *entry;
	static int class_count = 0;
	string class_name;
	string file_name, file_path;
	if ((dir = opendir (directory_path.c_str())) != NULL) {
	
		while ((entry = readdir (dir)) != NULL) {
			if (entry->d_type == DT_DIR && strcmp (entry->d_name, ".") !=0 && strcmp (entry->d_name, "..") !=0) {
				class_count++;
				class_name = string (entry->d_name);
				parse_images_path (directory_path + "/" + class_name, depth+1);				
				Label label;
				label.class_number = class_count;
				label.class_name = class_name;
				class_labels.push_back(label);
			}
			else if (entry->d_type != DT_DIR) {
				file_name = string (entry->d_name);
				file_path = directory_path + "/" + file_name;
				Mat image = imread (file_path);
				if (image.empty() || image.rows !=160 || image.cols !=160)
					cout << "Ignoring Image " + file_path << endl;
				else {
					cout << file_path << ":" << class_count << endl;
					input_images.push_back (image);
					output_labels.push_back (class_count); //For Training
					input_files.push_back (file_path); //For Classification
				}
			}
		}
		closedir (dir);
	}

}

void FacenetClassifier::create_input_tensor () {
	cout << "Using " << input_images.size() << " images" << endl;

	Tensor input_tensor(DT_FLOAT, TensorShape({(int) input_images.size(), 160, 160, 3}));
	// get pointer to memory for that Tensor
	float *p = input_tensor.flat<float>().data();
	int i;
	
	for (i = 0; i < input_images.size(); i++) {
		// create a "fake" cv::Mat from it 
		Mat camera_image(160 , 160, CV_32FC3, p + i*160*160*3);	
		input_images[i].convertTo(camera_image, CV_32FC3);

	}
	cout << input_tensor.DebugString() << endl;
	this->input_tensor = input_tensor;
}

void FacenetClassifier::create_phase_tensor () {
	tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
	phase_tensor.scalar<bool>()() = true;
	this->phase_tensor = phase_tensor;
}

void FacenetClassifier::run () {
	string input_layer = "input:0";
	string phase_train_layer = "phase_train:0";
	string output_layer = "embeddings:0";
	std::vector<tensorflow::Tensor> outputs;
	std::vector<std::pair<string, tensorflow::Tensor>> feed_dict = {
		  {input_layer, input_tensor},  
		  {phase_train_layer, phase_tensor},
	};    
	cout << "Input Tensor: " << input_tensor.DebugString() << endl;
	Status run_status = session->Run(feed_dict, {output_layer}, {} , &outputs);
	if (!run_status.ok()) {
		  LOG(ERROR) << "Running model failed: " << run_status << "\n";
		  return;
	}
	cout << "Output: " << outputs[0].DebugString() << endl;

	float *p = outputs[0].flat<float>().data();
	Mat output_mat;
	for (int i = 0; i < input_images.size(); i++) {
		Mat mat_row (cv::Size (128, 1), CV_32F, p + i*128, Mat::AUTO_STEP);
		if (i == 0)
			output_mat = mat_row;
		else
			vconcat (output_mat, mat_row, output_mat);
	}
	this->output_mat = output_mat;

}

void FacenetClassifier::save_svm () {
	svm = SVM::create();
	svm->setKernel(SVM::LINEAR);
	svm->train(output_mat, ROW_SAMPLE, output_labels);
	svm->save(svm_model_path);
	cout << "Training Complete" << endl;
}

void FacenetClassifier::load_svm () {
	svm = Algorithm::load<SVM>(svm_model_path);
	cout << "SVM Model Loaded" << endl;
}

void FacenetClassifier::predict_labels () {
	Mat svm_response;
	svm->predict (output_mat, svm_response, StatModel::RAW_OUTPUT);
	//svm->predict (output_mat, svm_response);
	cout << "SVM Response: " << svm_response << endl;
  for (int i = 0 ; i < svm_response.rows; i++) {

  	//int class_number = (int) svm_response.at<float>(i,0);
  	cout << input_files[i] << ": " << svm_response.at<float>(i) << endl;
    //cout << input_files[i] << ": " << class_labels[class_number-1].class_name << endl;
  }
}
