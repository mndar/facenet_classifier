#include "facenet_tf.h"
#include <dirent.h>
#include <string.h>

Ptr<ml::KNearest>  k_nearest(ml::KNearest::create());
Ptr<ml::ANN_MLP> 	ann;
#define K_VALUE 1
#define N_LAYERS 3
FacenetClassifier::FacenetClassifier (string operation, string model_path, string classifier_model_path, string labels_file_path) {
	this->operation = operation;
	this->model_path = model_path;
	this->classifier_model_path = classifier_model_path;
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
	class_labels.clear ();
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
	cout << class_labels.size() << endl;
}

void FacenetClassifier::preprocess_input_mat (long start_index, long end_index) {
	long i;
	for (i = start_index; i < end_index; i++) {
		Mat &image = input_images[i];
		//mean and std
		cvtColor(image,image, CV_RGB2BGR);
		cv::Mat temp = image.reshape(1, image.rows * 3);
		cv::Mat     mean3;
		cv::Mat     stddev3;
		cv::meanStdDev(temp, mean3, stddev3);

		double mean_pxl = mean3.at<double>(0);
		double stddev_pxl = stddev3.at<double>(0);
		cv::Mat image2;
		image.convertTo(image2, CV_64FC1);
		image = image2;
		image = image - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
		image = image / stddev_pxl;
	}
}

long FacenetClassifier::parse_images_path (string directory_path, int depth) {
	cout << "Parsing Directory: " << directory_path << endl;
	DIR *dir;
	struct dirent *entry;
	static int class_count = -1;
	string class_name;
	string file_name, file_path;
	if ((dir = opendir (directory_path.c_str())) != NULL) {
	
		while ((entry = readdir (dir)) != NULL) {
			if (entry->d_type == DT_DIR && strcmp (entry->d_name, ".") !=0 && strcmp (entry->d_name, "..") !=0) {
				class_name = string (entry->d_name);
				class_count++;
				parse_images_path (directory_path + "/" + class_name, depth+1);				
				Label label;
				label.class_number = class_count;
				label.class_name = class_name;
				class_labels.push_back(label);
			}
			else if (entry->d_type != DT_DIR) {
				file_name = string (entry->d_name);
				file_path = directory_path + "/" + file_name;
				cout << file_path << ":" << class_count << endl;					
				output_labels.push_back (class_count); //For Training
				mat_training_ints.push_back (class_count);
				input_files.push_back (file_path); //For Classification
			}
		}
		closedir (dir);
	}
	return input_files.size ();
}

void FacenetClassifier::load_input_images (long start_index, long end_index) {
	long i;
	for (i = start_index; i < end_index; i++) {
		cout << "Loading Image: " << input_files[i] << endl;
		Mat image = imread (input_files[i].c_str());
		if (image.empty() || image.rows !=160 || image.cols !=160)
			cout << "Ignoring Image " + input_files[i] << endl;
		input_images.push_back (image);
	}
}

void FacenetClassifier::create_input_tensor (long start_index, long end_index) {
	cout << "Using " << input_images.size() << " images" << endl;
	cout << "Start Index:" << start_index << " End Index:" << end_index << endl;
	Tensor input_tensor(DT_FLOAT, TensorShape({(int) (end_index - start_index), 160, 160, 3}));
	// get pointer to memory for that Tensor
	float *p = input_tensor.flat<float>().data();
	int i;
	
	for (i = 0; i < (end_index - start_index) ; i++) {
		// create a "fake" cv::Mat from it 

		Mat camera_image(160, 160, CV_32FC3, p + i*160*160*3);	
		input_images[i + start_index].convertTo(camera_image, CV_32FC3);
	}
	cout << input_tensor.DebugString() << endl;
	this->input_tensor = Tensor (input_tensor);
}

void FacenetClassifier::create_phase_tensor () {
	tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
	phase_tensor.scalar<bool>()() = false;
	this->phase_tensor = Tensor (phase_tensor);
}

void FacenetClassifier::run (long start_index, long end_index) {
	string input_layer = "input:0";
	string phase_train_layer = "phase_train:0";
	string output_layer = "embeddings:0";
	std::vector<tensorflow::Tensor> outputs;
	std::vector<std::pair<string, tensorflow::Tensor>> feed_dict = {
		  {input_layer, input_tensor},  
		  {phase_train_layer, phase_tensor},
	};    
	tensorflow::SessionOptions options;
	tensorflow::NewSession (options, &session);
	session->Create (graph_def);
	cout << "Input Tensor: " << input_tensor.DebugString() << endl;
	Status run_status = session->Run(feed_dict, {output_layer}, {} , &outputs);
	if (!run_status.ok()) {
		  LOG(ERROR) << "Running model failed: " << run_status << "\n";
		  return;
	}
	cout << "Output: " << outputs[0].DebugString() << endl;

	float *p = outputs[0].flat<float>().data();
	Mat output_mat;
	for (int i = start_index; i < end_index; i++) {		
		Mat mat_row (cv::Size (128, 1), CV_32F, p + (i - start_index)*128, Mat::AUTO_STEP);
		mat_training_tensors.push_back (mat_row);
		if (output_mat.empty())
			output_mat = mat_row;
		else
			vconcat (output_mat, mat_row, output_mat);
	}
	if (this->output_mat.empty())
		this->output_mat = output_mat.clone ();
	else
		vconcat (this->output_mat, output_mat.clone (), this->output_mat);
	output_mat.release ();
	cout << "Output Mat: " << this->output_mat.size () << endl;
	session->Close ();
	delete session;
}

void FacenetClassifier::save_mlp () {
	
	ann = ann->create ();
	Mat train_data = mat_training_tensors;
	Mat train_labels = mat_training_ints;
	int nfeatures = mat_training_tensors.cols;
	cout << "Labels: " << train_labels << endl;
	int nclasses = class_labels.size();
	cout << "Classes: " << nclasses << endl;
	Mat_<int> layers(4,1);
	layers(0) = nfeatures;     // input
	layers(1) = nclasses * 8;  // hidden
	layers(2) = nclasses * 4;  // hidden
	layers(3) = nclasses;      // output, 1 pin per class.
	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,1,1);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
	// ann requires "one-hot" encoding of class labels:
	Mat train_classes = Mat::zeros(train_data.rows, nclasses, CV_32FC1);

	for(int i=0; i<train_classes.rows; i++)
	{
		cout << i << ":" << train_labels.at<int>(i) << endl;
		train_classes.at<float>(i, train_labels.at<int>(i)) = 1.f;
	}

	cerr << train_data.size() << " " << train_classes.size() << endl;

	ann->train(train_data, ml::ROW_SAMPLE, train_classes);
	ann->save (classifier_model_path);
}

void FacenetClassifier::load_mlp () {
	ann = ann->load (classifier_model_path);	
}

void FacenetClassifier::predict_mlp_labels () {
	float prediction;
	int j;
	for (int i = 0; i < input_images.size(); i++) {
		Mat input_mat;
		Mat result;
		output_mat.row(i).convertTo (input_mat, CV_32F);
		prediction = ann->predict (input_mat, result);
		cout << input_files[i] << " " << result.size() << " " << prediction << " " << class_labels[prediction].class_name << " " << result.at<float>(0, prediction) << endl;
		
		input_mat.release ();
		result.release ();
	}
}

void FacenetClassifier::set_input_images (std::vector<Mat> input, std::vector<string> input_roi) {
        input_images = input;
        input_files = input_roi;
}

void FacenetClassifier::clear_input_images () {
        input_images.clear ();
        input_files.clear ();
}

void FacenetClassifier::release_batch_images (long start_index, long end_index) {
	long i;
	for (i = start_index; i < end_index; i++) {
		input_images[i].release();
	}
}

void FacenetClassifier::save_svm () {
	svm = SVM::create();
	svm->setKernel(SVM::LINEAR);
	svm->train(output_mat, ROW_SAMPLE, output_labels);
	svm->save(classifier_model_path);
	cout << "Training Complete" << endl;
}

void FacenetClassifier::save_knn () {
	Mat mat_training_floats;
	mat_training_ints.convertTo(mat_training_floats, CV_32FC3);
	
	FileStorage training_labels_file ("labels.xml", FileStorage::WRITE);
	training_labels_file << "classifications" << mat_training_floats;
	training_labels_file.release ();

	FileStorage training_tensors_file (classifier_model_path, FileStorage::WRITE);
	training_tensors_file << "images" << mat_training_tensors;
	training_tensors_file.release ();
}

void FacenetClassifier::load_svm () {
	svm = Algorithm::load<SVM>(classifier_model_path);
	cout << "SVM Model Loaded" << endl;
}

void FacenetClassifier::predict_labels () {
	Mat svm_response;
	svm->predict (output_mat, svm_response);
	//svm->predict (output_mat, svm_response);
	cout << "SVM Response: " << svm_response << endl;
  for (int i = 0 ; i < svm_response.rows; i++) {

  	//int class_number = (int) svm_response.at<float>(i,0);
  	cout << input_files[i] << ": " << svm_response.at<float>(i) << endl;
    //cout << input_files[i] << ": " << class_labels[class_number-1].class_name << endl;
  }
}

void FacenetClassifier::load_knn () {
	FileStorage labels_knn("labels.xml", FileStorage::READ);
	labels_knn["classifications"] >> mat_training_ints;
	labels_knn.release();
	cout << "KNN Labels: " << mat_training_ints.size() << endl;
	cout << "Reading " << model_path << endl;
	FileStorage tensors (classifier_model_path, FileStorage::READ);
	tensors["images"] >> mat_training_tensors;
	tensors.release();
	cout << "KNN Tensors: " << mat_training_tensors.size() << endl;
	k_nearest->setDefaultK (1);
	cout << "Train: " << k_nearest->train (mat_training_tensors, ml::ROW_SAMPLE, mat_training_ints) << endl;
}

void FacenetClassifier::predict_knn_labels () {
	Mat current_class(0, 0, CV_32F);
	Mat current_response(0, 0, CV_32F);
	Mat current_distance (0, 0, CV_32F);
	float current_class_float, response, distance;
	float prediction;
	int j;
	cout << "Input Images: " << input_images.size() << endl;
	cout << "mat_training_tensors: " << mat_training_tensors.size () << endl;
	cout << "Input Files: " << input_files.size () << endl;
	for (int i = 0; i < input_images.size(); i++) {
		Mat input_mat, input_mat_flattened;
		output_mat.row(i).convertTo (input_mat, CV_32FC3);
		input_mat_flattened = input_mat.reshape (0,1);
		
		prediction = k_nearest->findNearest (input_mat_flattened, K_VALUE, current_class, current_response, current_distance);
		current_class_float = (float) current_class.at<float>(0,0);
		response = (float) current_response.at<float>(0,0);
		distance = (float) current_distance.at<float>(0,0);
		cout << "Output Index: " << i << endl;
		cout << mat_training_tensors.row(i).size() << " " << prediction << " " << input_files[i] << ": " << current_class_float << " " << distance << endl;
	}
}

