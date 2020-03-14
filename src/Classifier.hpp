//
// Created by prostoichelovek on 09.03.2020.
//

#ifndef FACENET_CLASSIFIER_HPP
#define FACENET_CLASSIFIER_HPP


#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

namespace Facenet {

    template<typename Classifier_t>
    class Classifier {
    public:
        float threshold = 0.7; // it only works with MLP so far
        bool ok = false;

        Classifier();

        void save(std::string file);

        bool load(std::string file);

        float predict(const cv::Mat &input);

        void train(const cv::Mat &samples, const std::vector<int> &labels);

    private:
        cv::Ptr<Classifier_t> classifier;
    };

    template<typename Classifier_t>
    inline Classifier<Classifier_t>::Classifier() : classifier(Classifier_t::create()) {

    }

    template<typename Classifier_t>
    inline void Classifier<Classifier_t>::save(std::string file) {
        classifier->save(file);
    }

    template<typename Classifier_t>
    inline bool Classifier<Classifier_t>::load(std::string file) {
        try {
            classifier = cv::Algorithm::load<Classifier_t>(file);
            ok = true;
        } catch (std::exception &e) {
            LOG(ERROR) << "Cannot load classifier from " << file << ": " << e.what() << std::endl;
            ok = false;
        }
        return ok;
    }

    template<typename Classifier_t>
    inline float Classifier<Classifier_t>::predict(const cv::Mat &input) {
        return classifier->predict(input);
    }

    template<>
    inline float Classifier<cv::ml::ANN_MLP>::predict(const cv::Mat &input) {
        cv::Mat results;
        float r = classifier->predict(input, results);
        if (results.at<float>(r) < threshold) {
            r = -1;
        }
        // std::cout << std::endl << results << std::endl;
        return r;
    }

    template<>
    inline float Classifier<cv::ml::KNearest>::predict(const cv::Mat &input) {
        cv::Mat current_class(0, 0, CV_32F);
        return classifier->findNearest(input, 2, current_class);
    }

    template<typename Classifier_t>
    inline void Classifier<Classifier_t>::train(const cv::Mat &samples, const std::vector<int> &labels) {
        classifier->train(samples, cv::ml::ROW_SAMPLE, labels);
    }

    template<>
    inline void Classifier<cv::ml::ANN_MLP>::train(const cv::Mat &samples, const std::vector<int> &labels) {
        int nfeatures = samples.cols;
        auto labels_copy = labels;
        std::sort(labels_copy.begin(), labels_copy.end());
        int nclasses = std::unique(labels_copy.begin(), labels_copy.end()) - labels_copy.begin();
        cv::Mat_<int> layers(4, 1);
        layers(0) = nfeatures;     // input
        layers(1) = nclasses * 8;  // hidden
        layers(2) = nclasses * 4;  // hidden
        layers(3) = nclasses;      // output, 1 pin per class.
        classifier->setLayerSizes(layers);
        classifier->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
        classifier->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.0001));
        classifier->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

        // ann requires "one-hot" encoding of class labels:
        cv::Mat train_classes = cv::Mat::zeros(samples.rows, nclasses, CV_32FC1);
        for (int i = 0; i < train_classes.rows; i++) {
            train_classes.at<float>(i, labels.at(i)) = 1.f;
        }

        classifier->train(samples, cv::ml::ROW_SAMPLE, train_classes);
    }

}

#endif //FACENET_CLASSIFIER_HPP
