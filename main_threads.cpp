#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <exception>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/aruco.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <cmath>

//Color definitons and Constant assigments
//Renk ve Sabit atamalar�
const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;   //  This threshold used for non-maximum suppression to remove overlapping bounding boxes
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


class VideoCapture {
public:
    VideoCapture(const std::string& camera_link)
        : camera_link(camera_link), stop_thread(false) {
        camera_open();
        reader_thread = std::thread(&VideoCapture::_reader, this);
    }

    ~VideoCapture() {
        stop_thread = true;
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
        if (cap.isOpened()) {
            cap.release();
        }
    }

    cv::Mat read() {
        /*std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !q.empty() || stop_thread; });*/

        if (!q.empty()) {
            cv::Mat frame = q.front();
            q.pop();
            return frame;
        }
        return cv::Mat();
    }

private:
    std::string camera_link;
    cv::VideoCapture cap;
    std::queue<cv::Mat> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread reader_thread;
    bool stop_thread;

    void camera_open() {
        cap.open(camera_link);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error opening video stream or file");
        }
        else {
            std::cout << "Camera opened " << std::endl;
        }
    }

    void _reader() {
        try {
            while (!stop_thread) {
                if (!cap.isOpened()) {
                    cap.release();
                    camera_open();
                    continue;
                }

                cv::Mat frame;
                bool ret = cap.read(frame);

                if (!ret) {
                    cap.release();
                    camera_open();
                    continue;
                }

                cv::resize(frame, frame, cv::Size(640, 480));

                //std::unique_lock<std::mutex> lock(mtx);
                if (q.size() < 5) {
                    q.push(frame);
                }

                cv.notify_all();
                
            }
        }
        catch (const std::exception& e) {
            if (cap.isOpened()) {
                cap.release();
            }
            std::cerr << "Video Capture error: " << e.what() << std::endl;
        }
    }
};





bool is_blurred(const cv::Mat& frame, double threshold = 100.0)
{
    try
    {
        // Use Sobel filter to detect edges
        cv::Mat sobelx, sobely, sobel;

        cv::Sobel(frame, sobelx, CV_64F, 1, 0, 5);
        cv::Sobel(frame, sobely, CV_64F, 0, 1, 5);

        cv::magnitude(sobelx, sobely, sobel);

        double mean_sobel = cv::mean(sobel)[0];
        return mean_sobel < threshold;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Blur detection error: " << e.what() << std::endl;
        return false; // or handle the error as needed
    }
}

std::vector<std::string> load_class_list()
{
    //Reading a list of class names from the file which in "Models/classes.txt" and keep them in a vector
    // S�n�f isimleri Models dosyas�ndaki text file'dan al�n�r

    std::vector<std::string> class_list;
    std::ifstream ifs("D:/Assigments/Human_detection_cpp/Models/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        std::cout << "Classes adding :" << line << std::endl;
        class_list.push_back(line);
    }
    return class_list;
}


void load_net(cv::dnn::Net& net, bool is_cuda)
{
    //Loading yolov5s onnx model
    // E�itilmi� Onnx modeli cekilir
    try {

        auto result = cv::dnn::readNet("D:/Assigments/Human_detection_cpp/Models/yolov5s_simplified.onnx");
        if (is_cuda)
        {
            std::cout << "Using CUDA\n";
            result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        }
        else
        {
            std::cout << "CPU Mode " << std::endl;
            result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        net = result;
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}


cv::Mat format_yolov5(const cv::Mat& source) {
    try {
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));

        return result;
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}




void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    //try {

    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);



    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
    /*}
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }*/

}


int main() {
    std::string camera_link = "D:/Assigments/Human_detection_cpp/running.mp4"; // Replace with your camera link or file path
    VideoCapture capture(camera_link);

    // Loads the list of class names using load_class_list()
    std::vector<std::string> class_list = load_class_list();

    for (int i = 0; i < class_list.size(); i++) {
        std::cout << " Classes : " << class_list[i] << std::endl;
    }

    // Opens a video file for processing.
    cv::Mat current_frame, previous_frame, prev_frame_gray, current_frame_gray;

    // Use CUDA if existing
    bool is_cuda = false; // argc > 1 && strcmp(argv[1], "cuda") == 0;

    // Loading YOLOv5 model using load_net()
    cv::dnn::Net net;

    load_net(net, is_cuda);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    // Calculate the new dimensions while preserving the aspect ratio
    int targetWidth = 640; // Adjust the desired width
    previous_frame = capture.read();
    cv::resize(previous_frame, previous_frame, cv::Size(640, 480));
    cv::cvtColor(previous_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    int threshold = 50000;
    while (true) {
        current_frame = capture.read();
        if (current_frame.empty())
        {
            std::cout << "Media finished\n";
            break;
        }

        cv::resize(current_frame, current_frame, cv::Size(640, 480));
        cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);

        cv::Mat absolute_difference;
        cv::absdiff(current_frame_gray, prev_frame_gray, absolute_difference);

        int diff = cv::sum(absolute_difference)[0];

        if (diff > threshold) {
            frame_count++;
            total_frames++;
            if (!is_blurred(current_frame)) {
                std::vector<Detection> output;
                detect(current_frame, net, output, class_list);
                int detections = output.size();
                for (int i = 0; i < detections; ++i)
                {
                    auto detection = output[i];
                    auto box = detection.box;
                    auto classId = detection.class_id;
                    const auto color = colors[classId % colors.size()];
                    cv::rectangle(current_frame, box, color, 3);

                    cv::rectangle(current_frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                    cv::putText(current_frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }


            }
        }

        if (frame_count >= 30)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(current_frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", current_frame);

        int key = cv::waitKey(1);
        if (key == 27) // ESC to exit
        {
            break;
        }

        previous_frame = current_frame;
    }

    return 0;
}
