#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define PI 3.1415926
using namespace cv;
using namespace dnn;
using namespace std;

typedef struct
{
	cv::Rect rect;
	float prob;	
	vector<Point> kpt;
} face;

class YOLOv8_face
{
public:
	YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold);
	vector<face> detect(Mat frame);
private:
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw);
	const bool keep_ratio = true;
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	const int num_class = 1;  ///只有人脸这一个类别
	const int reg_max = 16;
	Net net;
	void softmax_(const float* x, float* y, int length);
	void generate_proposal(Mat out, vector<Rect>& boxes, vector<float>& confidences, vector< vector<Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<Point> landmark);
};

static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLOv8_face::YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->net = readNet(modelpath);
}

Mat YOLOv8_face::resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*padh = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOv8_face::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<Point> landmark)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

	//Get the label for the class name and its confidence
	string label = format("face:%.2f", conf);

	//Display the label at the top of the bounding box
	/*int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);*/
	putText(frame, label, Point(left, top-5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	for (int i = 0; i < 5; i++)
	{
		circle(frame, landmark[i], 4, Scalar(0, 255, 0), -1);
	}
}

void YOLOv8_face::softmax_(const float* x, float* y, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++)
	{
		y[i] /= sum;
	}
}

void YOLOv8_face::generate_proposal(Mat out, vector<Rect>& boxes, vector<float>& confidences, vector< vector<Point>>& landmarks, int imgh,int imgw, float ratioh, float ratiow, int padh, int padw)
{
	const int feat_h = out.size[2];
	const int feat_w = out.size[3];
	
	const int stride = (int)ceil((float)inpHeight / feat_h);
	const int area = feat_h * feat_w;
	float* ptr = (float*)out.data;
	float* ptr_cls = ptr + area * reg_max * 4;
	float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

	for (int i = 0; i < feat_h; i++)
	{
		for (int j = 0; j < feat_w; j++)
		{
			const int index = i * feat_w + j;
			int cls_id = -1;
			float max_conf = -10000;
			for (int k = 0; k < num_class; k++)
			{
				float conf = ptr_cls[k*area + index];
				if (conf > max_conf)
				{
					max_conf = conf;
					cls_id = k;
				}
			}
			float box_prob = sigmoid_x(max_conf);
			if (box_prob > this->confThreshold)
			{
				float pred_ltrb[4];
				float* dfl_value = new float[reg_max];
				float* dfl_softmax = new float[reg_max];
				for (int k = 0; k < 4; k++)
				{
					for (int n = 0; n < reg_max; n++)
					{
						dfl_value[n] = ptr[(k*reg_max + n)*area + index];
					}
					softmax_(dfl_value, dfl_softmax, reg_max);

					float dis = 0.f;
					for (int n = 0; n < reg_max; n++)
					{
						dis += n * dfl_softmax[n];
					}

					pred_ltrb[k] = dis * stride;
				}
				float cx = (j + 0.5f)*stride;
				float cy = (i + 0.5f)*stride;
				float xmin = max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);  ///还原回到原图,防止坐标越界
				float ymin = max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
				float xmax = min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
				float ymax = min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));
				Rect box = Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
				boxes.push_back(box);
				confidences.push_back(box_prob);

				vector<Point> kpts(5);
				for (int k = 0; k < 5; k++)
				{
					float x = ((ptr_kp[(k * 3)*area + index] * 2 + j)*stride - padw)*ratiow;  ///还原回到原图
					float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i)*stride - padh)*ratioh;
					///float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
					kpts[k] = Point(int(x), int(y));
				}
				landmarks.push_back(kpts);
			}
		}
	}
}


vector<face> YOLOv8_face::detect(Mat srcimg)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
	Mat blob;
	blobFromImage(dst, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	////net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	vector<Rect> boxes;
	vector<float> confidences;
	vector< vector<Point>> landmarks;
	float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

	generate_proposal(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	generate_proposal(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	generate_proposal(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	vector<face> face_boxes;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		const int idx = indices[i];
		face_boxes.push_back({ boxes[idx], confidences[idx], landmarks[idx] });
	}
	return face_boxes;
}

class GazeEstimation
{
public:
	GazeEstimation(string modelpath);
	void detect(Mat frame, face box);
	~GazeEstimation();  // 析构函数, 释放内存
	Mat draw_on(Mat srcimg, face box);
private:
	Mat normalize_(Mat img);
	const float mean_[3] = { 0.5, 0.5, 0.5 };
	const float std_[3] = { 0.5, 0.5, 0.5 };
	const int iris_idx_481[8] = { 248, 252, 224, 228, 232, 236, 240, 244 };
	const int num_eye = 481;
	const int input_size = 160;
	float* eye_kps; ///你也可以直接定义成静态数组 float eye_kps[481*2*3]; 这时候就不用考虑释放内存了
	Net net;
};

GazeEstimation::GazeEstimation(string modelpath)
{
	this->net = readNet(modelpath);
	this->eye_kps = new float[num_eye * 2 * 3]; ////左眼和右眼的关键点,每个点有3个元素
}

GazeEstimation::~GazeEstimation()
{
	delete[] this->eye_kps;
	this->eye_kps = nullptr;
}

Mat GazeEstimation::normalize_(Mat img)
{
	vector<cv::Mat> bgrChannels(3);
	split(img, bgrChannels);
	for (int c = 0; c < 3; c++)
	{
		bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1.0 / (255.0* std_[c]), (0.0 - mean_[c]) / std_[c]);
	}
	Mat m_normalized_mat;
	merge(bgrChannels, m_normalized_mat);
	return m_normalized_mat;
}

void GazeEstimation::detect(Mat img, face box)
{
	Point kps_right_eye = box.kpt[1];
	Point kps_left_eye = box.kpt[0];
	float center[2] = { float(kps_right_eye.x + kps_left_eye.x)*0.5, float(kps_right_eye.y + kps_left_eye.y)*0.5 };
	float _size = std::max(float(float(box.rect.width) / 1.5), fabsf(kps_right_eye.x - kps_left_eye.x))*1.5;
	float _scale = (float)this->input_size / _size;
	/////transform////
	float cx = center[0] * _scale;
	float cy = center[1] * _scale;
	Mat M = (Mat_<float>(2, 3) << _scale, 0, -cx + this->input_size * 0.5, 0, _scale, -cy + this->input_size * 0.5);
	Mat cropped;
	warpAffine(img, cropped, M, cv::Size(this->input_size, this->input_size));

	Mat rgbimg;
	cvtColor(cropped, rgbimg, COLOR_BGR2RGB);
	Mat normalized_mat = this->normalize_(rgbimg);
	Mat blob = blobFromImage(normalized_mat);

	/*// Check std values.
	Scalar std = Scalar(0.5, 0.5, 0.5);
	if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0)
	{
		// Divide blob by std.
		divide(blob, std, blob);
	}*/

	this->net.setInput(blob);
	vector<Mat> outs;
	////net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	float* opred = (float*)outs[0].data;////outs[0]的形状是(1,962,3)
	Mat IM;
	invertAffineTransform(M, IM);
	////trans_points////
	float scale = sqrt(IM.at<float>(0, 0)*IM.at<float>(0, 0) + IM.at<float>(0, 1)*IM.at<float>(0, 1));
	const int row = outs[0].size[1];  ////或者const int row = this->num_eye * 2;
	const int col = outs[0].size[2];
	for(int i = 0; i < row; i++)
	{
		this->eye_kps[i * 3] = IM.at<float>(0, 0)*opred[i * 3] + IM.at<float>(0, 1)*opred[i * 3 + 1] + IM.at<float>(0, 2);
		this->eye_kps[i * 3 + 1] = IM.at<float>(1, 0)*opred[i * 3] + IM.at<float>(1, 1)*opred[i * 3 + 1] + IM.at<float>(1, 2);
		this->eye_kps[i * 3 + 2] = opred[i * 3 + 2] * scale;
	}	
}

/*
输入参数eye的形状是(481,3)
输入参数iris_lms_idx的长度shi
输出theta_x_y_vec的长度是5, 分别是theta_x, theta_y, vec[0], vec[1], vec[2]
*/
void angles_and_vec_from_eye(const float* eye, const int* iris_lms_idx, float* theta_x_y_vec)
{
	float mean[3] = { 0,0,0 };
	for (int i = 0; i < 32; i++)
	{
		mean[0] += eye[i * 3];
		mean[1] += eye[i * 3 + 1];
		mean[2] += eye[i * 3 + 2];
	}
	mean[0] /= 32;
	mean[1] /= 32;
	mean[2] /= 32;

	float p_iris[8 * 3];
	for (int i = 0; i < 8; i++)
	{
		const int ind = iris_lms_idx[i];
		p_iris[i * 3] = eye[ind * 3] - mean[0];
		p_iris[i * 3 + 1] = eye[ind * 3 + 1] - mean[1];
		p_iris[i * 3 + 2] = eye[ind * 3 + 2] - mean[2];
	}

	float mean_p_iris[3] = { 0,0,0 };
	for (int i = 0; i < 8; i++)
	{
		mean_p_iris[0] += p_iris[i * 3];
		mean_p_iris[1] += p_iris[i * 3 + 1];
		mean_p_iris[2] += p_iris[i * 3 + 2];
	}
	mean_p_iris[0] /= 8;
	mean_p_iris[1] /= 8;
	mean_p_iris[2] /= 8;

	const float l2norm_p_iris = sqrt(mean_p_iris[0] * mean_p_iris[0] + mean_p_iris[1] * mean_p_iris[1] + mean_p_iris[2] * mean_p_iris[2]);
	theta_x_y_vec[2] = mean_p_iris[0] / l2norm_p_iris;  ///vec[0]
	theta_x_y_vec[3] = mean_p_iris[1] / l2norm_p_iris;  ///vec[1]
	theta_x_y_vec[4] = mean_p_iris[2] / l2norm_p_iris;  ///vec[2]

	/////angles_from_vec
	const float x = -theta_x_y_vec[4];
	const float y = theta_x_y_vec[3];
	const float z = -theta_x_y_vec[2];
	const float theta = atan2f(y, x);
	const float phi = atan2f(sqrt(x*x + y * y), z) - PI * 0.5;
	theta_x_y_vec[0] = phi;
	theta_x_y_vec[1] = theta;
}

Mat GazeEstimation::draw_on(Mat srcimg, face box)
{
	float rescale = 300.0 / (float)box.rect.width;
	Mat eimg;
	resize(srcimg, eimg, Size(), rescale, rescale);
	//// draw_item
	const int row = this->num_eye * 2;
	for (int i = 0; i < row; i++)
	{
		const float tmp = this->eye_kps[i * 3];
		this->eye_kps[i * 3] = this->eye_kps[i * 3 + 1] * rescale;
		this->eye_kps[i * 3 + 1] = tmp * rescale;
		this->eye_kps[i * 3 + 2] *= rescale;
	}
	/////angles_and_vec_from_eye
	const int slice = this->num_eye * 3;
	float theta_x_y_vec_l[5];
	angles_and_vec_from_eye(eye_kps, this->iris_idx_481, theta_x_y_vec_l);
	float theta_x_y_vec_r[5];
	angles_and_vec_from_eye(eye_kps + slice, this->iris_idx_481, theta_x_y_vec_r);
	const float gaze_pred[2] = { (theta_x_y_vec_l[0] + theta_x_y_vec_r[0])*0.5,(theta_x_y_vec_l[1] + theta_x_y_vec_r[1])*0.5 };
	const float diag = sqrt(float(eimg.rows*eimg.cols));

	float eye_pos_left[2] = { 0,0 };
	float eye_pos_right[2] = { 0,0 };
	for (int i = 0; i < 8; i++)
	{
		const int ind = this->iris_idx_481[i];
		eye_pos_left[0] += eye_kps[ind * 3];
		eye_pos_left[1] += eye_kps[ind * 3 + 1];
		eye_pos_right[0] += eye_kps[slice + ind * 3];
		eye_pos_right[1] += eye_kps[slice + ind * 3 + 1];
	}
	eye_pos_left[0] /= 8;
	eye_pos_left[1] /= 8;
	eye_pos_right[0] /= 8;
	eye_pos_right[1] /= 8;

	float dx = 0.4*diag*sinf(theta_x_y_vec_l[1]);
	float dy = 0.4*diag*sinf(theta_x_y_vec_l[0]);
	Point eye_left_a = Point(int(eye_pos_left[1]), int(eye_pos_left[0]));  ////左眼的箭头线的起始点坐标
	Point eye_left_b = Point(int(eye_pos_left[1] + dx), int(eye_pos_left[0] + dy));   ////左右的箭头线的终点坐标
	arrowedLine(eimg, eye_left_a, eye_left_b, Scalar(0, 0, 255), 5, LINE_AA, 0, 0.18);
	float yaw_deg_l = theta_x_y_vec_l[1] * (180 / PI);
	float pitch_deg_l = -theta_x_y_vec_l[0] * (180 / PI);

	dx = 0.4*diag*sinf(theta_x_y_vec_r[1]);
	dy = 0.4*diag*sinf(theta_x_y_vec_r[0]);
	Point eye_right_a = Point(int(eye_pos_right[1]), int(eye_pos_right[0]));  ////右眼的箭头线的起始点坐标
	Point eye_right_b = Point(int(eye_pos_right[1] + dx), int(eye_pos_right[0] + dy));  ////右眼的箭头线的终点坐标
	arrowedLine(eimg, eye_right_a, eye_right_b, Scalar(0, 0, 255), 5, LINE_AA, 0, 0.18);
	float yaw_deg_r = theta_x_y_vec_r[1] * (180 / PI);
	float pitch_deg_r = -theta_x_y_vec_r[0] * (180 / PI);

	resize(eimg, eimg, Size(srcimg.cols, srcimg.rows));
	////draw Yaw, Pitch
	string label = format("L-Yaw : %.2f", yaw_deg_l);
	putText(eimg, label, Point(int(eimg.cols-200), 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
	label = format("L-Pitch : %.2f", pitch_deg_l);
	putText(eimg, label, Point(int(eimg.cols - 200), 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
	label = format("R-Yaw : %.2f", yaw_deg_r);
	putText(eimg, label, Point(int(eimg.cols - 200), 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
	label = format("R-Pitch : %.2f", pitch_deg_r);
	putText(eimg, label, Point(int(eimg.cols - 200), 120), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
	return eimg;
}

int main()
{
	YOLOv8_face face_detector("weights/yolov8n-face.onnx", 0.45, 0.5);
	GazeEstimation gaze_predictor("weights/generalizing_gaze_estimation_with_weak_supervision_from_synthetic_views_1x3x160x160.onnx");

	string imgpath = "images/1.jpeg";
	Mat srcimg = imread(imgpath);
	vector<face> face_boxes = face_detector.detect(srcimg);

	Mat drawimg = srcimg.clone();
	for (int i = 0; i < face_boxes.size(); i++)
	{
		gaze_predictor.detect(srcimg, face_boxes[i]);
		drawimg = gaze_predictor.draw_on(drawimg, face_boxes[i]);
	}
	
	static const string kWinName = "Deep learning eye_gaze_estimation use OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, drawimg);
	waitKey(0);
	destroyAllWindows();

	////输入视频的
	/*string videopath = "images/test.mp4";
	string savepath = "result.mp4";
	VideoCapture vcapture(videopath);
	if (!vcapture.isOpened())
	{
		cout << "VideoCapture,open video file failed, " << videopath;
		return -1;
	}
	int height = vcapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int width = vcapture.get(cv::CAP_PROP_FRAME_WIDTH);
	VideoWriter vwriter;
	vwriter.open(savepath,
		cv::VideoWriter::fourcc('X', '2', '6', '4'),
		vcapture.get(cv::CAP_PROP_FPS),
		Size(width, height));

	Mat frame;
	while (vcapture.read(frame))
	{
		if (frame.empty())
		{
			cout << "cv::imread source file failed, " << videopath;
			return -1;
		}

		vector<face> face_boxes = face_detector.detect(frame);
		Mat drawimg = frame.clone();
		for (int i = 0; i < face_boxes.size(); i++)
		{
			gaze_predictor.detect(frame, face_boxes[i]);
			drawimg = gaze_predictor.draw_on(drawimg, face_boxes[i]);

		}
		vwriter.write(drawimg);

		static const string kWinName = "Deep learning eye_gaze_estimation use OpenCV";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, frame);
		int keyvalue = waitKey(1);
		if (keyvalue == 27)
		{
			break;
		}
	}
	destroyAllWindows();
	vwriter.release();
	vcapture.release();*/

	return 0;
}
