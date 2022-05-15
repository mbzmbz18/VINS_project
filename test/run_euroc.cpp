#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// 全局变量
const int nDelayTimes = 2;
string sData_path = "/home/dataset/EuRoC/MH-05/mav0/";	// 数据文件路径
string sConfig_path = "../config/";						// config文件路径
std::shared_ptr<System> pSystem;	// 系统指针

// 全局函数，用于发布IMU数据
void PubImuData()
{
	// 打开文件
	string sImu_data_file = sConfig_path + "MH_05_imu0.txt";
	cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;
	ifstream fsImu;
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open()) {
		cerr << "Failed to open imu file! " << sImu_data_file << endl;
		return;
	}
	std::string sImu_line;
	double dStampNSec = 0.0;
	Vector3d vAcc;	// 用于存储当前帧的加速度，三维向量
	Vector3d vGyr;	// 用于存储当前帧的角速度，三维向量
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) { // read imu data
		std::istringstream ssImuData(sImu_line);
		// 读入数据
		// 注：Euroc的
		ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
		// cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
		// 调用System::PubImuData()，用于发布IMU数据
		pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
		usleep(5000*nDelayTimes);
	}
	fsImu.close();
}

// 全局函数，用于发布图像数据
void PubImageData()
{
	// 打开文件
	string sImage_file = sConfig_path + "MH_05_cam0.txt";
	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;
	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open()) {
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}
	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;	// 用于存储当前图像文件名称
	// cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		// 当前图像的加载路径
		string imagePath = sData_path + "cam0/data/" + sImgFileName;
		// 加载图像
		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty()) {
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		pSystem->PubImageData(dStampNSec / 1e9, img);
		// cv::imshow("SOURCE IMAGE", img);
		// cv::waitKey(0);
		usleep(50000*nDelayTimes);
	}
	fsImage.close();
}

#ifdef __APPLE__
// support for MacOS
void DrawIMGandGLinMainThrd()
{
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open()) {
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;

	pSystem->InitDrawGL();
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty()) {
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		//pSystem->PubImageData(dStampNSec / 1e9, img);
		cv::Mat show_img;
		cv::cvtColor(img, show_img, CV_GRAY2RGB);
		if (SHOW_TRACK) {
			for (unsigned int j = 0; j < pSystem->trackerData[0].cur_pts.size(); j++) {
				double len = min(1.0, 1.0 *  pSystem->trackerData[0].track_cnt[j] / WINDOW_SIZE);
				cv::circle(show_img,  pSystem->trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
			}
			cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
			cv::imshow("IMAGE", show_img);
		  	// cv::waitKey(1);
		}
		pSystem->DrawGLFrame();
		usleep(50000*nDelayTimes);
	}
	fsImage.close();
} 
#endif

// 主函数，运行euroc数据集
int main(int argc, char **argv)
{
	// 查看程序输入
	if (argc != 3) {
		cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n" 
			 << "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
		return -1;
	}
	sData_path = argv[1];	// 得到数据集的路径
	sConfig_path = argv[2];	// 得到config文件的路径

	// 创建系统，即为指针pSystem赋值，这里new运算符会调用System构造函数
	pSystem.reset(new System(sConfig_path));
	
	// 创建线程，即后端，线程构造函数调用System类的成员函数System::ProcessBackEnd()以启动线程
	// 这里利用指针向线程传递类对象
	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);
	// 创建线程，用于发布IMU数据，线程构造函数调用全局函数PubImuData()以启动线程
	std::thread thd_PubImuData(PubImuData);
	// 创建线程，用于发布图像数据，线程构造函数调用全局函数PubImageData()以启动线程
	std::thread thd_PubImageData(PubImageData);

#ifdef __linux__	
	// 创建线程，用于可视化，线程构造函数调用System类的成员函数System::Draw()以启动线程
	// 这里利用指针向线程传递类对象
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
	// 可视化
	DrawIMGandGLinMainThrd();
#endif

	// 线程执行结束，join
	thd_PubImuData.join();
	thd_PubImageData.join();

	// thd_BackEnd.join();
	// thd_Draw.join();
	cout << "main end... see you ..." << endl;
	return 0;
}
