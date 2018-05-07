/*
Simple UDP Server
*/
//#include <thread>         // std::thread
#include <stdio.h>
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <iostream>
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

#pragma comment(lib,"ws2_32.lib") //Winsock Library

#define BUFLEN 512  //Max length of buffer
#define PORT 7777   //The port on which to listen for incoming data

#pragma comment(lib,"ws2_32.lib") //Winsock Library

#define SERVER "192.168.43.227"  //ip address of udp server
//#define SERVER "192.168.2.193"  //ip address of udp server
//#define SERVER "192.168.25.11"  //ip address of udp server

struct Circle
{
	Circle(int x, int y, int radius, int on) : X(x), Y(y), Radius(radius), On(on) {}
	int X, Y, Radius, On;
};

//========================================================

Scalar red;
float highestRadius;
RNG rng(12345);

Mat imgTmp;
Mat imgLines;

//Yellow
int iLowH_1 = 23;
int iHighH_1 = 38;

int iLowS_1 = 52;
int iHighS_1 = 255;

int iLowV_1 = 229;
int iHighV_1 = 255;

//Green
int iLowH_2 = 37;
int iHighH_2 = 88;

int iLowS_2 = 47;
int iHighS_2 = 255;

int iLowV_2 = 232;
int iHighV_2 = 255;

CascadeClassifier _faceCascade;
String _windowName = "Unity OpenCV Interop Sample";
VideoCapture _capture;
int _scale = 1;
Circle theCircle(0, 0, 0, 0);

int detectAndDrawCircle(VideoCapture cap, int iLowH, int iHighH, int iLowS, int iHighS, int iLowV, int iHighV, Scalar red, String windowName, String tWindowsName, bool shouldTryToDetect)
{
	if (!shouldTryToDetect)
		return 0;

	int returnValue = 0;
	Mat imgOriginal1;

	bool bSuccess_1 = cap.read(imgOriginal1); // read a new frame from video

	if (!bSuccess_1) //if not success, break loop
	{
		cout << "Cannot read a frame from video stream" << endl;
		return 0;
	}

	cv::flip(imgOriginal1, imgOriginal1, 1);
	//imshow("Original", imgOriginal);

	Mat imgHSV_1;
	cvtColor(imgOriginal1, imgHSV_1, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

	Mat imgThresholded_1;
	inRange(imgHSV_1, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded_1); //Threshold the image //morphological opening (removes small objects from the foreground)

	erode(imgThresholded_1, imgThresholded_1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(imgThresholded_1, imgThresholded_1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//morphological closing (removes small holes from the foreground)
	dilate(imgThresholded_1, imgThresholded_1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded_1, imgThresholded_1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	imshow(tWindowsName, imgThresholded_1); //show the thresholded image

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(imgThresholded_1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 5)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
		}
	}

	int index = -5;
	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(imgThresholded_1.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 100)
		{
			circle(imgOriginal1, center[i], (int)radius[i], red, 4, 8, 0);
			circle(imgOriginal1, center[i], 5, red, -1);

			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			circle(drawing, center[i], 20, color, 4, 8, 0);
			index = i;
			returnValue = 1;
		}
	}

	if (index >= 0)
	{
		if (center.size() > 0)
		{
			theCircle.X = center[index].x;
			theCircle.Y = center[index].y;
			theCircle.Radius = radius[index];
		}
	}

	imshow(windowName, imgOriginal1);

	return returnValue;
}

void Init()
{
	red = Scalar(0, 0, 255);

	string filename = "multicolor.mp4";

	_capture.open(0); //capture the video from webcam
	//_capture.open(filename); //capture the video from webcam

	if (!_capture.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return;
	}

	namedWindow("Control 1", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH_1, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH_1, 179);

	createTrackbar("LowS", "Control", &iLowS_1, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS_1, 255);

	createTrackbar("LowV", "Control", &iLowV_1, 255);//Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV_1, 255);

	namedWindow("Control 2", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH_2, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH_2, 179);

	createTrackbar("LowS", "Control", &iLowS_2, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS_2, 255);

	createTrackbar("LowV", "Control", &iLowV_2, 255);//Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV_2, 255);

	//Capture a temporary image from the camera
	imgTmp;
	_capture.read(imgTmp);

	//Create a black image with the size as the camera output
	imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);;
}

void Detect()
{
	Init();

	bool shouldTryToDetectColor1 = true;

	int count = 0;
	bool isFirstColor = true;
	int iLowH, iHighH, iLowS, iHighS, iLowV, iHighV;
	iLowH = iLowH_2;
	iHighH = iHighH_2;
	iLowS = iLowS_2;
	iHighS = iHighS_2;
	iLowV = iLowV_2;
	iHighV = iHighV_2;

	while (true)
	{
		int detectionReturnValue = detectAndDrawCircle(_capture, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, red, "Circle", "Threshold window", shouldTryToDetectColor1);

		if (detectionReturnValue == 0)
		{
			count++;

			if (count > 0)
			{
				if (!isFirstColor)
				{
					iLowH = iLowH_2;
					iHighH = iHighH_2;
					iLowS = iLowS_2;
					iHighS = iHighS_2;
					iLowV = iLowV_2;
					iHighV = iHighV_2;
				}
				else
				{
					iLowH = iLowH_1;
					iHighH = iHighH_1;
					iLowS = iLowS_1;
					iHighS = iHighS_1;
					iLowV = iLowV_1;
					iHighV = iHighV_1;
				}
				isFirstColor = !isFirstColor;
				count = 0;
			}
		}

		if (detectionReturnValue == 1 && !isFirstColor)
			theCircle.On = 1;
		else 
			theCircle.On = 0;

		//if (detectAndDrawCircle(_capture, iLowH_2, iHighH_2, iLowS_2, iHighS_2, iLowV_2, iHighV_2, red, "Circle 2", "Threshold window 2", shouldTryToDetectColor1) == 1)
		//{
		//	//cout << "Off" << endl;
		//	theCircle.On = 0;
		//}
		//else
		//{
		//	shouldTryToDetectColor1 = false;
		//	if (detectAndDrawCircle(_capture, iLowH_1, iHighH_1, iLowS_1, iHighS_1, iLowV_1, iHighV_1, red, "Circle 1", "Threshold window 1", true) == 0);
		//	{
		//		if (count < 1)
		//		{
		//			count++;
		//		}
		//		else
		//		{
		//			shouldTryToDetectColor1 = true;
		//			count = 0;
		//		}
		//	}
		//	//cout << "On" << endl;
		//	theCircle.On = 1;
		//}

		/*string text = to_string(theCircle.X) + "-" + to_string(theCircle.Y) + "-" + to_string(theCircle.Radius) + "-" + to_string(theCircle.On);
		cout << text << endl;*/

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
}

void parseString(string s, Circle *circle)
{
	std::string delimiter = "-";
	bool firstLoop = false;
	size_t pos = 0;
	std::string token;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		if (!firstLoop)
		{
			circle->X = std::stof(token, nullptr);
			firstLoop = true;
		}
		else
		{
			circle->Y = std::stof(token, nullptr);
		}
		s.erase(0, pos + delimiter.length());
	}

	circle->Radius = std::stof(s, nullptr);
}

void runClient()
{
	struct sockaddr_in si_other;
	int s, slen = sizeof(si_other);
	char buf[BUFLEN];
	char message[BUFLEN];
	WSADATA wsa;

	//Initialise winsock
	printf("\nInitialising Winsock...");
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		printf("Failed. Error Code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
	}
	printf("Initialised.\n");

	//create socket
	if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == SOCKET_ERROR)
	{
		printf("socket() failed with error code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
	}

	//setup address structure
	memset((char *)&si_other, 0, sizeof(si_other));
	si_other.sin_family = AF_INET;
	si_other.sin_port = htons(PORT);
	si_other.sin_addr.S_un.S_addr = inet_addr(SERVER);

	//start communication
	while (1)
	{
		//printf("Enter message : ");
		//gets(message);
		//cin >> message;
		//string text = "120.0-150.0-180.0";
		string text = to_string(theCircle.X) + "|" + to_string(theCircle.Y) + "|" + to_string(theCircle.Radius) + "|" + to_string(theCircle.On);
		strcpy_s(message, text.c_str());

		if (message != "0|0|0|0")
		{
			//send the message
			if (sendto(s, message, strlen(message), 0, (struct sockaddr *) &si_other, slen) == SOCKET_ERROR)
			{
				printf("sendto() failed with error code : %d", WSAGetLastError());
				cout << message << endl;
				exit(EXIT_FAILURE);
			}
		}

		cout << message << endl;

		//receive a reply and print it
		//clear the buffer by filling null, it might have previously received data
		memset(buf, '\0', BUFLEN);
		//try to receive some data, this is a blocking call
		/*if (recvfrom(s, buf, BUFLEN, 0, (struct sockaddr *) &si_other, &slen) == SOCKET_ERROR)
		{
		printf("recvfrom() failed with error code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
		}

		puts(buf);*/
	}

	closesocket(s);
	WSACleanup();
}

int main()
{
	//Init();
	thread first(Detect);
	//Detect();
	runClient();

	return 0;
}