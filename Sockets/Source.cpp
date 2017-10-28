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

#define SERVER "192.168.2.193"  //ip address of udp server

struct Circle
{
	Circle(int x, int y, int radius) : X(x), Y(y), Radius(radius) {}
	int X, Y, Radius;
};

//========================================================

Scalar red;
float highestRadius;
RNG rng(12345);

Mat imgTmp;
Mat imgLines;

int iLowH;
int iHighH;

int iLowS;
int iHighS;

int iLowV;
int iHighV;

int iLastX;
int iLastY;

CascadeClassifier _faceCascade;
String _windowName = "Unity OpenCV Interop Sample";
VideoCapture _capture;
int _scale = 1;
Circle theCircle(0, 0, 0);


void Init()
{
	red = Scalar(0, 0, 255);

	string filename = "video2.mp4";

	//_capture.open(0); //capture the video from webcam
	_capture.open(filename); //capture the video from webcam

	if (!_capture.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return;
	}

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	iLowH = 16;
	iHighH = 40;

	iLowS = 54;
	iHighS = 252;

	iLowV = 169;
	iHighV = 255;

	//Create trackbars in "Control" window
	/*createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);*/

	iLastX = -1;
	iLastY = -1;

	//Capture a temporary image from the camera
	imgTmp;
	_capture.read(imgTmp);

	//Create a black image with the size as the camera output
	imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);;
}

void Detect()
{
	while (true)
	{
		Mat imgOriginal;

		bool bSuccess = _capture.read(imgOriginal); // read a new frame from video


		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		Mat imgHSV;
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded;
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image //morphological opening (removes small objects from the foreground)

		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing (removes small holes from the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//Calculate the moments of the thresholded image
		Moments oMoments = moments(imgThresholded);

		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;

		//("Thresholded Image", imgThresholded); //show the thresholded image

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(imgThresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() > 120)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
			}
		}

		//highestRadius = getHighestFloat(&radius);

		/// Draw polygonal contour + bonding rects + circles
		Mat drawing = Mat::zeros(imgThresholded.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			if (contours[i].size() > 120)
			{
				circle(imgOriginal, center[i], (int)radius[i], red, 4, 8, 0);
				circle(imgOriginal, center[i], 5, red, -1);
			}
		}

		if (center.size() > 0)
		{
			theCircle.X = center[0].x;
			theCircle.Y = center[0].y;
			theCircle.Radius = radius[0];
		}


		//string text = to_string(theCircle.X) + "-" + to_string(theCircle.Y) + "-" + to_string(theCircle.Radius);
		//cout << text << endl;

		//cv::flip(imgOriginal, imgOriginal, 1);
		imshow("Original", imgOriginal);

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
		string text = to_string(theCircle.X) + "-" + to_string(theCircle.Y) + "-" + to_string(theCircle.Radius);
		strcpy_s(message, text.c_str());

		//send the message
		if (sendto(s, message, strlen(message), 0, (struct sockaddr *) &si_other, slen) == SOCKET_ERROR)
		{
			printf("sendto() failed with error code : %d", WSAGetLastError());
			//cout << message << endl;
			exit(EXIT_FAILURE);
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
	Init();
	thread first(Detect);
	//Detect();
	runClient();

	return 0;
}