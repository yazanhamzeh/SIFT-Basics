

#include <stdio.h>
#include <iostream>
#include <conio.h>    
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2\xfeatures2d.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
	/********************** Choose what to do **********************/
	bool ShowMatch = 1; //0: Show Keypoints with relative size and angle, 1: show matched keypoints.
	bool SearchMethod = 1; //0: Brute force search, 1: FLANN search
	bool DetectionMethod = 0;// 0: SIFT detector, 1: SURF Detector

	/**********************Load the two images **********************/
	// 'query' and 'train' are the notation used by the parameters in the 'match' function.
	Mat queryImg, trainImg;
	queryImg = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	trainImg = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	// Verify the images loaded successfully.
	if (queryImg.empty() && trainImg.empty())
	{
		printf("Can't read Both of the images\n");
		//cerr << cv::getBuildInformation() << endl;
		system("pause");
		return -1;
	}
	if (queryImg.empty())
	{
		cerr<<"Can't read image one\n"<<endl;
		return -1;
	}
	if (trainImg.empty())
	{
		cerr << "Can't read image two\n" << endl;
		return -1;
	}

	/*********************** Detection ***********************/
	std::vector<KeyPoint> queryKeypoints, trainKeypoints;
	Mat queryDescriptors, trainDescriptors;
	/*********************** DetectionMethod = 0--> Detect the keypoints using SIFT Detector ***********************/
	if (DetectionMethod == 0)
	{
		// Detect keypoints in both images.
		Ptr<SIFT> ptrSIFT = SIFT::create();
		ptrSIFT->detect(queryImg, queryKeypoints);
		ptrSIFT->detect(trainImg, trainKeypoints);
		// Print how many keypoints were found in each image.
		printf("Found %d and %d keypoints.\n", queryKeypoints.size(), trainKeypoints.size());
		// Compute the SIFT feature descriptors for the keypoints.
		// Multiple features can be extracted from a single keypoint, so the result is a
		// matrix where row 'i' is the list of features for keypoint 'i'.
		ptrSIFT->compute(queryImg, queryKeypoints, queryDescriptors);
		ptrSIFT->compute(trainImg, trainKeypoints, trainDescriptors);
		// Print some statistics on the matrices returned.
		Size size = queryDescriptors.size();
		printf("Query descriptors height: %d, width: %d, area: %d, non-zero: %d\n",
			size.height, size.width, size.area(), countNonZero(queryDescriptors));
		size = trainDescriptors.size();
		printf("Train descriptors height: %d, width: %d, area: %d, non-zero: %d\n",
			size.height, size.width, size.area(), countNonZero(trainDescriptors));
	}
	/*********************** DetectionMethod = 1--> Detect the keypoints using SURF Detector ***********************/
	else {		
		int minHessian = 400;
		Ptr<SURF> detector = SURF::create();
		detector->setHessianThreshold(minHessian);
		//std::vector<KeyPoint> queryKeypoints, trainKeypoints;
		detector->detectAndCompute(queryImg, Mat(), queryKeypoints, queryDescriptors);
		detector->detectAndCompute(trainImg, Mat(), trainKeypoints, trainDescriptors);
	}

	/*********************** Matching ***********************/
		/*********************** SearchMethod = 0--> search and match keypoints using Brute Force ***********************/
	if (SearchMethod == 0)
	// For each of the descriptors in 'queryDescriptors', find the closest 
	// matching descriptor in 'trainDescriptors' (performs an exhaustive search).
	// This seems to only return as many matches as there are keypoints. For each
	// keypoint in 'query', it must return the descriptor which most closesly matches a
	// a descriptor in 'train'?
	{
		BFMatcher matcher(NORM_L2);
		vector<DMatch> matches;
		matcher.match(queryDescriptors, trainDescriptors, matches);

		printf("Found %d matches.\n", matches.size());
		system("pause");
		// Draw the results. Displays the images side by side, with colored circles at
		// each keypoint, and lines connecting the matching keypoints between the two 
		// images.
		namedWindow("matches", 2);
		/************************ ShowMatch = 1--> Show all matches *********************** */
		if (ShowMatch == 1) {
			Mat img_matches;
			drawMatches(queryImg, queryKeypoints, trainImg, trainKeypoints, matches, img_matches);
			imshow("matches", img_matches);
			waitKey(0);
		}
	}
	/*********************** SearchMethod = 1--> search and match keypoints using FLANN matcher ***********************/
	else {
		
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(queryDescriptors, trainDescriptors, matches);
		double max_dist = 0; double min_dist = 250;
		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < queryDescriptors.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);
		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
		//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
		//-- small)
		std::vector< DMatch > good_matches;
		for (int i = 0; i < queryDescriptors.rows; i++)
		{
			if (matches[i].distance <= max(2 * min_dist, 0.02))
			{
				good_matches.push_back(matches[i]);
			}
		}
		/************************ ShowMatch = 1--> Show only good matches *********************** */
		
		if (ShowMatch == 1) {
			Mat img_matches;
			drawMatches(queryImg, queryKeypoints, trainImg, trainKeypoints,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			waitKey(0);
		
		//-- Show detected matches
		imshow("Good Matches", img_matches);
		}
		for (int i = 0; i < (int)good_matches.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
		}
		waitKey(0);

		}

	/************************ ShowMatch = 0--> Show Keypoints with relative size and angle *********************** */
	if (ShowMatch == 0) {
		Mat trainImgOut, queryImgOut;
		namedWindow("Train Image Keypoints", 0);
		drawKeypoints(trainImg, trainKeypoints, trainImgOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("Train Image Keypoints", trainImgOut);
		namedWindow("Query Image Keypoints", 0);
		drawKeypoints(queryImg, queryKeypoints, queryImgOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("Query Image Keypoints", queryImgOut);
		waitKey(0);
	}

	return 0;


}
