---
title: Panorama Stitching
date: 2023-05-27 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Panorama Stitching
  ]
pin: true
---

# 1. 직접 촬영한 영상 세 장으로 panorama stitching을 수행해볼 것
❑ 금일 실습 두 가지 방법을 각각 적용하고 분석할 것
❑ Tip. 주변 대상이 적어도 5m 이상 떨어져 있고 특징점이 많이 추출될 수있는 장면에서 수행할 것

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void ex_panorama_simple() {
	Mat img;
	vector<Mat> imgs;
	img = imread("D:/study/디영처/13주차/13주차 실습 자료/tech_left.jpg", IMREAD_COLOR);
	resize(img, img, Size(0, 0), 0.2, 0.2, INTER_LINEAR);
	imgs.push_back(img);
	img = imread("D:/study/디영처/13주차/13주차 실습 자료/tech_center.jpg", IMREAD_COLOR);
	resize(img, img, Size(0, 0), 0.2, 0.2, INTER_LINEAR);
	imgs.push_back(img);
	img = imread("D:/study/디영처/13주차/13주차 실습 자료/tech_right.jpg", IMREAD_COLOR);
	resize(img, img, Size(0, 0), 0.2, 0.2, INTER_LINEAR);
	imgs.push_back(img);
	Mat result;
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, false);
	Stitcher::Status status = stitcher->stitch(imgs, result);
	if (status != Stitcher::OK) {
		cout << "Can't stitch images, error code = " << int(status) << endl;
		exit(-1);
	}

	imshow("ex_panorama_simple_result", result);
	imwrite("panorama/ex_panorama_simple_resized_result.png", result);
	waitKey(0);
}

Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	// <Grayscale로 변환>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	// <특징점(key point) 추출>
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	// <특징점 시각화>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("img_kpts_l.png", img_kpts_l);
	imshow("img_kpts_r.png", img_kpts_r);
	imwrite("D:/study/디영처/13주차/13주차 실습 자료/panorama/img_kpts_l.png", img_kpts_l);
	imwrite("D:/study/디영처/13주차/13주차 실습 자료/panorama/img_kpts_r.png", img_kpts_r);

	// <기술자(descriptor) 추출>
	Ptr<SurfDescriptorExtractor> Extractor = 
								SURF::create(100, 4, 3, false, true);
	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	// <기술자를 이용한 특징점 매칭>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	// <매칭 결과 시각화>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("img_matches.png", img_matches);
	imwrite("D:/study/디영처/13주차/13주차 실습 자료/img_matches.png", img_matches);
	// <매칭 결과 정제>
	// 매칭 거리가 작은 우수한 매칭 결과를 정제하는 과정
	// 최소 매칭 거리의 3배 또는 우수한 매칭 결과 60이상 까지 정제
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;
	}
	printf("max_dist : %f \n", dist_max); // max는 사실상 불필요
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	// <우수한 매칭 결과 시각화>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("img_matches_good.png", img_matches_good);
	imwrite("D:/study/디영처/13주차/13주차 실습 자료/img_matches_good.png", img_matches_good);

	// <매칭 결과 좌표 추출>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);
	}

	// <매칭 결과로부터 homography 행렬을 추출>
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	// 이상치 제거를 위해 RANSAC 추가

	// <Homography 행렬을 이용해 시점 역변환>
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo,
		Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	imshow("img_warp", img_result);
	// 영상이 잘리는 것을 방지하기 위해 여유공간을 부여
	// 
	// <기준 영상과 역변환된 시점 영상 합체>
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
	img_l.copyTo(roi);
	// 검은 여백 잘라내기
	int cut_x = 0, cut_y = 0;
	for (int y = 0; y < img_pano.rows; y++) {
		for (int x = 0; x < img_pano.cols; x++) {
			if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
				img_pano.at<Vec3b>(y, x)[1] == 0 &&
				img_pano.at<Vec3b>(y, x)[2] == 0) {
				continue;
			}
			if (cut_x < x)cut_x = x;
			if (cut_y < y)cut_y = y;
		}
	}
	Mat img_pano_cut;
	img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
	imshow("img_pano_cut.png", img_pano_cut);
	imwrite("panorama/img_pano_cut.png", img_pano_cut);

	return img_pano_cut;
}

void ex_panorama() {
	Mat matImage1 = imread("D:/study/디영처/13주차/13주차 실습 자료/tech_center.jpg", IMREAD_COLOR);
	resize(matImage1, matImage1, Size(0, 0), 0.2, 0.2, INTER_LINEAR);
	Mat matImage2 = imread("D:/study/디영처/13주차/13주차 실습 자료/tech_left.jpg", IMREAD_COLOR);
	resize(matImage2, matImage2, Size(0, 0), 0.2, 0.2, INTER_LINEAR);
	Mat matImage3 = imread("D:/study/디영처/13주차/13주차 실습 자료/tech_right.jpg", IMREAD_COLOR);
	resize(matImage3, matImage3, Size(0, 0), 0.2, 0.2, INTER_LINEAR);
	if (matImage1.empty() || matImage2.empty() || matImage3.empty()) exit(-1);

	Mat result;
	flip(matImage1, matImage1, 1);
	flip(matImage2, matImage2, 1);
	result = makePanorama(matImage1, matImage2, 3, 60);
	flip(result, result, 1);
	result = makePanorama(result, matImage3, 3, 60);

	imshow("ex_panorama_result", result);
	imwrite("D:/study/디영처/13주차/13주차 실습 자료/ex_panorama_resized_result.png", result);
	waitKey(0);
}

int main() {
	//ex_panorama_simple();
	ex_panorama();
	destroyAllWindows();
	return 0;
}
```

## 1-2) 실험결과

### [실험에 사용한 사진]

![tech_left](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d8c8f9e1-fd64-403c-814a-de47ac1ceb5e)

![tech_center](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d230fc61-6071-4d02-abad-70ae234fa00e)

![tech_right](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/189cc6be-7640-4afe-b86b-08fb18feb3c3)

## [ex_panorama_simple()결과(Stitcher이용)]

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fb43865c-600a-4ea6-a84d-6e4084c86c01)

## [ex_panorama()결과(단계별 구현 방법)]

### [특징점 추출 결과(Left와 center는  stitching된 이후의 사진을 캡처하였습니다.]

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ab762cf1-234a-411a-a773-9bb5af53de07)

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/12cb57fb-6a96-4521-8b15-8159e7dbe11c)

### [특징점 매칭 결과]

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ba08e381-1ec7-4c74-8374-02521c6421cd)

### [정제 결과]

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fd89fef0-0ee5-4203-bed4-7c58eb7d824f)

### [시점 역 변환 결과]

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/cc22c097-7335-425d-b01b-ea0fc385d088)

### [Stitching 결과]

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/069fbcaa-0698-4880-8396-a247cc5fdca6)

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0e6cacc5-ac14-42df-93a4-83ccdb9613be)

## 1-3)구현 과정

- 직접 촬영한 사진의 크기가 커서 모두 resize()함수를 통해 사진의 크기를 줄여주었습니다. 여러 파라미터 값들 중에서 0.2가 가장 적절했습니다.
- **`void ex_panorama_simple()`** :Open CV의 Stitcher를 이용하여 Panorama Stitching을 수행하는 Function입니다. 다소 간단하게 구현을 할 수 있었습니다.
- **`Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches)`** :단계별로 Panorama Stitching을 진행하는 함수입니다. 먼저 SURF특징점을 추출합니다. 이는 각 3개의 image에서 유의미한 특징점을 추출하기 위한 과정입니다. 다음으로는 특징점을 매칭합니다. 여기서 올바르게 매칭되는 경우도 있지만, 올바르지 않게 매칭되는 경우도 있습니다. 따라서 올바르게 매칭되는 점을 고르기 위해, 즉 유의미한 매칭만 남기기 위해서 distance를 구해 Threshold를 만족하는 매칭 결과만 남기게 됩니다. 이를 이용하여 Stitching을 진행하게 됩니다. 이렇게 구한 유의미한 매칭 결과에서 homography행렬을 추출합니다. 이때 RANSAC알고리즘을 통해 Outlier를 방지해줍니다. 이후 Panorama Stitching을 진행할 수 있도록 H행렬에 맞는 시점으로 image를 변환한 후 영상을 합쳐주었습니다.
- **`void ex_panorama()`** :이는 3개의 영상을 stitching하기 위한 함수입니다.

## 1-4)결과 분석

1번은 Panorama Stitching을 3개의 이미지에 대해 진행하는 실습이었습니다. 코드는 실습 노트에 대부분 있어서 코드를 짜는 것은 어렵지 않았습니다. Open CV함수를 사용하여 간단하게 수행할 수 있는 것도 해보았고, 단계별로 구현하는 것도 해보았는데, 단계별로 구현하는 과정에서 Panorama Stitching에 대한 이해도가 매우 높아질 수 있었습니다. 각 구현과정을 코딩하고, 코딩한 결과를 위 결과 사진처럼 확인하였을 때, 실제로 Panorama Stitching을 진행하면 어떤 과정들이 순차적으로 진행되는지 직관적으로 알 수 있었습니다. 또한 각 단계별로 어떤 원리로 해당 단계가 동작하는지 파악할 수 있어서 이론적인 내용을 직접 확인할 수 있는 과제였습니다. 결과를 보았을 때, 모두 정상동작하는 것을 알 수 있었습니다. 특히 단계별로 동작 상태를 확인하였기 때문에, 최종 결과 사진뿐만 아니라 중간 과정들까지도 정상적으로 도출된 것을 확인할 수 있었습니다.

# 2. Book1.jpg, Book2.jpg, Book3.jpg가 주어졌을 때 Scene.jpg에서 이것들을찾아 아래의 그림처럼 윤곽을 찾아 그려주는 프로그램을 구현할 것
❑ SIFT 특징점 추출, brute force 매칭, findHomograpy()를 사용해 구현할 것
❑ 상세한 코드 설명과 주석을 첨부할 것

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Mat makeBookMatch(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	// <Grayscale로 변환>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	// <특징점(key point) 추출>
	Ptr<SiftFeatureDetector> Detector = SIFT::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	// <특징점 시각화>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("img_kpts_l.png", img_kpts_l);
	imshow("img_kpts_r.png", img_kpts_r);

	// 기술자(descriptor) 추출
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create(100, 4, 3, false, true);
	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	// <기술자를 이용한 특징점 매칭>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	// <매칭 결과 시각화>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("img_matches.png", img_matches);
	// <매칭 결과 정제>
	// 매칭 거리가 작은 우수한 매칭 결과를 정제하는 과정
	// 최소 매칭 거리의 3배 또는 우수한 매칭 결과 60이상 까지 정제
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;
	}
	printf("max_dist : %f \n", dist_max); // max는 사실상 불필요
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	// <우수한 매칭 결과 시각화>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("img_matches_good.png", img_matches_good);

	// <매칭 결과 좌표 추출>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);
	}

	// <매칭 결과로부터 homography 행렬을 추출>
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	// 이상치 제거를 위해 RANSAC 추가

	/*과제 2번 윤곽선을 그리기 위한 코드입니다.*/
	//책의 꼭지점을 알면 윤곽선을 그릴 수 있습니다. 따라서 지난 실습시간에 배운 perpectiveTransform에서 영상변환을 할 때,
	//좌표를 이용했던 특성을 이용하려고 합니다. 
	Mat img_result=img_r.clone();//윤곽선을 그릴 결과 image

	vector<Point2f> srcQuad(4);// 정면 시점의 책의 꼭지점 좌표를 저장할 변수
	srcQuad[0] = Point(0, 0);
	srcQuad[1] = Point(img_l.cols, 0);
	srcQuad[2] = Point(0, img_l.rows);
	srcQuad[3] = Point(img_l.cols, img_l.rows);//img_l이 book의 사진이고 가로 세로의 길이가 각 꼭지점 좌표입니다.

	vector<Point2f> dstQuad(4);//영상 변환된 책의 꼭지점을 저장할 변수입니다.
	Mat homo_transpose = mat_homo.inv();//H행렬을 perspective transform에 적용하기 위한 역행렬을 구해줍니다.
	perspectiveTransform(srcQuad, dstQuad, homo_transpose);//이를 통해 transform된 책의 꼭지점 좌표를 dstQuad에 저장합니다.

	line(img_result, dstQuad[0], dstQuad[1], Scalar(0, 0, 255), 4);//각 꼭지점을 line함수를 통해 선을 그어 윤곽선을 그려줍니다.
	line(img_result, dstQuad[1], dstQuad[3], Scalar(0, 0, 255), 4);
	line(img_result, dstQuad[3], dstQuad[2], Scalar(0, 0, 255), 4);
	line(img_result, dstQuad[2], dstQuad[0], Scalar(0, 0, 255), 4);

	return img_result;
}

int main() {
	Mat src_img = imread("D:/study/디영처/13주차/13주차 실습 자료/Scene.jpg", IMREAD_COLOR);
	resize(src_img, src_img, Size(0, 0), 0.7, 0.7, INTER_LINEAR);
	Mat book1 = imread("D:/study/디영처/13주차/13주차 실습 자료/book1.jpg", IMREAD_COLOR);
	resize(book1, book1, Size(0, 0), 0.7, 0.7, INTER_LINEAR);
	Mat book2 = imread("D:/study/디영처/13주차/13주차 실습 자료/book2.jpg", IMREAD_COLOR);
	resize(book1, book1, Size(0, 0), 0.7, 0.7, INTER_LINEAR);
	Mat book3 = imread("D:/study/디영처/13주차/13주차 실습 자료/book3.jpg", IMREAD_COLOR);
	resize(book1, book1, Size(0, 0), 0.7, 0.7, INTER_LINEAR);

	Mat dst_img1, dst_img2, dst_img3;
	dst_img1 = makeBookMatch(book1, src_img, 3, 60);
	dst_img2 = makeBookMatch(book2, src_img, 3, 60);
	dst_img3 = makeBookMatch(book3, src_img, 3, 60);
	imshow("book1", dst_img1);
	imshow("book2", dst_img2);
	imshow("book3", dst_img3);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
```

## 2-2) 실험 결과

### [특이점 추출 결과]

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6663f44e-94e1-4851-808a-ed1a435c0745)

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1ba1b7c7-c95d-479b-bc72-ea49414074ba)

### [특이점 매칭 결과]

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/72e74c91-259a-4c0c-9851-d0ff48bd2703)

### [정제 결과]

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/41e139c9-9423-4a01-af5b-7ea6cc2e274c)

### [최종 결과 (book3 윤곽선)]

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8fed9b1c-6639-4943-a9be-987fa0cca13c)

### [book1, boo2에 대한 결과]

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1631ae5e-0969-43a9-ac5f-8c7bfd6100e6)

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/78e0ebf1-9e8b-4795-b59c-c10a34a440ea)

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/994479f8-2c3a-4fe5-94fc-1ad3184e9fbe)

## 2-3) 구현 과정(코드 설명)

- **`Mat makeBookMatch(Mat img_l, Mat img_r, int thresh_dist, int min_matches)`** :
makeBookMatch함수는 특징점을 매칭하여 Homograpy행렬을 추출하는 것까지는 이전 과제1과 동일한 코드를 가지고 있습니다. 책만 있는 사진과, 다양한 사물들이 존재하는 사진을 서로 매칭하여 정제합니다. 이를 통해 특정 책을 매칭할 수 있습니다.
이후 코드가 1번과 다릅니다. 책을 매칭하여 그 책을 박스 형태로 윤곽선을 따야 합니다. 따러서 이전 코드가 1번과 같지만, 과제 1번은 Panorama Stitching을 위해서 진행한다면 과제 2번은 책의 영상변환을 구하기 위한 Task와 같다고 생각했습니다. 즉 book이미지는 똑바로 정면에서 보는 사진이지만, 여러 사물이 포함된 사진에서는 해당 책이 perspective transform된 형태이기 때문입니다. 따라서 변환된 사물의 좌표를 구하고, 그 좌표를 이용해서 box를 그리면 될 것이라고 생각했습니다. 따라서 매칭된 결과를 통해 나온 homgrapy 행렬을 통해 perspective transform을 수행하면 여러 사물이 있는 이미지에서 책의 꼭지점 좌표를 구할 수 있습니다. 이 연산을 위해 perspectiveTransform()함수를 사용하여 homgrapy행렬에 해당하는 영상 변환을 하였고, 그 결과 변환된 책의 꼭지점을 구할 수 있었습니다. 따라서 이렇게 구한 꼭지점을 사용하여 line함수를 사용하여 윤곽선을 그려주었습니다.
perspectiveTransform()은 지난주 실습시간에 직접 실습을 진행했던 함수였기 때문에 어렵지 않게 사용할 수 있었습니다.

## 2-4) 결과 분석

과제 2번은 책의 윤곽선을 찾는 실습이었습니다. 이번 과제는 개인적으로 많은 고민들을 했던 과제였습니다. 우선 특징점을 추출하고, Homograpy행렬을 구하는 것까지는 과제의 힌트로 나와있었기 때문에 어렵지 않게 생각할 수 있었지만, Stitching을 위한 task가 아닌데, 영상을 변환하는 것이 어떤 의미가 있을까 라는 생각을 많이 했습니다. 따라서 이론시간에 배웠던 내용들과 지난 주 실습에서 배웠던 내용들을 다시 한번 찾아보고 복습해보았는데, 지난 실습에서 perspective transform을 어떻게 수행하였는지 문득 떠올랐습니다. 바로 점 4개를 변환하는 방식이었던 것입니다. 책에 대한 꼭지점 좌표는 다소 쉽게 구할 수 있었고, 이론 수업시간에 배웠던 Homograpy행렬의 특징을 다시 한번 복습하면서 어떻게 윤곽선을 나타낼 수 있을지 생각할 수 있었습니다.

이번 과제는 과제 해결을 위한 고민을 하면서 Homograpy행렬의 이론적인 특성을 다시 한번 이해하고, perspective transform에 대해서도 복습할 수 있었습니다. 실제로 이러한 특성들을 생각하고, 직접 적용했을 때, 각각의 book에 대해 정확하게 윤곽선을 그려내는 것을 확인할 수 있었습니다.

# [고찰]

이번 실습는 Panorama stitching을 직접 실습해보고, 이 과정에서 얻을 수 있는 homograpy행렬의 원리를 통해 책의 윤곽선을 그리는 과제를 수행하였습니다. Panorama stitching과제에서는 이번주에 해당 내용을 이론적으로 배웠기 때문에, 이론에 대해서 완벽하게는 숙지하지 못한 상태였습니다. 하지만 이번 실습을 통해 이론적인 원리를 직접 눈으로 확인하고 공부할 수 있는 기회였습니다. 특히 직접 사진을 찍어 panorama의 결과를 확인할 수 있어서 매우 흥미로웠습니다.

2번의 경우 제가 최근 했던 실습 중 가장 많이 배워가는 실습이었습니다. 어떻게 윤곽선을 그릴지 고민을 하고, 이번주에 배운 이론 내용 뿐만 아니라 지난 주에 배운 transform개념을 활용함으로써 다시 한번 복습할 수 있는 기회를 얻었습니다. 실습 결과도 만족할 정도로 나와서 더욱 보람있는 실습이었습니다.