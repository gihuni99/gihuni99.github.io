---
title: Local Features and SIFT
date: 2023-05-12 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Feature Detection
  ]
pin: true
---

# 1. coin.png의 동전 개수를 알아내는 프로그램을 구현

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;

void cvBlobDetection() {
	Mat img = imread("D:/study/디영처/11주차/11주차 실습 자료/coin.png", IMREAD_COLOR);
	// <Set params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 3000;
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 7000;
	params.filterByCircularity = true;
	params.minCircularity = 0.6;
	params.filterByConvexity = true;
	params.minConvexity = 0.9;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// <Set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// <Detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	// <Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result,
		Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("keypoints", result);
	printf("동전의 개수는 %d개 입니다.",keypoints.size());
	waitKey(0);
	destroyAllWindows();
}

void main() {
	cvBlobDetection();
}
```

## 1-2) 실험결과

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/892ec936-62f9-4ba8-952b-8a735f4c6f08)

## 1-3)구현 과정

- cvBlobDetection함수를 사용하여 동전을 detection할 수 있도록 구현하였습니다.
- Open CV의 Blob detector를 사용하였고, 동전은 원의 형태이기 때문에 Blob detection으로 찾기에 적합하였습니다. 다만 모든 동전을 올바르게 detection하기 위해 파라미터 값들의 설정이 중요했습니다. 각 파라미터에 대한 정보는 아래와 같습니다.

	params.minThreshold = 10;
	params.maxThreshold = 3000;
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 7000;
	params.filterByCircularity = true;
	params.minCircularity = 0.6;
	params.filterByConvexity = true;
	params.minConvexity = 0.9;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

- 강의노트에 있는 코드를 사용하여 코드를 구현하였습니다. 다만 해당되는 파라미터 조건으로 사용했을 때 모든 동전을 detection을 하지 못했습니다. 기존 코드에 있던 파라미터 값들을 모두 변경하며 찾아보았지만, 완벽하게 동전을 세지 못했습니다. 하지만 maxArea값을 추가했을 때, 정상적으로 동작했고, 코드에 사용한 파라미터 값들은 위와 같습니다.
- cvBlobDetection의 과정에서 keypoints.size()는 keypoint의 개수를 나타내고, 이것이 동전의 개수와 같습니다. 따라서 이를 출력해주면 총 동전의 개수를 알 수 있습니다.

## 1-4)결과 분석

Blob Detection은 Scale정보를 포함한다는 것이 가장 특징적입니다. 이론 시간에 배웠던 내용이지만 파라미터값들을 설정하면서 각 변수가 미치는 영향에 대해 생각하면서 실험 결과를 분석했습니다. Open CV를 사용해서 코딩자체는 비교적 간단했지만, 원하는 Task 즉 과제1번에서는 동전을 세는 Task를 제대로 동작하기 위해서는 각 파라미터 값들을 설정하여 모든 동전들을 Detection할 수 있도록 설정하는 과정이 필요했습니다. 따라서 실험적으로 어떤 값이, 어떤 영향을 미치는지 분석하였고 이를 토대로 결과를 내기 위한 파라미터 값들을 찾아내었습니다. 결과는 원하는 결과가 나왔습니다.

# 2. OpenCV의 1. corner detection과 2. circle detection을 이용해 삼각형, 사각형, 오각형, 육각형의 영상을 순차적으로 읽어와 각각 몇 각형인지 알아내는 프로그램을 구현(도형 4개는 그림판, PPT 등을 이용해 각자 생성할 것)

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;

void cvBlobDetection(Mat img) {
	// <Set params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 1;
	params.maxThreshold = 100;
	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea = 500;
	params.filterByCircularity = true;
	params.minCircularity = 0.7;
	params.filterByConvexity = true;
	params.minConvexity = 0.9;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// <Set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// <Detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	// <Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result,
		Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cout << "꼭지점 개수는 " << keypoints.size() << "개 입니다.따라서 ";
	imshow("keypoints", result);

	if (keypoints.size() == 3) cout << "삼각형입니다." << endl;
	else if (keypoints.size() == 4) cout << "사각형입니다." << endl;
	else if (keypoints.size() == 5) cout << "오각형입니다." << endl;
	else if (keypoints.size() == 6) cout << "육각형입니다." << endl;
	else cout << "잘못된 Detection결과입니다." << endl;

}

Mat cvHarrisCorner(Mat img) {
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}

	resize(img, img, Size(300, 300), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	// <Do Harris corner detection>
	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.03, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// <Get abs for harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	// <Print corners>
	int thresh = 120;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y += 1) {
		for (int x = 0; x < harr.cols; x += 1) {
			if ((int)harr.at<float>(y, x) > thresh){
				circle(result, Point(x, y), 7, Scalar(0, 0, 0), -1, 4, 0);
			}
		}
	}

	imshow("Harris image", harr_abs);
	imshow("Target image", result);

	return result;
}

void main() {
	Mat Tri = imread("D:/study/디영처/11주차/11주차 실습 자료/Tri.PNG", 1);
	Mat Rec = imread("D:/study/디영처/11주차/11주차 실습 자료/Rec.PNG", 1);
	Mat Pen = imread("D:/study/디영처/11주차/11주차 실습 자료/Pen.PNG", 1);
	Mat Hex = imread("D:/study/디영처/11주차/11주차 실습 자료/Hex.PNG", 1);
	imshow("origin_Tri", Tri);
	imshow("origin_Rec", Rec);
	imshow("origin_Pen", Pen);
	imshow("origin_Hax", Hex);
	Mat res_Tri = cvHarrisCorner(Tri);
	cvBlobDetection(res_Tri);
	Mat res_Rec = cvHarrisCorner(Rec);
	cvBlobDetection(res_Rec);
	Mat res_Pen = cvHarrisCorner(Pen);
	cvBlobDetection(res_Pen);
	Mat res_Hex = cvHarrisCorner(Hex);
	cvBlobDetection(res_Hex);

	waitKey(0);
	destroyAllWindows();
}
```

## 2-2) 실험 결과

### [삼각형]

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/01faadae-8dd4-4823-bde8-07606076539b)

### [사각형]

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/135d1f97-38ed-48c4-b0e0-9441060cd763)

### [오각형]

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/38957228-6be1-454b-916f-4fc1ca632572)
### [육각형]

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/72dc304e-c245-4f22-9d52-b175d65bf68b)

### [최종 결과]

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/97e94dd8-bf7e-4cf4-8981-bb8cc5bf1ca8)

## 2-3) 구현 과정

- **`Mat cvHarrisCorner(Mat img)`** : cvHarrisCorner함수는 Open CV를 사용해 구현한 Corner detection Implementation입니다. 이 함수를 통해 corner를 찾고, corner에 원을 그립니다. 이 때, 원은 과제 1번에서 사용했던 Blobdetector를 사용하여 찾을 것인데, 더 찾기 쉽게 하여 정확도를 높이기 위해 꽉 찬 원을 그리도록 코딩하였습니다. cvHarrisCorner함수를 통과하면 각 도형의 꼭지점에는 검은색 원이 그려집니다. 따라서 이 원을 모두 detect하면 꼭지점의 개수를 알 수 있고 이를 통해 어떤 도형인지 알 수 있습니다.
- **`void cvBlobDetection(Mat img)`** : cvBlobDetection은 과제 1번에서 사용했던 Blob detector와 같은 동작을 합니다. 다만 파라미터 값들을 과제 2번을 수행할 수 있도록 수정하였고, keypoints.size()를 통해 어떤 도형인지 출력할 수 있도록 코딩하였습니다.

## 2-4) 결과 분석

2번은 BlobDetector는 과제 1번과 동일하게 사용하였습니다. 하지만 HarrisCorner Detector를 Open CV를 사용하여 corner detection을 진행했다는 점이 달랐습니다. 또한 detect한 후 Corner부분에 원을 그려서, 이를 통해 Corner의 개수를 파악하고 어떤 도형인지 알아내는 과정이 흥미로웠습니다. 이번 과제는 여러가지 고려해야할 상황들이 많아서 많은 파라미터 값들을 바꾸면서 진행하였습니다. 그 결과 최적의 값들을 찾아내었고, 이를 통해 올바른 결과를 도출할 수 있었습니다.

# 3. church.jpg에 투시변환(perspective change)과 밝기 변화를 같이 수행한 후 SIFT 특징점을 구했을 때 원본 영상의 SIFT 특징점이 보존되는지 확인해볼 것(warpPerspective()함수 사용)

## 3-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;

Mat cvFeatureSIFT(Mat img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	Ptr<cv::SiftFeatureDetector> detector = SiftFeatureDetector::create();
	std::vector<KeyPoint> keypoints;
	detector->detect(gray, keypoints);

	Mat result;
	drawKeypoints(img, keypoints,result,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	return result;
}

Mat warpPers(Mat src) {
	Mat dst;
	Point2f src_p[4], dst_p[4];

	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(512, 0);
	src_p[2] = Point2f(0, 512);
	src_p[3] = Point2f(512, 512);

	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(512, 0);
	dst_p[2] = Point2f(0, 512);
	dst_p[3] = Point2f(412, 412);

	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(src, dst, pers_mat, Size(512, 512));
	return dst;
}

void main() {
	Mat org_church = imread("D:/study/디영처/11주차/11주차 실습 자료/church.jpg", 1);
	resize(org_church, org_church, Size(512, 512), 0, 0, INTER_CUBIC);
	imshow("org_church", org_church);

	Mat Lum_church = org_church + Scalar(125, 125, 125);
	imshow("Lum_church", Lum_church);

	Mat SIFT_church = cvFeatureSIFT(org_church);
	imshow("SIFT_church", SIFT_church);

	Mat Warping_church = warpPers(org_church);
	Mat Warping_SIFT_church = cvFeatureSIFT(Warping_church);
	imshow("Warping_SIFT_church", Warping_SIFT_church);

	Mat Warping_Lum_church = warpPers(Lum_church);
	Mat Warping_Lum_SIFT_church= cvFeatureSIFT(Warping_Lum_church);
	imshow("Warping_Lum_SIFT_church", Warping_Lum_SIFT_church);
	waitKey(0);
	destroyAllWindows();
}
```

## 3-2) 실험 결과

### [원본 이미지 vs 밝기 값 변화 이미지]

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ce083cb0-e877-4fab-b737-df19db0991e3)

### [원본 SIFT vs Warping SIFT vs Warping&Luminance SIFT]

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f4972f33-99ff-4769-8e3d-11eef7d44e42)

## 3-3) 구현 과정

- **`Mat cvFeatureSIFT(Mat img)`** : Open CV를 사용하여 SIFT특징점을 구하는 함수입니다.
- **`Mat warpPers(Mat src)`** : 투시 변환을 수행하는 함수입니다.

위 함수를 통해 결과 도출에 대해서는 결과 분석에 작성하겠습니다.

## 3-4) 결과 분석

3번은 SIFT 특징점을 구하는 과제입니다. 여기서 총 3가지 image에 대한 SIFT 특징점을 구했습니다. 원본 사진의 SIFT특징점 도출, Warping된 사진의 SIFT특징점 도출, 밝기 값에 변화가 있고 Warping된 사진의 SIFT특징점 도출, 이렇게 총 3가지 경우입니다. 여기에서 알 수 있었던 것은 Warping은 SIFT특징점을 구하는 것에 거의 영향을 미치지 않는 다는 점입니다. 반면 밝기 값의 변화는 SIFT특징점을 구하는 것에 많은 영향을 미쳤습니다. 위 결과에서 1, 2번 사진을 비교해보면 Warping된 이후에도 SIFT특징점이 구해지는 차이가 거의 없다는 것을 알 수 있었습니다. 물론 Warping됨에 따라, image가 바뀌어 해당 위치에 맞게 SIFT특징점의 위치가 변경되었지만 Warping된 이후에도 같은 지점에서의 SIFT특징점을 구하는 것을 볼 수 있었습니다. 하지만 3번 사진에서 밝기값을 R, G, B각각 125만큼 증가시킨 사진에 대해서는 많은 SIFT특징점들이 원본 사진과 달리 구해지지 않는 것을 볼 수 있습니다. 

위 같은 결과가 나온 원인을 생각해보았을 때, 밝기 값이 증가함에 따라 영상 전체의 Contrast가 줄어드는 것을 볼 수 있습니다. 원본 사진에 비해 SIFT 특징점이 많이 사라진 부분을 보면 하늘 부분의 거의 대부분 사라진 것을 볼 수 있습니다. 이는 사진에서 하늘 부분이 특히 Contrast가 낮아졌기 때문입니다. 즉 Contrast가 감소함에 따라 Gradient가 줄어든 것 입니다. 따라서 과제 3번을 통해 Gradient에 많은 영향을 주는 밝기 값의 변화가 SIFT특징점 변화에 많은 영향을 준다는 것을 알 수 있었습니다.

# [고찰]

이번 과제는 Blob Detector, Corner Detector, SIFT 세가지 연산에 대한 결과를 도출하고 분석하는 과제였습니다. 이론이 어려워져서 코드를 구현하는 것에 많은 어려움이 있지 않을까 하는 걱정이 있었지만, Open CV를 이용하여 구현할 수 있어서 오히려 코딩 자체는 난이도가 낮아진 것 같습니다. 이에 따라 각 연산의 특징들과 이를 통해 나온 결과에 대해서 이론적, 실험적으로 분석하는 것에 집중할 수 있어서 의미있었던 과제였습니다.