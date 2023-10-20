---
title: Filtering in Frequency Domain
date: 2023-03-31 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Filtering,
    Fourier Transform
  ]
pin: true
---

# 1. img1.jpg에 band pass filter를 적용할 것

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리
	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//magnitude 취득
	//log(1+ sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);
	
	return dst;
}

Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);
	//phase 취득

	return phaImg;
}

Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);
	//magnitude계산을 반대로 수행
	
	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	//극좌표계 -> 직교좌표계(각도와 크기로부터 2차원좌표)
	
	Mat complexImg;
	merge(planes, 2, complexImg);
	//실수부, 허수부 합체
	
	return complexImg;
}

Mat doIdft(Mat complexImg) {
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT를 이용한 원본 영상 취득
	
	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);
	
	//일반 영상의 type과 표현범위로 변환
	return dstImg;
}

Mat padding(Mat srcImg) {
	int dftrow = getOptimalDFTSize(srcImg.rows);
	int dftcols = getOptimalDFTSize(srcImg.cols);
	Mat paddingImg;
	copyMakeBorder(srcImg, paddingImg, 0, dftrow - srcImg.rows, 0, dftcols - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));
		return paddingImg;
}

Mat doBPF(Mat srcImg) {//Band Pass Filter
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<BPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);
	imshow("mag_img", magImg);

	Mat maskImg1= Mat::zeros(magImg.size(), CV_32F);//LPF
	Mat maskImg2 = Mat::ones(magImg.size(), CV_32F);//HPF
	Mat maskImg_BPF = Mat::zeros(magImg.size(), CV_32F);//BPF

	circle(maskImg1, Point(maskImg1.cols / 2, maskImg1.rows / 2), 50, Scalar::all(1), -1, -1, 0);
	imshow("LPF", maskImg1);//BPF를 위해 조금 더 넓은 범위의 LPF

	circle(maskImg2, Point(maskImg2.cols / 2, maskImg2.rows / 2), 20, Scalar::all(0), -1, -1, 0);
	imshow("HPF", maskImg2);//BPF를 위해 조금 더 좁은 범위의 HPF

	bitwise_and(maskImg1, maskImg2, maskImg_BPF);//LPF와 HPF를 and 연산으로 BPF를 만든다.
	imshow("BPF", maskImg_BPF);
	
	Mat magImg2;
	multiply(magImg, maskImg_BPF, magImg2);
	imshow("magImg2", magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

int main() {

	Mat src_img = imread("D:/study/디영처/5주차/img1.jpg", 0);
	Mat dst_img;

	dst_img = doBPF(src_img);

	imshow("src_img", src_img);
	imshow("BPF_img", dst_img);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
```

## 1-2) 실험결과

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7e398d8e-33a3-4bd7-a326-4b7d4426b91f)

## 1-3)구현 과정

- **`Mat doDft(Mat srcImg):` 입력 이미지에 대해 DFT(Discerete Fourier Transform)를 수행하는 doDft함수를 정의하였습니다. 이 함수를 통해 이미지를 주파수 영역으로 변환합니다.**
- **`Mat getMagnitude(Mat complexImg):` DFT 결과인 복소수 형태의 이미지를 입력으로 받아 해당 이미지의 주파수 성분의 Magnitude를 계산하는 getMagnitude함수를 정의하였습니다.**
- **`Mat myNormalize(Mat src):` 입력 이미지를 0부터 255까지의 값으로 정규화(normalization)하는 함수 myNormalize함수를 정의하였습니다.**
- **`Mat getPhase(Mat complexImg):` 복소수 형태의 이미지를 입력으로 받아, Phase를 계산하는 getPhase함수를 정의하였습니다.**
- **`Mat centralize(Mat complex):` 복소수 형태의 이미지를 입력으로 받아 좌표계를 중앙으로 이동시키는 centralize함수를 정의합니다. 이는 실습에서 좌표계의 형태가 다르기 때문에 정의해주어야 합니다.**
- **`Mat setComplex(Mat magImg, Mat phaImg):` Magnitude와 Phase를 입력으로 받아 complex 이미지를 만드는 setComplex 함수를 정의합니다.**
- **`Mat doIdft(Mat complexImg):` IDFT(Inverse Discerete Fourier Transform)를 수행하는 doIdft함수를 정의하였습니다. 이를 통해 주파수 영역의 데이터를 다시 이미지로 변환합니다.**
- **`Mat doBPF(Mat srcImg):` Band Pass Filter를 적용하는 함수로 LPF와 HPF를 사용하여 Band Pass Filter를 만들어내는 역할을 합니다. 자세한 절차는 결과 분석에서 말씀드리겠습니다.**

## 1-4)결과 분석

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fb7052b9-adc1-4278-844d-c126daa0de0f)

- 위는 img1의 magnitude image입니다.

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/116aa91e-aa3f-4182-adca-686955ef8716)

Low Pass Filter Mask

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/149fa1ff-96ef-4a11-8771-1b67ed30aebd)

High Pass Filter Mask

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/53b53b7e-d806-49e8-a467-9d5134050a6b)

Band Pass Filter Mask

- 위는 각각 LPF, HPF를 구한 결과입니다. 실습시간에 적용했던 것보다 HPF는 더 좁게, LPF는 더 넓게 원을 만들어 Mask를 만들었고, Bitwise_and연산자를 사용하여 가장 오른쪽 Mask와 같은 Band Pass Filter의 Mask를 만들었습니다. 이를 Magnitude image에 적용하였고, 그 결과는 아래와 같습니다.

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/3c412f54-65fa-4dd2-8b6e-3bc05cf94f6e)

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7f6c0879-c56a-4a3d-a10a-3af4e2e96b40)

- 이후 이를 다시 이미지로 변환시켰을 때 최종 결과는 오른쪽 그림과 같습니다.

**최종 결과를 보았을 때, 원래의 이미지에서 Band의 주파수에 해당하는 image가 오른쪽 그림과 같다는 것을 알 수 있었고, 저주파와 고주파 모두 없는 이미지는 형체를 잘 알아볼 수 없을만큼 많이 변형되었다는 것을 알 수 있었습니다. 또한 Band Pass Filter를 구현하는 과정에서 여러가지 Filter의 특성을 활용하였는데, 이러한 방식을 사용하면 원하는 주파수 이미지에 대해서 추출 할 수 있을 것이라고 예상할 수 있었습니다.**

# 2. Spatial domain, frequency domain 각각에서 sobel filter를 구현하고 img2.jpg에 대해 비교할 것

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리
	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//magnitude 취득
	//log(1+ sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

Mat doIdft(Mat complexImg) {
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT를 이용한 원본 영상 취득

	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);

	//일반 영상의 type과 표현범위로 변환
	return dstImg;
}

Mat padding(Mat srcImg) {
	int dftrow = getOptimalDFTSize(srcImg.rows);
	int dftcols = getOptimalDFTSize(srcImg.cols);
	Mat paddingImg;
	copyMakeBorder(srcImg, paddingImg, 0, dftrow - srcImg.rows, 0, dftcols - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));
		return paddingImg;
}

int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) {
		return sum / sumKernel; //합이 1로 정규화되도록 해 영상의 밝기변화 방지
	}
	else {
		return sum;
	}
}

Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { -1,0,1,
						   -2,0,2,
						   -1,0,1 };//가로방향 Sobel 마스크

	int kernelY[3][3] = { -1,-2,-1,
						   0, 0, 0,
						   1, 2, 1 };//세로방향 Sobel 마스크
	//마스크 합이 0이 되므로 1로 정규화하는 과정은 필요 없음
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
			// 두 에지 결과의 절대값 합 형태로 최종결과 도출
		}
	}
	return dstImg;
}

Mat DFT_image(Mat srcImg) {//원본 image를 DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat dstImg = myNormalize(magImg);

	return dstImg;
}

Mat DFT_sobel_image(Mat srcImg) {//sobel filter가 적용된 image를 DFT
	Mat sobelImg = mySobelFilter(srcImg);
	imshow("sobelImg", sobelImg);
	Mat padImg = padding(sobelImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magSobelimg = getMagnitude(centerComplexImg);
	Mat dstImg = myNormalize(magSobelimg);

	return dstImg;
}

Mat get_fre_sobel(Mat image, Mat sobel_image) {//sobel filter 자체를 DFT도메인에서 구하는 함수
	Mat fre_sobel = Mat::zeros(sobel_image.size(), CV_32F);
	divide(sobel_image, image, fre_sobel);
	Mat dstImg = myNormalize(fre_sobel);

	return dstImg;
}

int main() {
	Mat src_img = imread("D:/study\/디영처/5주차/img2.jpg", 0);
	Mat dftImg;
	Mat dftSobelimg;
	Mat dftFreSobel;
	imshow("src_img", src_img);

	dftImg = DFT_image(src_img);
	imshow("dftImg", dftImg);

	dftSobelimg = DFT_sobel_image(src_img);
	imshow("dftSobelimg", dftSobelimg);

	dftFreSobel = get_fre_sobel(dftImg,dftSobelimg);
	imshow("dftFreSobel", dftFreSobel);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
```

## 2-2) 실험 결과

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8a42fe0e-adda-439f-b963-24cd8879cb05)

- 위 결과는 sobel filter 자체를 주파수 도메인에서 구현한 것이고, 이를 만들기 위한 일련의 과정들은 결과분석에서 자세히 설명드리겠습니다.
- 잘 보이지 않지만 자세히 보면, 4개의 밝은 점이 별처럼 있는 것을 확인할 수 있었습니다.

## 2-3) 구현 과정

- **`int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height)`** 과 **`Mat mySobelFilter(Mat srcImg)`** 함수가 각각 추가되었습니다. 하지만 위 함수들은 모두 지난 과제에서 Sobel Filter를 구현하기 위해 작성한 코드와 동일합니다. 따라서 위 코드에 대한 설명을 생략하도록 하겠습니다.
- **`Mat DFT_image(Mat srcImg):` DFT_image함수는 원본 image에 대한 DFT결과를 얻기 위해 정의한 함수입니다. 함수의 구현 과정에서 padding, doDft등의 함수는 1번 과제에서 사용하였던 함수의 코드와 모두 동일합니다.**
- **`Mat DFT_sobel_image(Mat srcImg):` sobel filtering을 한 image에 대한 DFT결과를 얻기 위해 정의한 함수입니다.**
- **`Mat get_fre_sobel(Mat image, Mat sobel_image):` 이 함수가 가장 주요한 역할을 하는 함수입니다. 바로 Sobel Filter를 주파수 도메인에서 구하는 함수인데, 그 원리는 Convolution 연산의 특성에 있습니다.**

![Untitled]![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8c35296a-3afd-4435-9bc1-8e5236b3e2c7)

- 위는 강의노트에서 가져온 FT에서의 Convolution 연산의 특성입니다. Sobel Filter는 convolution연산과 같기 때문에, i**mage*sobel_filter=sobel_image라면 F[sobel_filter]=F[sobel_image]/F[image]의 식을 이용하면 F[sobel_filter], 즉 주파수 도메인에서의 sobel filter를 구할 수 있습니다.** 따라서 위 함수는 이를 위해 정의된 함수입니다.

## 2-4) 결과 분석

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8208ca17-5ddc-43b1-a90b-deedec5556f8)

- 위는 원래의 image에 DFT를 적용한 결과입니다.

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1ef1046d-3e8d-4b79-9ed3-129ec8c909d4)

- 위는 sobel filtering이 된 image에 DFT를 적용한 결과입니다. 주파수 도메인에서 봤을 때, Sobel filter는 x, y축에 대한 edge를 검출하는 특성으로 인해 십자가 모양이 생긴 것을 볼 수 있습니다.

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/55c6cc75-088e-42d1-80c3-e184c5660e39)

- 위 결과는 FT에서의 convolution특성을 사용하여, 주파수 도메인에서 구한 Sobel Filter입니다. 처음에 이 결과를 보고 많이 당황했습니다. coding이 틀렸는지 확인해보았고, 이론을 잘못 적용했는지 확인해보았습니다. 그러나 틀린 부분이 없었고, 결과를 자세히 보니, 밝은 점 4개가 찍혀있는 것을 볼 수 있었습니다. 따라서 위 밝은 점 부분이 Edge의 주파수가 모여있는 부분이라는 것을 예상할 수 있었습니다. FT의 convolution의 특성을 사용하여 Sobel Filter의 주파수를 구할 수 있다는 것이 흥미로웠습니다.

# 3. img3.jpg에서 나타나는 flickering 현상을 frequency domain filtering을 통해 제거할 것

## 3-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리
	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//magnitude 취득
	//log(1+ sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);
	//phase 취득

	return phaImg;
}

Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);
	//magnitude계산을 반대로 수행

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	//극좌표계 -> 직교좌표계(각도와 크기로부터 2차원좌표)

	Mat complexImg;
	merge(planes, 2, complexImg);
	//실수부, 허수부 합체

	return complexImg;
}

Mat doIdft(Mat complexImg) {
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT를 이용한 원본 영상 취득

	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);

	//일반 영상의 type과 표현범위로 변환
	return dstImg;
}

Mat padding(Mat srcImg) {
	int dftrow = getOptimalDFTSize(srcImg.rows);
	int dftcols = getOptimalDFTSize(srcImg.cols);
	Mat paddingImg;
	copyMakeBorder(srcImg, paddingImg, 0, dftrow-srcImg.rows, 0, dftcols-srcImg.cols, BORDER_CONSTANT, Scalar::all(0));
	return paddingImg;
}

Mat doFlickF(Mat srcImg) {
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);
	imshow("mag_img", magImg);

	Mat maskImg1 = Mat::ones(magImg.size(), CV_32F);//가로 직사각형 필터
	Mat maskImg2 = Mat::ones(magImg.size(), CV_32F);//세로 직사각형 필터
	Mat maskImg3 = Mat::zeros(magImg.size(), CV_32F);//가운데 정사각형 필터
	Mat maskImg_FF = Mat::zeros(magImg.size(), CV_32F);//가로 직사각형+세로 직사각형 필터
	Mat maskImg_FF2 = Mat::zeros(magImg.size(), CV_32F);//가로 직사각형+세로직사각형+ 가운데 정사각형 필터

	rectangle(maskImg1, Rect(Point(maskImg1.cols / 2 - 3, 0), Point(maskImg1.cols / 2 + 3, maskImg1.rows)), Scalar::all(0), CV_FILLED, 1, 0);
	imshow("가로 직사각형 필터", maskImg1);
	rectangle(maskImg2, Rect(Point(0, maskImg1.rows / 2 - 3), Point(maskImg1.cols, maskImg1.rows / 2 + 3)), Scalar::all(0), CV_FILLED, 1, 0);
	imshow("세로 직사각형 필터", maskImg2);
	rectangle(maskImg3, Rect(Point(maskImg1.cols / 2 - 3, maskImg1.rows / 2 - 3), Point(maskImg1.cols / 2 + 3, maskImg1.rows / 2 + 3)), Scalar::all(1), CV_FILLED, 1, 0);
	imshow("가운데 정사각형 필터", maskImg3);
	bitwise_and(maskImg1, maskImg2, maskImg_FF);//LPF와 HPF를 and 연산으로 BPF를 만든다.
	bitwise_or(maskImg3, maskImg_FF, maskImg_FF2);
	imshow("최종 Filter",maskImg_FF2);

	Mat magImg2;
	multiply(magImg, maskImg_FF2, magImg2);
	imshow("magImg2", magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

int main() {

	Mat src_img = imread("D:/study/디영처/5주차/img3.jpg", 0);
	Mat dst_img;

	dst_img = doFlickF(src_img);

	imshow("src_img", src_img);
	imshow("FF_img", dst_img);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
```

## 3-2) 실험 결과

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ae2d7f29-2ca2-42f9-9d83-72b3ac467710)

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/715a6626-0c9b-4804-afd9-5bdca17f3655)

## 3-3) 구현 과정

- **`Mat doFlickF(Mat srcImg):` doFlickF함수는 Flickering 현상을 없애기 위해 구현한 코드입니다. 코드 내에서 쓰이는 함수들은 모두 이전 과제들에서 정의된 함수들입니다. 여기에서 다른점이 있다면 Flickering현상이 있는 img3를 DFT로 주파수 도메인으로 전환했을 때, 십자가 모양의 선이 생겼는데, 이를 통해 가로 세로 방향으로 많은 밝기 변화, 즉 고주파가 많다는 것을 알 수 있었고, 이 고주파 성분으로 인해 Flickering현상이 발생한다고 생각했습니다. 따라서 중앙의 저주파 성분은 남기면서 십자가 모양의 Filter를 만들기 위해 코딩을 했습니다. 자세한 내용은 결과 분석에서 이미지와 함께 설명드리겠습니다.**

## 3-4) 결과 분석

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f39b8c3e-da98-433d-b573-345bc9efb193)

- 위는 Flickering현상이 있는 img3를 주파수 도메인으로 나타낸 영상입니다. DFT를 적용하기 이전에도 가로, 세로 방향으로 밝기 변화가 많아서 고주파 성분이 많을 것이라고 예상했는데, 예상대로 고주파 성분이 많은 것을 볼 수 있었습니다. 특히 Flickering의 특성 상 주파수 도메인에서 십자가 형태가 나왔습니다. 따라서 저 부분의 주파수를 Filtering한다면 Flickering현상을 완화할 수 있을 것이라고 생각했습니다. 해당 Filter를 만드는 과정은 다음과 같습니다.

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bb533667-add7-4a59-bce5-4addd5ad3c9d)

![Untitled 16](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0d3861cb-2385-49cc-b26b-9d12efb3d487)

- 위와 같은 가로 세로의 Mask를 생성했는데, rectangle함수를 이용했습니다. Flickering현상의 원인이 위 mask부분에 있다고 예측할 수 있었습니다. 따라서 위 2개의 Filter를 bitwise_and연산을 통해 십자가 모양을 제외한 mask를 생성했습니다. 하지만 여기서 한가지 문제가 있었습니다. 이렇게 Filtering을 진행하게 된다면, 가운데에 있는 저주파 성분까지 모두 Filtering되어 제대로 된 결과가 나오지 않을 것이라고 생각했습니다. 따라서 다음과 같은 mask를 하나 더 생성했습니다.

![Untitled 17](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1ff9f268-1bf0-49b6-93a9-e88668152b82)

- 위 Mask는 정확히 십자가 모양의 Mask의 중앙 부분 넓이와 같습니다. 따라서 위 Mask를 이전에 생성한 십자가 모양의 Mask와 결합한다면, 저주파 성분은 유지하면서 Flickering 현상을 완화할 수 있을 것이라고 생각했습니다. 따라서 이전에 합성한 Mask와 위 Mask를 bitwise_or연산을 통해 합쳐주었습니다. 그 결과는 아래와 같습니다.

![Untitled 18](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6d24acfc-3abd-44e8-90f5-5b28e0c87ba8)

- 위의 최종 Mask를 볼 수 있듯이, 가운데 저주파는 살리고, 십자가 모양의 주파수 성분을 Filtering할 수 있도록 만들었습니다. 이를 주파수 도메인에 적용하면 결과는 다음과 같습니다.

![Untitled 19](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d03b897a-014c-4395-80d3-9a94255df87b)

- 위는 Filtering을 적용한 주파수 도메인의 이미지이고, 고주파 성분이 사라진 것을 볼 수 있습니다. 이에 다시 IDFT를 적용하여 영상으로 전환하였을 때의 결과는 다음과 같습니다.

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2d9970f9-84e6-4066-8488-92fc4980e186)

- 위는 최종적으로 Filtering을 한 결과 이미지이고 이전 img3와 비교해보았을 때, 가로 방향으로 줄지어 있던 검은색 Flickering현상이 완화된 것을 볼 수 있었습니다. 하지만 가로 방향의 미세한 줄무늬 현상은 없어지지 않았는데, 이는 다른 방향의 고주파 성분인 것으로 생각됩니다. 실제로 DFT를 적용한 이미지를 보았을 때, 다른 방향으로 주파수 성분들이 이어져있는 것을 보았는데, rectangle함수를 사용하여 없애는 것에는 한계가 있고, 또한 십자가 모양의 Filtering이 완벽하게 Flickering현상을 유도하는 주파수를 없앨 수 없기 때문에, 위 결과에 만족해야 했습니다. 이후에 더 좋은 Filtering기술에 대해 배우면, 조금 더 완성도 높은 Filtering을 할 수 있을 것으로 기대 됩니다.

# 고찰

이번 실습과 과제는 FT를 이용하여 주파수 도메인에서 영상을 다루었습니다. FT은 수학적으로만 접해보았기에, 이전까지는 그저 외우고 시험을 잘 치기 위한 수학 공식에 불가했습니다. 하지만 영상을 주파수 도메인으로 보내고, 분석하면서 영상을 의도한대로 바꾸고 개선할 수 있는 것에 흥미를 많이 느꼈습니다. 또한 강의시간에 배웠던 이론적인 부분을 실제 코드로 개선되어지는 영상들을 보면서 더욱 직관적으로 영상처리에 대해 이해할 수 있게 된 과제였습니다.