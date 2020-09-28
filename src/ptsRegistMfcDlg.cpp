
// ptsRegistMfcDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "MFCptsRegist.h"
#include "ptsRegistMfcDlg.h"
#include "afxdialogex.h"
#include "CNewfileIn.h"
#include "CNewtip.h"
#include "CNewIcpDlg.h"
#include "CNewLineExtract.h"
#include "CNewDlgtip1.h"
#include"CNewDlgtip2.h"
#include"CoutputError.h"
#include"CRegistError.h"
#include"CfileError.h"

#include "lasreader.hpp"
#include "laswriter.hpp"
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <string>
#include <vector>
#include <fstream>
#include <ios>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <vtkAutoInit.h>
#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeType,vtkRenderingOpenGL) 
#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
using namespace std;
using namespace cv;
typedef pcl::PointXYZI ptype;
typedef pcl::PointCloud<ptype>::Ptr ptrtype;
typedef struct
{
	float max_heightdiff = 0;
	vector<int> index;
}voxel;

typedef struct {
	int grayScale = 0;
	float maxdif_height = 0;
	float region_difheight = 0;
	float max_height = 0;
	int candidate = 0;
	vector<int> indexID;

}flat_grid;
typedef struct {
	vector<int> als_index;
	vector<cv::Vec2f> xy_drift;
}lines_combination;
typedef struct {
	pcl::PointXYZ startpoint;
	pcl::PointXYZ endpoint;
}Point_vec;
//全局变量
float differ_x1(0),  t1_c(0), differ_y1(0), differ_z1(0), gps_time(0);
U16 pts_sourceID; 
float p_gpsT;
ptype pmin;
bool file_exist(false), hasRoughed(false), hasPrecised(false);
Vec2f xy_drift;
pcl::PointCloud<ptype>::Ptr mls_rect(new pcl::PointCloud<ptype>);
pcl::PointCloud<ptype>::Ptr als_cloud(new pcl::PointCloud<ptype>);
pcl::PointCloud<ptype>::Ptr mls_cloud(new pcl::PointCloud<ptype>);
int fileNums(0);
ptrtype readlas(string filepath)
{
	LASreadOpener lasreadopener;
	lasreadopener.set_file_name(filepath.c_str());
	LASreader* lasreader = lasreadopener.open();
	size_t count = lasreader->header.number_of_point_records;
	pcl::PointCloud<ptype>::Ptr pointCloudPtr(new pcl::PointCloud<ptype>);
	pointCloudPtr->resize(count);
	size_t j = 0;
	pts_sourceID = lasreader->point.get_point_source_ID();
	p_gpsT = lasreader->point.get_gps_time();
	while (lasreader->read_point() && j < count)
	{
		pointCloudPtr->points[j].x = lasreader->point.get_x();
		pointCloudPtr->points[j].y = lasreader->point.get_y();
		pointCloudPtr->points[j].z = lasreader->point.get_z();
		pointCloudPtr->points[j].intensity = lasreader->point.get_gps_time();
		j++;
	}
	pointCloudPtr->resize(j);
	pointCloudPtr->width = count;
	pointCloudPtr->height = 1;
	pointCloudPtr->is_dense = false;
	return pointCloudPtr;
}

void cloudvisual(ptrtype cloud, const char* name)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(name));

	pcl::visualization::PointCloudColorHandlerGenericField<ptype> fildColor(cloud, "z"); // 按照z字段进行渲染

	viewer->addPointCloud<ptype>(cloud, fildColor, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud"); // 设置点云大小

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(0);

	}
}
void cloudvisual2(ptrtype src, ptrtype tgt, const char* name)
{
	//创建视窗对象并给标题栏设置一个名称“3D Viewer”并将它设置为boost::shared_ptr智能共享指针，这样可以保证指针在程序中全局使用，而不引起内存错误
	pcl::visualization::PCLVisualizer viewer(name);
	//设置视窗的背景色，可以任意设置RGB的颜色，这里是设置为黑色
	viewer.setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<ptype> target_color(tgt, 0, 255, 0);
	//将点云添加到视窗对象中，并定一个唯一的字符串作为ID 号，利用此字符串保证在其他成员中也能标志引用该点云，多次调用addPointCloud可以实现多个点云的添加，每调用一次就会创建一个新的ID号，如果想更新一个已经显示的点云，先调用removePointCloud（），并提供需要更新的点云ID 号，也可使用updatePointCloud
	viewer.addPointCloud<ptype>(tgt, target_color, "target cloud");
	//用于改变显示点云的尺寸，可以利用该方法控制点云在视窗中的显示方法,1设置显示点云大小
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

	pcl::visualization::PointCloudColorHandlerCustom<ptype> source_color(src, 255, 0, 0);
	//将点云添加到视窗对象中，并定一个唯一的字符串作为ID 号，利用此字符串保证在其他成员中也能标志引用该点云，多次调用addPointCloud可以实现多个点云的添加，每调用一次就会创建一个新的ID号，如果想更新一个已经显示的点云，先调用removePointCloud（），并提供需要更新的点云ID 号，也可使用updatePointCloud
	viewer.addPointCloud<ptype>(src, source_color, "source cloud");
	//用于改变显示点云的尺寸，可以利用该方法控制点云在视窗中的显示方法,1设置显示点云大小
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(0);

	}
}
void thinImage(Mat & srcImg) {
	vector<Point> deleteList;
	int neighbourhood[9];
	int nl = srcImg.rows;
	int nc = srcImg.cols;
	bool inOddIterations = true;
	while (true) {
		for (int j = 1; j < (nl - 1); j++) {
			uchar* data_last = srcImg.ptr<uchar>(j - 1);
			uchar* data = srcImg.ptr<uchar>(j);
			uchar* data_next = srcImg.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++) {
				if (data[i] == 255) {
					int whitePointCount = 0;
					neighbourhood[0] = 1;
					if (data_last[i] == 255) neighbourhood[1] = 1;
					else  neighbourhood[1] = 0;
					if (data_last[i + 1] == 255) neighbourhood[2] = 1;
					else  neighbourhood[2] = 0;
					if (data[i + 1] == 255) neighbourhood[3] = 1;
					else  neighbourhood[3] = 0;
					if (data_next[i + 1] == 255) neighbourhood[4] = 1;
					else  neighbourhood[4] = 0;
					if (data_next[i] == 255) neighbourhood[5] = 1;
					else  neighbourhood[5] = 0;
					if (data_next[i - 1] == 255) neighbourhood[6] = 1;
					else  neighbourhood[6] = 0;
					if (data[i - 1] == 255) neighbourhood[7] = 1;
					else  neighbourhood[7] = 0;
					if (data_last[i - 1] == 255) neighbourhood[8] = 1;
					else  neighbourhood[8] = 0;
					for (int k = 1; k < 9; k++) {
						whitePointCount += neighbourhood[k];
					}
					if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
						int ap = 0;
						if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
						if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
						if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
						if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
						if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
						if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
						if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
						if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
						if (ap == 1) {
							if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
								deleteList.push_back(Point(i, j));
							}
							else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
								deleteList.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deleteList.size() == 0)
			break;
		for (size_t i = 0; i < deleteList.size(); i++) {
			Point tem;
			tem = deleteList[i];
			uchar* data = srcImg.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deleteList.clear();

		inOddIterations = !inOddIterations;
	}
}
int findmaxValue(Mat img, int rol, int col)
{
	int maxV = 0;
	for (int i = rol - 1; i <= rol + 1; i++)
	{
		for (int j = col - 1; j <= col + 1; j++)
		{
			if (i < 0 || j < 0) continue;
			if (maxV < img.ptr<uchar>(i)[j]) maxV = img.ptr<uchar>(i)[j];
		}
	}
	return maxV;
}

Vec2f roughRegist(ptrtype cloud_mls, ptrtype als_cloud, float mls_grid, float als_grid, float threholdDiff, float roofHeight)
{
	cloudvisual(als_cloud, "目标区域机载点云3D图"); //直接显示机载点云
	//////////////队车载点云滤波（过滤低于地表的噪声点）//////
	pcl::RadiusOutlierRemoval<ptype> outrem;  //创建滤波器
	pcl::PointCloud<ptype>::Ptr mls_cloud(new pcl::PointCloud<ptype>);
	outrem.setInputCloud(cloud_mls);    //设置输入点云
	outrem.setRadiusSearch(0.5);     //设置半径为0.5的范围内找临近点
	outrem.setMinNeighborsInRadius(4); //设置查询点的邻域点集数小于10的删除
	outrem.setNegative(false);
	// apply filter
	outrem.filter(*mls_cloud);     //执行条件滤波   在半径为0.8 在此半径内必须要有两个邻居点，此点才会保存
	std::cerr << "Cloud after filtering" << endl;
	std::cerr << mls_cloud->size() << endl;
	cloudvisual(mls_cloud, "目标区域车载点云3D图");

	//提取点云最值
	pcl::PointXYZI min_mls;
	pcl::PointXYZI max_mls;
	pcl::getMinMax3D(*mls_cloud, min_mls, max_mls);
	//计算区域内格网XY方向数量
	int width = int((max_mls.x - min_mls.x) / mls_grid) + 1;
	int height = int((max_mls.y - min_mls.y) / mls_grid) + 1;

	//构建二维平面格网
	flat_grid **voxel = new flat_grid*[width];
	for (int i = 0; i < width; ++i)
		voxel[i] = new flat_grid[height];
	int row, col;
	for (size_t i = 0; i < mls_cloud->points.size(); i++)
	{
		row = int((mls_cloud->points[i].x - min_mls.x) / mls_grid);
		col = int((mls_cloud->points[i].y - min_mls.y) / mls_grid);
		voxel[row][col].indexID.push_back(i);
		if (voxel[row][col].grayScale != 5)
			voxel[row][col].grayScale++;
	}
	for (int i = 0; i < width; i++)   //遍历格网
	{
		for (int j = 0; j < height; j++)
		{
			if (voxel[i][j].grayScale >= 1)
			{
				float  max_h = 0.0;
				for (int num = 0; num < voxel[i][j].indexID.size(); num++)
				{
					if (mls_cloud->points[voxel[i][j].indexID[num]].z > max_h)max_h = mls_cloud->points[voxel[i][j].indexID[num]].z;
				}
				voxel[i][j].region_difheight = max_h - min_mls.z;
			}
			if (voxel[i][j].grayScale == 5)         //提取非空格网数
			{
				int point_num = 0;
				float  max_h = 0.0;
				for (int num = 0; num < voxel[i][j].indexID.size(); num++)
				{
					if (mls_cloud->points[voxel[i][j].indexID[num]].z > max_h)max_h = mls_cloud->points[voxel[i][j].indexID[num]].z;
				}

				if ((max_h - min_mls.z) > threholdDiff)
				{
					voxel[i][j].candidate = 1;
				}
			}
		}
	}
	int num_continues = 0;
	vector<int>pointIndices;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (voxel[i][j].candidate == 1)
			{
				num_continues++;
				pcl::PointCloud<pcl::PointXYZI>::Ptr voxelPointCloudPtr(new pcl::PointCloud<pcl::PointXYZI>);   //构建格网点云集
				voxelPointCloudPtr->width = voxel[i][j].indexID.size();
				voxelPointCloudPtr->height = 1;
				voxelPointCloudPtr->is_dense = false;
				voxelPointCloudPtr->resize(voxelPointCloudPtr->width * voxelPointCloudPtr->height);
				for (size_t k = 0; k < voxelPointCloudPtr->points.size(); k++)     //读取格网点云数据
				{

					voxelPointCloudPtr->points[k].x = mls_cloud->points[voxel[i][j].indexID[k]].x;
					voxelPointCloudPtr->points[k].y = mls_cloud->points[voxel[i][j].indexID[k]].y;
					voxelPointCloudPtr->points[k].z = mls_cloud->points[voxel[i][j].indexID[k]].z;
				}
				pcl::PointXYZI min_voxel;
				pcl::PointXYZI max_voxel;
				pcl::getMinMax3D(*voxelPointCloudPtr, min_voxel, max_voxel);
				for (int num = 0; num < voxelPointCloudPtr->points.size(); num++)
				{
					if (fabs(voxelPointCloudPtr->points[num].z - max_voxel.z) < 0.3)
					{
						pointIndices.push_back(voxel[i][j].indexID[num]);
					}
				}
			}
		}
	}
	boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(pointIndices);
	pcl::ExtractIndices<pcl::PointXYZI> extract;
	// Extract the inliers
	pcl::PointCloud<ptype>::Ptr Mcloud_flitered(new pcl::PointCloud<ptype>);
	extract.setInputCloud(mls_cloud);
	extract.setIndices(index_ptr);
	extract.setNegative(false);//如果设为true,可以提取指定index之外的点云
	extract.filter(*Mcloud_flitered);
	
	//////////栅格化///////////
		//构建二维平面格网
	flat_grid **voxel_1 = new flat_grid*[width];
	for (int i = 0; i < width; ++i)
		voxel_1[i] = new flat_grid[height];
	for (size_t i = 0; i < Mcloud_flitered->points.size(); i++)
	{
		row = int((Mcloud_flitered->points[i].x - min_mls.x) / mls_grid);
		col = int((Mcloud_flitered->points[i].y - min_mls.y) / mls_grid);
		voxel_1[row][col].indexID.push_back(i);
		if (voxel_1[row][col].grayScale < 1)
			voxel_1[row][col].grayScale++;
	}
	float max_heightDiff = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (voxel_1[i][j].grayScale == 1)         //提取非空格网数
			{
				float max_height = -999.0;
				//float min_height = 1000.0;
				for (int num = 0; num < voxel_1[i][j].indexID.size(); num++)
				{
					if (Mcloud_flitered->points[voxel_1[i][j].indexID[num]].z > max_height) max_height = Mcloud_flitered->points[voxel_1[i][j].indexID[num]].z;
					//if (cloud_flitered->points[voxel_1[i][j].indexID[num]].z < min_height) min_height = cloud_flitered->points[voxel_1[i][j].indexID[num]].z;
				}
				// 提取格网最大高程值作为灰度（车载）
				voxel_1[i][j].max_height = max_height;
				// 提取格网高差作为灰度（机载） 
				voxel_1[i][j].region_difheight = max_height - min_mls.z;
				if (voxel_1[i][j].region_difheight > max_heightDiff) max_heightDiff = voxel_1[i][j].region_difheight;
			}
		}
	}
	// 机载点云灰度转换公式
	float scale_trans = 255.0 / max_heightDiff;
	//写入Mat图像
	Mat image(width, height, CV_8UC1, Scalar(0));
	uchar a1 = 0;
	for (int i = 0; i < width; i++)  //image.at<uchar>(i,j)
	{
		uchar* data = image.ptr<uchar>(i);
		for (int j = 0; j < height; j++)
		{
			if (voxel_1[i][j].grayScale != 0)
			{
				data[j] = uchar((voxel_1[i][j].region_difheight) * scale_trans);
				a1 = data[j] > a1 ? data[j] : a1;
			}
		}
	}

	
	////////骨架提取算法////////////
	Mat binaryImage(width, height, CV_8UC1);
	threshold(image, binaryImage, 20, 255, CV_THRESH_BINARY);
	thinImage(binaryImage);
	
	//////霍夫直线检测//////////
	vector<Vec4f> line_data;
	//输入图像，输出极坐标来表示直线，像素扫描步长，极坐标角度步长，
	//int类型的threshold，累加平面的阈值参数，即识别某部分为图中的一条直线时它在累加平面中必须达到的值(threhold越大，扫描出的直线局部密集度越小，局部分叉少
	//最小直线长度，连接线段上最近两点之间的阈值
	HoughLinesP(binaryImage, line_data, 1, CV_PI / 180.0, 40, 10, 5);
	vector<Vec4f> lines_2;
	///////因为通常边界会画出两条线，舍弃端点值没有灰度的线
	for (size_t i = 0; i < line_data.size(); i++) {
		Vec4f temp = line_data[i];
		int a = findmaxValue(image, int(temp[3]), int(temp[2]));
		int b = findmaxValue(image, int(temp[1]), int(temp[0]));
		//int a = image.at<uchar>(int(temp[3]), int(temp[2]));
		//int b = image.at<uchar>(int(temp[1]), int(temp[0]));
		if (a != 0 && b != 0)//&& fabs((a - b) / scale_trans) <= 10)
		{
			lines_2.push_back(temp);
		}

	}
	vector<Point_vec>mls_xyzline(lines_2.size());
	for (size_t i = 0; i < lines_2.size(); i++)
	{
		Vec4f temp = lines_2[i];
		int a = findmaxValue(image, int(temp[3]), int(temp[2]));
		int b = findmaxValue(image, int(temp[1]), int(temp[0]));
		float x_1 = int(temp[3]) * mls_grid + min_mls.x;
		float y_1 = int(temp[2]) * mls_grid + min_mls.y;
		float z_1 = a / scale_trans + min_mls.z;
		float x_2 = int(temp[1]) * mls_grid + min_mls.x;
		float y_2 = int(temp[0]) * mls_grid + min_mls.y;
		float z_2 = b / scale_trans + min_mls.z;
		float z = z_1 > z_2 ? z_1 : z_2;
		if (x_1 < x_2)
		{
			mls_xyzline[i].startpoint = pcl::PointXYZ(x_1, y_1, z);
			mls_xyzline[i].endpoint = pcl::PointXYZ(x_2, y_2, z);
		}
		else
		{
			mls_xyzline[i].startpoint = pcl::PointXYZ(x_2, y_2, z);
			mls_xyzline[i].endpoint = pcl::PointXYZ(x_1, y_1, z);
		}
	}
	Scalar color = Scalar(255);
	//写入DSM影像
	Mat MdsmImg(width, height, CV_8UC1, Scalar(0));
	for (int i = 0; i < width; i++)  //image.at<uchar>(i,j)
	{
		uchar* data = MdsmImg.ptr<uchar>(i);
		for (int j = 0; j < height; j++)
		{
			data[j] = uchar((voxel[i][j].region_difheight) * 255.0 / (max_mls.z - min_mls.z));
		}
	}
	Mat MdsmImg_1(MdsmImg);
	cvtColor(MdsmImg_1, MdsmImg, COLOR_GRAY2RGB);
	for (size_t i = 0; i < line_data.size(); i++) {
		Vec4f temp = line_data[i];
		//line(image, Point(temp[0], temp[1]), Point(temp[2], temp[3]), color, 1);
		line(MdsmImg, Point(temp[0], temp[1]), Point(temp[2], temp[3]), Scalar(0, 0, 255), 1);
	}

	namedWindow("车载点云特征线提取结果", CV_WINDOW_NORMAL);
	imshow("车载点云特征线提取结果", MdsmImg);
	/****************机载点云处理*******************/
	///提取格网内部最高点云///////
	//提取点云最值
	pcl::PointXYZI min_als;
	pcl::PointXYZI max_als;
	pcl::getMinMax3D(*als_cloud, min_als, max_als);
	//计算区域内格网XYZ方向数量
	int width_als = int((max_als.x - min_als.x) / als_grid) + 1;
	int height_als = int((max_als.y - min_als.y) / als_grid) + 1;

	//构建二维平面格网
	flat_grid **voxel_2 = new flat_grid*[width_als];
	for (int i = 0; i < width_als; ++i)
		voxel_2[i] = new flat_grid[height_als];
	int row_als, col_als;
	for (size_t i = 0; i < als_cloud->points.size(); i++)
	{
		row_als = int((als_cloud->points[i].x - min_als.x) / als_grid);
		col_als = int((als_cloud->points[i].y - min_als.y) / als_grid);
		voxel_2[row_als][col_als].indexID.push_back(i);
		if (voxel_2[row_als][col_als].grayScale < 1)
			voxel_2[row_als][col_als].grayScale++;
	}
	int count_grid = 0;
	vector<int>pointIndices_als;
	//提取屋顶边沿点云
	for (int i = 0; i < width_als; i++)
	{
		for (int j = 0; j < height_als; j++)
		{
			if (voxel_2[i][j].grayScale == 1)
			{
				count_grid++;
				pcl::PointCloud<pcl::PointXYZI>::Ptr voxelPointCloudPtr(new pcl::PointCloud<pcl::PointXYZI>);   //构建格网点云集
				voxelPointCloudPtr->width = voxel_2[i][j].indexID.size();
				voxelPointCloudPtr->height = 1;
				voxelPointCloudPtr->is_dense = false;
				voxelPointCloudPtr->resize(voxelPointCloudPtr->width * voxelPointCloudPtr->height);
				for (size_t k = 0; k < voxelPointCloudPtr->points.size(); k++)     //读取格网点云数据
				{

					voxelPointCloudPtr->points[k].x = als_cloud->points[voxel_2[i][j].indexID[k]].x;
					voxelPointCloudPtr->points[k].y = als_cloud->points[voxel_2[i][j].indexID[k]].y;
					voxelPointCloudPtr->points[k].z = als_cloud->points[voxel_2[i][j].indexID[k]].z;
				}
				pcl::PointXYZI min;
				pcl::PointXYZI max;
				pcl::getMinMax3D(*voxelPointCloudPtr, min, max);
				voxel_2[i][j].region_difheight = max.z - min_als.z;
				for (size_t k = 0; k < voxelPointCloudPtr->points.size(); k++)     //读取格网点云数据
				{
					if (voxelPointCloudPtr->points[k].z - min_als.z >= roofHeight) pointIndices_als.push_back(voxel_2[i][j].indexID[k]);;
				}
			}
		}
	}
	pcl::PointCloud<pcl::PointXYZI>::Ptr Acloud_flitered(new pcl::PointCloud<pcl::PointXYZI>);
	boost::shared_ptr<std::vector<int>> index_Aptr = boost::make_shared<std::vector<int>>(pointIndices_als);
	pcl::ExtractIndices<pcl::PointXYZI> Aextract;
	// Extract the inliers
	Aextract.setInputCloud(als_cloud);
	Aextract.setIndices(index_Aptr);
	Aextract.setNegative(false);
	Aextract.filter(*Acloud_flitered);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::RadiusOutlierRemoval<pcl::PointXYZI> sor;   //创建滤波器对象
	sor.setInputCloud(Acloud_flitered);                           //设置待滤波的点云
	sor.setRadiusSearch(4);                               //设置在进行统计时考虑查询点临近点数
	sor.setMinNeighborsInRadius(20); //设置查询点的邻域点集数小于2的删除
	sor.setKeepOrganized(false);  //如果设置为true,原文件的滤除点会被置为nan
	sor.filter(*cloud_1);                    //存储
	/*
	pcl::visualization::PCLVisualizer viewer_1("视窗1");
	viewer_1.setBackgroundColor(0, 0, 0);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
		target_Acolor(cloud_1, 205, 92, 92);

	viewer_1.addPointCloud<pcl::PointXYZI>(cloud_1, target_Acolor, "1");//显示点云，其中fildColor为颜色显示

	viewer_1.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "1");//设置点云大小
	while (!viewer_1.wasStopped())
	{
		viewer_1.spinOnce();
	}
	*/
	//////////栅格化///////////
	pcl::PointXYZI min_p;  //用于存放三个轴最小值
	pcl::PointXYZI max_p;
	pcl::getMinMax3D(*cloud_1, min_p, max_p);
	//构建二维平面格网
	flat_grid **voxel_3 = new flat_grid*[width_als];
	for (int i = 0; i < width_als; ++i)
		voxel_3[i] = new flat_grid[height_als];
	for (size_t i = 0; i < cloud_1->points.size(); i++)
	{
		float a = cloud_1->points[i].x;
		float b = cloud_1->points[i].y;
		row_als = int((cloud_1->points[i].x - min_als.x) / als_grid);
		col_als = int((cloud_1->points[i].y - min_als.y) / als_grid);
		voxel_3[row_als][col_als].indexID.push_back(i);
		if (voxel_3[row_als][col_als].grayScale < 1)
			voxel_3[row_als][col_als].grayScale++;
	}
	for (int i = 0; i < width_als; i++)
	{
		for (int j = 0; j < height_als; j++)
		{
			if (voxel_3[i][j].grayScale == 1)         //提取非空格网数
			{
				float max_height = -999.0;
				for (int num = 0; num < voxel_3[i][j].indexID.size(); num++)
				{
					if (cloud_1->points[voxel_3[i][j].indexID[num]].z > max_height) max_height = cloud_1->points[voxel_3[i][j].indexID[num]].z;
				}
				// 提取格网最大高程值作为灰度（车载）
				voxel_3[i][j].max_height = max_height;
				// 提取格网高差作为灰度（机载） 
				voxel_3[i][j].region_difheight = max_height - min_als.z;
			}
		}
	}
	// 机载点云灰度转换公式 float scale_trans = 255.0 / max_heightDiff;
	// 车载点云灰度转换公式
	float scale_Atrans = 255.0 / (max_als.z - min_als.z);
	//写入Mat图像
	Mat image_1(width_als, height_als, CV_8UC1, Scalar(0));
	for (int i = 0; i < width_als; i++)  //image.at<uchar>(i,j)
	{
		uchar* data = image_1.ptr<uchar>(i);
		for (int j = 0; j < height_als; j++)
		{
			if (voxel_3[i][j].grayScale != 0)
			{
				// 车载点云灰度赋值 
				data[j] = uchar(voxel_3[i][j].region_difheight * scale_Atrans);
				// 机载点云灰度赋值 data[j] = uchar((voxel_1[i][j].max_heightDiff) * scale_trans);
			}
		}
	}

	// Create and LSD detector with standard or no refinement.
#if 1
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
#else
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
#endif

	double start = double(getTickCount());
	vector<Vec4f> lines_std;
	vector<Vec4f> lines_1;
	ls->detect(image_1, lines_std);
	//写入DSM影像
	Mat AdsmImg(width_als, height_als, CV_8UC1, Scalar(0));
	for (int i = 0; i < width_als; i++)  //image.at<uchar>(i,j)
	{
		uchar* data = AdsmImg.ptr<uchar>(i);
		for (int j = 0; j < height_als; j++)
		{
			if (voxel_2[i][j].grayScale != 0)
			{
				// 车载点云灰度赋值 
				data[j] = uchar(voxel_2[i][j].region_difheight * scale_Atrans);
			}
		}
	}
	///////因为通常边界会画出两条线，舍弃端点值没有灰度的线
	for (size_t i = 0; i < lines_std.size(); i++) {
		Vec4f temp = lines_std[i];
		int a = findmaxValue(image_1, int(temp[3]), int(temp[2]));
		int b = findmaxValue(image_1, int(temp[1]), int(temp[0]));
		//int a = image.at<uchar>(int(temp[3]), int(temp[2]));
		//int b = image.at<uchar>(int(temp[1]), int(temp[0]));
		if (a != 0 || b != 0) //&& fabs((a - b) / scale_Atrans) <= 10)
		{
			lines_1.push_back(temp);
		}

	}
	vector<Point_vec> als_xyzline(lines_1.size());
	for (size_t i = 0; i < lines_1.size(); i++)
	{
		Vec4f temp = lines_1[i];
		int a = findmaxValue(image_1, int(temp[3]), int(temp[2]));
		int b = findmaxValue(image_1, int(temp[1]), int(temp[0]));
		float x_1 = int(temp[3]) * als_grid + min_als.x;
		float y_1 = int(temp[2]) * als_grid + min_als.y;
		float z_1 = a / scale_Atrans + min_als.z;
		float x_2 = int(temp[1]) * als_grid + min_als.x;
		float y_2 = int(temp[0]) * als_grid + min_als.y;
		float z_2 = b / scale_Atrans + min_als.z;
		float z = z_1 > z_2 ? z_1 : z_2;
		if (x_1 < x_2)
		{
			als_xyzline[i].startpoint = pcl::PointXYZ(x_1, y_1, z);
			als_xyzline[i].endpoint = pcl::PointXYZ(x_2, y_2, z);
		}
		else
		{
			als_xyzline[i].startpoint = pcl::PointXYZ(x_2, y_2, z);
			als_xyzline[i].endpoint = pcl::PointXYZ(x_1, y_1, z);
		}
	}
	Mat AdsmImg_1(AdsmImg);
	ls->drawSegments(AdsmImg_1, lines_1);
	namedWindow("机载点云特征线提取结果", CV_WINDOW_AUTOSIZE);
	imshow("机载点云特征线提取结果", AdsmImg_1);
	/*
	for (int i = 0; i < width_als; ++i)
		delete[] voxel_2[i];
	delete[] voxel_2;
	for (int i = 0; i < width_als; ++i)
		delete[] voxel_3[i];
	delete[] voxel_3;
	*/

	/*******线特征匹配***********/
	vector<Vec6f>mls_lines, als_lines;
	for (size_t i = 0; i < lines_2.size(); i++)    //lines_2代表提出的车载特征线
	{
		Vec4f temp = lines_2[i];
		if (temp[1] > temp[3])
		{
			float x_start = int(temp[3]) * mls_grid + min_mls.x;
			float y_start = int(temp[2]) * mls_grid + min_mls.y;
			float x_end = int(temp[1]) * mls_grid + min_mls.x;
			float y_end = int(temp[0]) * mls_grid + min_mls.y;
			float angle = atan((y_end - y_start) / (x_end - x_start)) / 3.14 * 180.0;
			float lineLength = sqrt((x_end - x_start) * (x_end - x_start) + (y_end - y_start) * (y_end - y_start));
			Vec6f keyline(x_start, y_start, x_end, y_end, angle, lineLength);
			mls_lines.push_back(keyline);
		}
		else
		{
			float x_start = int(temp[1]) * mls_grid + min_mls.x;
			float y_start = int(temp[0]) * mls_grid + min_mls.y;
			float x_end = int(temp[3]) * mls_grid + min_mls.x;
			float y_end = int(temp[2]) * mls_grid + min_mls.y;
			float angle = atan((y_end - y_start) / (x_end - x_start)) / 3.14 * 180.0;
			float lineLength = sqrt((x_end - x_start) * (x_end - x_start) + (y_end - y_start) * (y_end - y_start));
			Vec6f keyline(x_start, y_start, x_end, y_end, angle, lineLength);
			mls_lines.push_back(keyline);
		}
	}
	for (size_t i = 0; i < lines_1.size(); i++)   //lines_1代表提取出的机载特征线
	{
		Vec4f temp = lines_1[i];
		if (temp[1] > temp[3])
		{
			float x_start = int(temp[3]) * als_grid + min_als.x;
			float y_start = int(temp[2]) * als_grid + min_als.y;
			float x_end = int(temp[1]) * als_grid + min_als.x;
			float y_end = int(temp[0]) * als_grid + min_als.y;
			float angle = atan((y_end - y_start) / (x_end - x_start)) / 3.14 * 180.0;
			float lineLength = sqrt((x_end - x_start) * (x_end - x_start) + (y_end - y_start) * (y_end - y_start));
			Vec6f keyline(x_start, y_start, x_end, y_end, angle, lineLength);
			als_lines.push_back(keyline);
		}
		else
		{
			float x_start = int(temp[1]) * als_grid + min_als.x;
			float y_start = int(temp[0]) * als_grid + min_als.y;
			float x_end = int(temp[3]) * als_grid + min_als.x;
			float y_end = int(temp[2]) * als_grid + min_als.y;
			float angle = atan((y_end - y_start) / (x_end - x_start)) / 3.14 * 180.0;
			float lineLength = sqrt((x_end - x_start) * (x_end - x_start) + (y_end - y_start) * (y_end - y_start));
			Vec6f keyline(x_start, y_start, x_end, y_end, angle, lineLength);
			als_lines.push_back(keyline);
		}
	}
	///以车载点云特征线为基础寻找同名线对
	vector<lines_combination> Mmatch(mls_lines.size());
	for (size_t i = 0; i < mls_lines.size(); i++)
	{
		float mls_angle = mls_lines[i][4];
		float mls_length = mls_lines[i][5];
		for (int j = 0; j < als_lines.size(); j++)
		{
			if (abs(als_lines[j][4] - mls_angle) < 5)
			{
				float als_length = als_lines[j][5];
				if ((mls_length / als_length) > 0.67 && (mls_length / als_length) < 1.5 && fabs(mls_lines[i][0] - als_lines[j][0]) < 100 && fabs(mls_lines[i][1] - als_lines[j][1]) < 100)
				{
					Vec2f temp;
					temp[0] = mls_lines[i][0] - als_lines[j][0];
					temp[1] = mls_lines[i][1] - als_lines[j][1];

					Mmatch[i].als_index.push_back(j);
					Mmatch[i].xy_drift.push_back(temp);
				}
			}
		}

	}
	////寻找最佳的匹配组合
	vector <int> vaild_index; //vaild_index为存储有候选机载特征线匹配的车载特征线序号
	double num_match = 1;  //num_match为所有有效地匹配可能情况
	for (int i = 0; i < mls_lines.size(); i++)
	{
		if (Mmatch[i].als_index.size() != 0)
		{
			num_match *= Mmatch[i].als_index.size();
			vaild_index.push_back(i);
		}
	}
	if (vaild_index.size() == 0) return { 0, 0 };
	//构建一个有对应情况的存储数组
	vector<lines_combination> Mmatch1(vaild_index.size());
	for (int i = 0; i < vaild_index.size(); i++)
	{
		Mmatch1[i] = Mmatch[vaild_index[i]];
	}
	//每种组合的XY距离方差存储数组
	//vector<Vec2f> lines_xydistance(num_match);
	vector<float> lines_distance(num_match);
	//遍历所有可能的排列情况
	for (int nums = 0; nums < num_match; nums++)
	{
		int index_number = nums;
		//将组合序号与组合情况对应起来，计算每种组合的距离方差
		vector<int> corresepond_vec(vaild_index.size()); //vaild_index.size等于可匹配的车载特征线
		for (int i = vaild_index.size() - 1; i > 0; i--)
		{
			corresepond_vec[i] = index_number % Mmatch1[i].als_index.size();
			index_number = int(index_number / Mmatch1[i].als_index.size());
		}
		corresepond_vec[0] = index_number; //correspond_vec向量组为数字num_match对应于匹配组合的序号，类似于010在Match1 = {1,2,3}对应3.

		//根据导出的特征线组合计算距离方差
		float diff_distance = 0;
		float diff_xdistance = 0;
		float diff_ydistance = 0;
		//计算XY距离平均值
		float x_distance = 0;
		float y_distance = 0;
		for (int i = 0; i < corresepond_vec.size(); i++)
		{
			Vec2f tmep = Mmatch1[i].xy_drift[corresepond_vec[i]];
			x_distance += tmep[0];
			y_distance += tmep[1];
		}
		//计算偏移均值
		float x_mean = x_distance / corresepond_vec.size();
		float y_mean = y_distance / corresepond_vec.size();
		//计算XY方向的方差
		for (int i = 0; i < corresepond_vec.size(); i++)
		{
			Vec2f tmep = Mmatch1[i].xy_drift[corresepond_vec[i]];
			diff_xdistance += pow((x_mean - tmep[0]), 2);
			diff_ydistance += pow((y_mean - tmep[1]), 2);
		}
		diff_xdistance = diff_xdistance / corresepond_vec.size();
		diff_ydistance = diff_ydistance / corresepond_vec.size();
		diff_distance = sqrt(pow(diff_xdistance, 2) + pow(diff_ydistance, 2));
		lines_distance[nums] = diff_distance;
		//lines_xydistance[nums] = {diff_xdistance, diff_ydistance};
	}

	//从各种情况中找到距离方差最小的情况
	auto min_distance = min_element(lines_distance.begin(), lines_distance.end());
	int index_mindiff = distance(begin(lines_distance), min_distance);

	//对求得的最小距离方差进行偏移量优化
	int mindiffID = index_mindiff;
	//将组合序号与组合情况对应起来，计算每种组合的距离方差
	vector<int> mindiff_vec(vaild_index.size());
	for (int i = vaild_index.size() - 1; i > 0; i--)
	{
		mindiff_vec[i] = mindiffID % Mmatch1[i].als_index.size();
		mindiffID = int(mindiffID / Mmatch1[i].als_index.size());
	}
	mindiff_vec[0] = mindiffID;
	//利用带权平差法迭代精化
	float minx_distance = 0;
	float miny_distance = 0;
	for (int i = 0; i < mindiff_vec.size(); i++)
	{
		Vec2f tmep = Mmatch1[i].xy_drift[mindiff_vec[i]];
		minx_distance += tmep[0];
		miny_distance += tmep[1];
	}
	float x_mean = minx_distance / mindiff_vec.size();
	float y_mean = miny_distance / mindiff_vec.size();
	float xy_mean = sqrt(pow(x_mean, 2) + pow(y_mean, 2));
	//计算XY方向的方差
	//构建权矩阵
	vector<float> weight(mindiff_vec.size());
	vector<float> diff(mindiff_vec.size());
	float diff_minxydistance = 0;
	for (int i = 0; i < mindiff_vec.size(); i++)
	{
		Vec2f tmep = Mmatch1[i].xy_drift[mindiff_vec[i]];
		diff[i] = 1.0 / (sqrt(pow(x_mean - tmep[0], 2) + pow(y_mean - tmep[1], 2)) + 0.00000000001);
		diff_minxydistance += diff[i];
	}
	for (int i = 0; i < mindiff_vec.size(); i++)
	{
		weight[i] = diff[i] / diff_minxydistance;
	}
	float x1_mean = 0;
	float y1_mean = 0;
	float xy1_mean = 0;
	for (int i = 0; i < mindiff_vec.size(); i++)
	{
		x1_mean += weight[i] * Mmatch1[i].xy_drift[mindiff_vec[i]][0];
		y1_mean += weight[i] * Mmatch1[i].xy_drift[mindiff_vec[i]][1];
	}
	x1_mean = x1_mean;
	y1_mean = y1_mean;
	xy1_mean = sqrt(pow(x1_mean, 2) + pow(y1_mean, 2));
	//构建权矩阵
	while (fabs(xy1_mean - xy_mean) > 0.1)
	{
		x_mean = x1_mean;
		y_mean = y1_mean;
		xy_mean = xy1_mean;
		//计算XY方向的方差
		float diff_minxydistance = 0;
		for (int i = 0; i < mindiff_vec.size(); i++)
		{
			Vec2f tmep = Mmatch1[i].xy_drift[mindiff_vec[i]];
			diff[i] = 1.0 / (sqrt(pow(x_mean - tmep[0], 2) + pow(y_mean - tmep[1], 2)) + 0.00000000001);
			diff_minxydistance += diff[i];
		}
		for (int i = 0; i < mindiff_vec.size(); i++)
		{
			weight[i] = diff[i] / diff_minxydistance;
		}
		float x1_mean = 0;
		float y1_mean = 0;
		for (int i = 0; i < mindiff_vec.size(); i++)
		{
			x1_mean += weight[i] * Mmatch1[i].xy_drift[mindiff_vec[i]][0];
			y1_mean += weight[i] * Mmatch1[i].xy_drift[mindiff_vec[i]][1];
		}
		x1_mean = x1_mean;
		y1_mean = y1_mean;
		xy1_mean = sqrt(pow(x1_mean, 2) + pow(y1_mean, 2));
	}
	
	mls_rect->resize(mls_cloud->points.size());
	mls_rect->width = mls_cloud->points.size();
	mls_rect->height = 1;
	mls_rect->is_dense = false;
	for (int i = 0; i < mls_cloud->points.size(); i++)
	{
		mls_rect->points[i].x = mls_cloud->points[i].x - x1_mean;
		mls_rect->points[i].y = mls_cloud->points[i].y - y1_mean;
		mls_rect->points[i].z = mls_cloud->points[i].z;
		mls_rect->points[i].intensity = mls_cloud->points[i].intensity;
	}
	
	return { -x1_mean, -y1_mean };
}

ptrtype pointfilter_wall(ptrtype cloud, float gsd = 0.4)
{
	//计算二维格网数量
	ptype pmin;
	ptype pmax;
	pcl::getMinMax3D(*cloud, pmin, pmax);
	//cout << "输入格网间隔：" << endl;
	//cin >> gsd;
	int rows = int((pmax.x - pmin.x) / gsd) + 1;
	int cols = int((pmax.y - pmin.y) / gsd) + 1;
	//创建二维格网
	voxel **cloud_voxel = new voxel*[rows];
	for (int i = 0; i < rows; i++) {
		cloud_voxel[i] = new voxel[cols];
	}
	//遍历搜索点云，放入格网
	for (int i = 0; i < cloud->points.size(); i++)
	{
		int row = int((cloud->points[i].x - pmin.x) / gsd);
		int col = int((cloud->points[i].y - pmin.y) / gsd);
		cloud_voxel[row][col].index.push_back(i);
	}
	pcl::PointCloud<ptype>::Ptr pointCloudPtr(new pcl::PointCloud<ptype>);
	vector<int>candidate_point;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (cloud_voxel[i][j].index.size() >= 2)
			{
				float maxz = -100.0;
				float minz = 1000;
				for (int k = 0; k < cloud_voxel[i][j].index.size(); k++)
				{
					if (maxz < cloud->points[cloud_voxel[i][j].index[k]].z) maxz = cloud->points[cloud_voxel[i][j].index[k]].z;
					if (minz > cloud->points[cloud_voxel[i][j].index[k]].z) minz = cloud->points[cloud_voxel[i][j].index[k]].z;
				}
				cloud_voxel[i][j].max_heightdiff = maxz - minz;
				if (cloud_voxel[i][j].max_heightdiff > 15)
				{
					for (int k = 0; k < cloud_voxel[i][j].index.size(); k++)
					{
						candidate_point.push_back(cloud_voxel[i][j].index[k]);
					}

				}
			}

		}
	}
	//根据索引滤除点云
	boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(candidate_point);
	pcl::PointCloud<ptype>::Ptr target(new pcl::PointCloud<ptype>);
	pcl::ExtractIndices<ptype> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(index_ptr);
	extract.setNegative(false);
	extract.filter(*target);
	for (int i = 0; i < rows; i++) {
		delete[] cloud_voxel[i];
	}
	delete[] cloud_voxel;
	return target;
}

ptrtype pointfilter_road(ptrtype cloud, float distThreshold = 5)
{

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<ptype> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(distThreshold);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);


	pcl::PointCloud<ptype>::Ptr target(new pcl::PointCloud<ptype>);
	pcl::ExtractIndices<ptype> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*target);
	return target;
}

vector<float> PairwiseICP(pcl::PointCloud<ptype>::Ptr &src, pcl::PointCloud<ptype>::Ptr &tgt, int max_iter, const char* name)
{
	CNewDlgtip2 *m_pTipDlg = new CNewDlgtip2() ;
	m_pTipDlg->Create(IDD_Dlgtip2);
	m_pTipDlg->ShowWindow(SW_SHOW);
	UpdateWindow(false);
	pcl::PointCloud<ptype>::Ptr output(new pcl::PointCloud<ptype>);
	vector<float> trans(3);
	pcl::PointCloud<ptype>::Ptr cloud_tr(new pcl::PointCloud<ptype>);
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

	pcl::IterativeClosestPoint<ptype, ptype> icp;
	icp.setMaximumIterations(max_iter);
	icp.setInputSource(src);
	icp.setInputTarget(tgt);
	icp.setEuclideanFitnessEpsilon(0.01);//前后两次迭代误差的差值
	//icp.setTransformationEpsilon(1e-10); //上次转换与当前转换的差值；
	icp.setMaxCorrespondenceDistance(10); //忽略在此距离之外的点，对配准影响较大
	icp.align(*src);
	m_pTipDlg->DestroyWindow();
	cloudvisual2(tgt, src, name);
	Eigen::Matrix4f Mtransformation = icp.getFinalTransformation();
	trans[0] = Mtransformation(0, 3);
	trans[1] = Mtransformation(1, 3);
	trans[2] = Mtransformation(2, 3);
	return trans;
}
void centralize(ptrtype mls_vscloud, ptrtype als_vscloud)
{
	 pmin = mls_vscloud->points[0];

	for (size_t i = 0; i < mls_vscloud->points.size(); i++)
	{
		mls_vscloud->points[i].x -= pmin.x;
		mls_vscloud->points[i].y -= pmin.y;
		mls_vscloud->points[i].z -= pmin.z;
	}
	for (size_t i = 0; i < als_vscloud->points.size(); i++)
	{
		als_vscloud->points[i].x -= pmin.x;
		als_vscloud->points[i].y -= pmin.y;
		als_vscloud->points[i].z -= pmin.z;
	}
}

void differ_offset(const char* als_path, const char* mls_path)
{
	//读取las文件
	//vector<float> offset;
	als_cloud = readlas(als_path);
	mls_cloud = readlas(mls_path);

	//提取车载点云GPS时间
	int pSize = mls_cloud->size();
	float maxT(0), minT(1e10);
	for (int i = 0; i < pSize; i++)
	{
		if (mls_cloud->points[i].intensity > maxT) maxT = mls_cloud->points[i].intensity;
		if (mls_cloud->points[i].intensity < minT) minT = mls_cloud->points[i].intensity;
	}
	float timeC = (maxT + minT) / 2;
	gps_time = timeC;
	//offset.push_back(0);
	//offset.push_back(0);
	///////////////////粗配准//////////////////////////

	CNewLineExtract Dlg_2;
	Dlg_2.DoModal();

	xy_drift = roughRegist(mls_cloud, als_cloud, Dlg_2.mGridDist, Dlg_2.aGridDist, Dlg_2.mHeightDiff, Dlg_2.AheightDiff);
	cloudvisual2(mls_rect, als_cloud, "粗配准界面");
	CNewDlgtip1 tipDlg_1;
	tipDlg_1.DoModal();
}


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CptsRegistMfcDlg 对话框



CptsRegistMfcDlg::CptsRegistMfcDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_PTSREGISTMFC_DIALOG, pParent)
	, differ_value(_T(""))
{
	//m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_hIcon = AfxGetApp()->LoadIcon(IDI_ICON1);
}

void CptsRegistMfcDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, differ_value);
}

BEGIN_MESSAGE_MAP(CptsRegistMfcDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CptsRegistMfcDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CptsRegistMfcDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CptsRegistMfcDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CptsRegistMfcDlg::OnBnClickedButton4)
END_MESSAGE_MAP()


// CptsRegistMfcDlg 消息处理程序

BOOL CptsRegistMfcDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CptsRegistMfcDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CptsRegistMfcDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CptsRegistMfcDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CptsRegistMfcDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	CNewfileIn dlg_1;
	dlg_1.DoModal();
	for (int i = 0; i < dlg_1.mls_file.size(); i++)
	{
		mls_1.push_back(dlg_1.mls_file[i]);
		als_1.push_back(dlg_1.als_file[i]);
	}
	file_exist = true;
	CNewtip dlg_2;
	dlg_2.DoModal();
}


void CptsRegistMfcDlg::OnBnClickedButton2()
{
	// TODO: 粗配准
	if (file_exist == false)
	{
		CfileError dlg_1;
		dlg_1.DoModal();
	}
	if (fileNums < mls_1.size())
	{
		const char *mls_path = mls_1[fileNums].c_str();
		const char *als_path = als_1[fileNums].c_str();
		differ_offset(als_path, mls_path);
		hasRoughed = true;
	}
}


void CptsRegistMfcDlg::OnBnClickedButton3()
{
	// TODO: 输出文件
	if (hasPrecised == false)
	{
		CfileError dlg_1;
		dlg_1.DoModal();
	}
	char* doc_path = "output.txt";
	ofstream fout(doc_path);
	if (fout)
	{
		fout << mls_1.size() << endl;
	}
	for (int i = 0; i < mls_1.size(); i++)
	{
		fout << fixed << output[i][0] << "          " << output[i][1] << "         " << output[i][2] << "         " << output[i][3] << endl;
	}
	fout.close();
	WinExec("notepad.exe output.txt", SW_SHOW);
}


void CptsRegistMfcDlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	////////////////精配准/////////////////////
	if (hasRoughed == false)
	{
		CRegistError dlg_1;
		dlg_1.DoModal();
	}
	CNewIcpDlg Dlg_1;
	Dlg_1.DoModal();
	centralize(mls_rect, als_cloud);

	pcl::PointCloud<ptype>::Ptr als_filter(new pcl::PointCloud<ptype>);
	pcl::PointCloud<ptype>::Ptr mls_filter(new pcl::PointCloud<ptype>);

	als_filter = pointfilter_wall(als_cloud, Dlg_1.mGridDist);
	mls_filter = pointfilter_wall(mls_rect, Dlg_1.mGridDist);
	pcl::io::savePCDFileBinary("awall.pcd", *als_filter);
	pcl::io::savePCDFileBinary("mwall.pcd", *mls_filter);
	pcl::PointCloud<ptype>::Ptr als_road(new pcl::PointCloud<ptype>);
	pcl::PointCloud<ptype>::Ptr mls_road(new pcl::PointCloud<ptype>);
	als_road = pointfilter_road(als_cloud, Dlg_1.RDistThreshold);
	mls_road = pointfilter_road(mls_rect, Dlg_1.RDistThreshold);
	pcl::io::savePCDFileBinary("aroad.pcd", *als_road);
	pcl::io::savePCDFileBinary("mroad.pcd", *mls_road);
	cloudvisual(als_road,"a");
	cloudvisual(mls_road, "m");
	//ICP配准 
	vector<float> trans, trans1;
	vector<float> offset;
	trans = PairwiseICP(mls_filter, als_filter, Dlg_1.wMaxiter, "XY配准可视化界面");
	//trans1 = Cicp_1(mls_road, als_road);
	trans1 = PairwiseICP(mls_road, als_road, Dlg_1.RmaxIter, "高程配准可视化界面");  //车载往机载上配效果好于机载往车载上配 车载0.4，机载1.1
	
	for (size_t i = 0; i < mls_rect->points.size(); i++)
	{
		mls_rect->points[i].x += trans[0];
		mls_rect->points[i].y += trans[1];
		mls_rect->points[i].z += trans1[2];
	}
	/*补充*/
	vector<float> trans2;
	mls_filter = pointfilter_wall(mls_rect, Dlg_1.mGridDist);
	trans2 = PairwiseICP(mls_filter, als_filter, 10, "再配准可视化界面");
	for (size_t i = 0; i < mls_rect->points.size(); i++)
	{
		mls_rect->points[i].x += trans2[0];
		mls_rect->points[i].y += trans2[1];
	}

	offset.push_back(trans[0] + xy_drift[0] + trans2[0]);
	offset.push_back(trans[1] + xy_drift[1] + trans2[1]);
	offset.push_back(trans1[2]);
	//可视化
	cloudvisual2(mls_rect, als_cloud, "最终配准三维视图");
	//存入las文件
	const char savefile[] = "mls_rect.las";
	LASwriteOpener lasWriterOpener;
	lasWriterOpener.set_file_name(savefile);
	//init header
	LASheader lasHeader;
	lasHeader.x_scale_factor = 0.0001;
	lasHeader.y_scale_factor = 0.0001;
	lasHeader.z_scale_factor = 0.0001;
	lasHeader.x_offset = mls_rect->points[0].x +pmin.x;
	lasHeader.y_offset = mls_rect->points[0].y +pmin.y;
	lasHeader.z_offset = mls_rect->points[0].z +pmin.z;
	lasHeader.point_data_format = 3;
	lasHeader.point_data_record_length = 34;

	//open laswriter
	LASwriter* lasWriter = lasWriterOpener.open(&lasHeader);

	// init point
	LASpoint lasPoint;
	lasPoint.init(&lasHeader, lasHeader.point_data_format, lasHeader.point_data_record_length, 0);

	// write points
	double minX = DBL_MAX, minY = DBL_MAX, minZ = DBL_MAX;
	double maxX = -DBL_MAX, maxY = -DBL_MAX, maxZ = -DBL_MAX;
	for (int i = 0; i < mls_rect->size(); i++)
	{
		// populate the point
		lasPoint.set_x(mls_rect->points[i].x +pmin.x);
		lasPoint.set_y(mls_rect->points[i].y +pmin.y);
		lasPoint.set_z(mls_rect->points[i].z +pmin.z);
		lasPoint.set_intensity(0);
		lasPoint.set_point_source_ID(pts_sourceID);
		lasPoint.set_gps_time(p_gpsT);
		lasPoint.set_R(0);
		lasPoint.set_G(0);
		lasPoint.set_B(0);
		lasPoint.set_classification(0);

		// write the point
		lasWriter->write_point(&lasPoint);

		// add it to the inventory
		lasWriter->update_inventory(&lasPoint);

		//range
		double x = mls_rect->points[i].x +pmin.x;
		double y = mls_rect->points[i].y + pmin.y;
		double z = mls_rect->points[i].z +pmin.z;
		if (x < minX) minX = x;
		if (x > maxX) maxX = x;
		if (y < minY) minY = y;
		if (y > maxY) maxY = y;
		if (z < minZ) minZ = z;
		if (z > maxZ) maxZ = z;
	}

	// update the boundary
	lasHeader.set_bounding_box(minX, minY, minZ, maxX, maxY, maxZ);

	// update the header
	lasWriter->update_header(&lasHeader, true);

	// close the writer
	lasWriter->close();
	delete lasWriter;
	lasWriter = nullptr;

	differ_x1 = offset[0], differ_y1 = offset[1], differ_z1 = offset[2];
	output.push_back({ gps_time, differ_x1, differ_y1, differ_z1 });
	CString x1, y1, z1, ID_number;
	ID_number.Format(_T("          %d           "), fileNums + 1), x1.Format(_T("%f            "), differ_x1), y1.Format(_T("%f            "), differ_y1), z1.Format(_T("%f           "), differ_z1);
	differ_value += ID_number + x1 + y1 + z1;
	differ_value += "\r\n";
	UpdateData(false);
	fileNums++;
	hasRoughed = false;
	if(fileNums==mls_1.size()) hasPrecised = true;
}
