// CoutputError.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CoutputError.h"
#include "afxdialogex.h"


// CoutputError 对话框

IMPLEMENT_DYNAMIC(CoutputError, CDialogEx)

CoutputError::CoutputError(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_outputError, pParent)
{

}

CoutputError::~CoutputError()
{
}

void CoutputError::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CoutputError, CDialogEx)
END_MESSAGE_MAP()


// CoutputError 消息处理程序
