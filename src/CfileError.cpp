// CfileError.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CfileError.h"
#include "afxdialogex.h"


// CfileError 对话框

IMPLEMENT_DYNAMIC(CfileError, CDialogEx)

CfileError::CfileError(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_fileError, pParent)
{

}

CfileError::~CfileError()
{
}

void CfileError::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CfileError, CDialogEx)
END_MESSAGE_MAP()


// CfileError 消息处理程序
