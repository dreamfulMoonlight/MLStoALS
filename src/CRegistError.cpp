// CRegistError.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CRegistError.h"
#include "afxdialogex.h"


// CRegistError 对话框

IMPLEMENT_DYNAMIC(CRegistError, CDialogEx)

CRegistError::CRegistError(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_RegistError, pParent)
{

}

CRegistError::~CRegistError()
{
}

void CRegistError::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CRegistError, CDialogEx)
END_MESSAGE_MAP()


// CRegistError 消息处理程序
