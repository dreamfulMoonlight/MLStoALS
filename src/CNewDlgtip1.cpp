// CNewDlgtip1.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CNewDlgtip1.h"
#include "afxdialogex.h"


// CNewDlgtip1 对话框

IMPLEMENT_DYNAMIC(CNewDlgtip1, CDialogEx)

CNewDlgtip1::CNewDlgtip1(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_Dlgtip1, pParent)
{

}

CNewDlgtip1::~CNewDlgtip1()
{
}

void CNewDlgtip1::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CNewDlgtip1, CDialogEx)
	ON_BN_CLICKED(IDCANCEL, &CNewDlgtip1::OnBnClickedCancel)
END_MESSAGE_MAP()


// CNewDlgtip1 消息处理程序


void CNewDlgtip1::OnBnClickedCancel()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
	PostQuitMessage(0);//最常用
}
