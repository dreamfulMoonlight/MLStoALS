// CNewDlgtip2.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CNewDlgtip2.h"
#include "afxdialogex.h"


// CNewDlgtip2 对话框

IMPLEMENT_DYNAMIC(CNewDlgtip2, CDialogEx)

CNewDlgtip2::CNewDlgtip2(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_Dlgtip2, pParent)
{

}

CNewDlgtip2::~CNewDlgtip2()
{
}

void CNewDlgtip2::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CNewDlgtip2, CDialogEx)
	ON_BN_CLICKED(IDCANCEL, &CNewDlgtip2::OnBnClickedCancel)
END_MESSAGE_MAP()


// CNewDlgtip2 消息处理程序


void CNewDlgtip2::OnBnClickedCancel()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
}
