// CNewtip.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CNewtip.h"
#include "afxdialogex.h"


// CNewtip 对话框

IMPLEMENT_DYNAMIC(CNewtip, CDialogEx)

CNewtip::CNewtip(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_TipDlg, pParent)
{

}

CNewtip::~CNewtip()
{
}

void CNewtip::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CNewtip, CDialogEx)
	ON_BN_CLICKED(IDCANCEL, &CNewtip::OnBnClickedCancel)
	ON_BN_CLICKED(IDOK, &CNewtip::OnBnClickedOk)
END_MESSAGE_MAP()


// CNewtip 消息处理程序


void CNewtip::OnBnClickedCancel()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
	PostQuitMessage(0);//最常用
}


void CNewtip::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}
