// CNewLineExtract.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CNewLineExtract.h"
#include "afxdialogex.h"


// CNewLineExtract 对话框

IMPLEMENT_DYNAMIC(CNewLineExtract, CDialogEx)

CNewLineExtract::CNewLineExtract(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_LineExtract, pParent)
	, mGridDist(0.6)
	, mHeightDiff(25)
	, aGridDist(0.9)
	, AheightDiff(25)
{

}

CNewLineExtract::~CNewLineExtract()
{
}

void CNewLineExtract::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, mGridDist);
	DDX_Text(pDX, IDC_EDIT2, mHeightDiff);
	DDX_Text(pDX, IDC_EDIT3, aGridDist);
	DDX_Text(pDX, IDC_EDIT4, AheightDiff);
}


BEGIN_MESSAGE_MAP(CNewLineExtract, CDialogEx)
	ON_EN_CHANGE(IDC_EDIT1, &CNewLineExtract::OnEnChangeEdit1)
END_MESSAGE_MAP()


// CNewLineExtract 消息处理程序


void CNewLineExtract::OnEnChangeEdit1()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}
