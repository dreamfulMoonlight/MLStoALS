// CNewIcpDlg.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CNewIcpDlg.h"
#include "afxdialogex.h"


// CNewIcpDlg 对话框

IMPLEMENT_DYNAMIC(CNewIcpDlg, CDialogEx)

CNewIcpDlg::CNewIcpDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_icpDlg, pParent)
	, mGridDist(0.4)
	, wMaxiter(30)
	, RDistThreshold(3)
	, RmaxIter(10)
{

}

CNewIcpDlg::~CNewIcpDlg()
{
}

void CNewIcpDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, mGridDist);
	DDX_Text(pDX, IDC_EDIT2, wMaxiter);
	DDX_Text(pDX, IDC_EDIT3, RDistThreshold);
	DDX_Text(pDX, IDC_EDIT4, RmaxIter);
}


BEGIN_MESSAGE_MAP(CNewIcpDlg, CDialogEx)
	ON_BN_CLICKED(IDOK, &CNewIcpDlg::OnBnClickedOk)
	ON_EN_CHANGE(IDC_EDIT1, &CNewIcpDlg::OnEnChangeEdit1)
END_MESSAGE_MAP()


// CNewIcpDlg 消息处理程序


void CNewIcpDlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}


void CNewIcpDlg::OnEnChangeEdit1()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}
