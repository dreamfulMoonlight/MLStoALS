// CNewfileIn.cpp: 实现文件
//

#include "pch.h"
#include "MFCptsRegist.h"
#include "CNewfileIn.h"
#include "afxdialogex.h"


// CNewfileIn 对话框

IMPLEMENT_DYNAMIC(CNewfileIn, CDialogEx)

CNewfileIn::CNewfileIn(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_fileInDlg, pParent)
	, filepath(_T(""))
{

}

CNewfileIn::~CNewfileIn()
{
}

void CNewfileIn::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT1, mls_path);
	DDX_Control(pDX, IDC_EDIT2, als_path);
	DDX_Text(pDX, IDC_EDIT3, filepath);
}


BEGIN_MESSAGE_MAP(CNewfileIn, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON1, &CNewfileIn::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON3, &CNewfileIn::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON2, &CNewfileIn::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON4, &CNewfileIn::OnBnClickedButton4)
END_MESSAGE_MAP()


// CNewfileIn 消息处理程序


void CNewfileIn::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	CFileDialog dlg(TRUE/*这个参数为TRUE就是“打开”对话框，为FALSE就是“保存”对话框*/, NULL/*默认文件类型*/, NULL/*默认文件名*/, OFN_HIDEREADONLY/*样式，这里设置为“隐藏只读”*/, _T("所有文件(*.*)|*.*||")/*文件类型列表*/, NULL, NULL, FALSE/*指定文件打开对话框是否为Vista样式*/);
	if (dlg.DoModal() == IDOK)//如果用户在文件对话框中点击了“确定”按钮
	{
		CString strPathName = dlg.GetPathName();//获取文件路径到strPathName
		mls_path.SetWindowText(strPathName);//显示文件路径到编辑框
	}
}


void CNewfileIn::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	CFileDialog dlg(TRUE/*这个参数为TRUE就是“打开”对话框，为FALSE就是“保存”对话框*/, NULL/*默认文件类型*/, NULL/*默认文件名*/, OFN_HIDEREADONLY/*样式，这里设置为“隐藏只读”*/, _T("所有文件(*.*)|*.*||")/*文件类型列表*/, NULL, NULL, FALSE/*指定文件打开对话框是否为Vista样式*/);
	if (dlg.DoModal() == IDOK)//如果用户在文件对话框中点击了“确定”按钮
	{
		CString strPathName = dlg.GetPathName();//获取文件路径到strPathName
		als_path.SetWindowText(strPathName);//显示文件路径到编辑框
	}
}


void CNewfileIn::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	CString mls_1;
	GetDlgItem(IDC_EDIT1)->GetWindowText(mls_1);
	//将mls_1转换为string，存放在mls文件容器中
	string m1 = (CT2A)mls_1;
	mls_file.push_back(m1);
	filepath.Append(mls_1);
	filepath += "\r\n";
	UpdateData(false);
}


void CNewfileIn::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	CString als_1;
	GetDlgItem(IDC_EDIT2)->GetWindowText(als_1);
	//将mls_1转换为string，存放在mls文件容器中
	string a1 = (CT2A)als_1;
	als_file.push_back(a1);
	filepath.Append(als_1);
	filepath += "\r\n";
	UpdateData(false);
}
