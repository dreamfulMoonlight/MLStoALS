#pragma once
#include"vector"
using namespace std;
// CNewfileIn 对话框

class CNewfileIn : public CDialogEx
{
	DECLARE_DYNAMIC(CNewfileIn)

public:
	CNewfileIn(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CNewfileIn();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_fileInDlg };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	CEdit mls_path;
	CEdit als_path;
	CString filepath;
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton2();
	vector<string> mls_file;
	vector<string> als_file;
	afx_msg void OnBnClickedButton4();
};
