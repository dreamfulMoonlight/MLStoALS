
// ptsRegistMfcDlg.h: 头文件
//

#pragma once

#include"vector"


using namespace std;
// CptsRegistMfcDlg 对话框
class CptsRegistMfcDlg : public CDialogEx
{
// 构造
public:
	CptsRegistMfcDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_PTSREGISTMFC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	vector<string> mls_1;
	vector<string> als_1;
	afx_msg void OnBnClickedButton2();
	vector<vector<float>>output;
	CString differ_value;
	afx_msg void OnBnClickedButton3();
	
	afx_msg void OnBnClickedButton4();
};
