#pragma once


// CNewtip 对话框

class CNewtip : public CDialogEx
{
	DECLARE_DYNAMIC(CNewtip)

public:
	CNewtip(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CNewtip();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TipDlg };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedCancel();
	afx_msg void OnBnClickedOk();
};
