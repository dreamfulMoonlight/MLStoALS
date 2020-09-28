#pragma once


// CNewDlgtip2 对话框

class CNewDlgtip2 : public CDialogEx
{
	DECLARE_DYNAMIC(CNewDlgtip2)

public:
	CNewDlgtip2(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CNewDlgtip2();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_Dlgtip2 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedCancel();
};
