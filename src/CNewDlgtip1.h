#pragma once


// CNewDlgtip1 对话框

class CNewDlgtip1 : public CDialogEx
{
	DECLARE_DYNAMIC(CNewDlgtip1)

public:
	CNewDlgtip1(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CNewDlgtip1();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_Dlgtip1 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedCancel();
};
