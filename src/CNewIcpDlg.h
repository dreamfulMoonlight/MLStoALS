#pragma once


// CNewIcpDlg 对话框

class CNewIcpDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CNewIcpDlg)

public:
	CNewIcpDlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CNewIcpDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_icpDlg };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	float mGridDist;
	int wMaxiter;
	float RDistThreshold;
	int RmaxIter;
	afx_msg void OnBnClickedOk();
	afx_msg void OnEnChangeEdit1();
};
