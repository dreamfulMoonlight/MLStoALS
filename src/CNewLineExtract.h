#pragma once


// CNewLineExtract 对话框

class CNewLineExtract : public CDialogEx
{
	DECLARE_DYNAMIC(CNewLineExtract)

public:
	CNewLineExtract(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CNewLineExtract();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_LineExtract };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	float mGridDist;
	float mHeightDiff;
	float aGridDist;
	float AheightDiff;
	afx_msg void OnEnChangeEdit1();
};
