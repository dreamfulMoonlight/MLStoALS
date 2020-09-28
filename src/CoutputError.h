#pragma once


// CoutputError 对话框

class CoutputError : public CDialogEx
{
	DECLARE_DYNAMIC(CoutputError)

public:
	CoutputError(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CoutputError();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_outputError };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
};
