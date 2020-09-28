#pragma once


// CfileError 对话框

class CfileError : public CDialogEx
{
	DECLARE_DYNAMIC(CfileError)

public:
	CfileError(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CfileError();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_fileError };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
};
