#pragma once


// CRegistError 对话框

class CRegistError : public CDialogEx
{
	DECLARE_DYNAMIC(CRegistError)

public:
	CRegistError(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CRegistError();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_RegistError };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
};
