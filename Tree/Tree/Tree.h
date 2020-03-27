#pragma once

typedef int dataType;
typedef struct MyTree* pTreeNode;

struct MyTree {
	dataType val;
	pTreeNode pFirstChild;
	pTreeNode pNextBrother;
	MyTree(dataType x) :val(x), pFirstChild(nullptr), pNextBrother(nullptr) {}
};

class MyTestTree {
public:
	MyTestTree();
	MyTestTree(dataType value);
	~MyTestTree();

	void Insert(dataType parentvalue, dataType value);
	void InsertBrother(pTreeNode pBrotherNode, dataType value);
	pTreeNode search(pTreeNode pNode, dataType value);

	void freeMemory(pTreeNode pNode);

	void preorder(pTreeNode pNode);
	void Inorder(pTreeNode pNode);
	void postorder(pTreeNode pNode);

	pTreeNode pRoot;
};