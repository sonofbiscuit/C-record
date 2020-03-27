/*
* insert new data
* query data
* preorder
* inorder
* postorder

*/

#include <iostream>
#include "Tree.h"

using namespace std;

MyTestTree::MyTestTree() {
	pRoot = NULL;
}

MyTestTree::MyTestTree(dataType value) {
	pRoot = new MyTree(value);
	if (pRoot == NULL)
		return;
}

MyTestTree::~MyTestTree() {
	if (pRoot == NULL)
		return;
	freeMemory(pRoot);
}

void MyTestTree::freeMemory(pTreeNode pNode) {
	if (pNode == NULL)
		return;
	if (pNode->pFirstChild != NULL)
		freeMemory(pNode->pFirstChild);
	if (pNode->pNextBrother != NULL)
		freeMemory(pNode->pNextBrother);
	delete(pNode);
	pNode = NULL;
}

void MyTestTree::Insert(dataType parentvalue, dataType value) {
	if (pRoot == NULL)
		return;
	pTreeNode pfind = search(pRoot, parentvalue);
	if (pfind == NULL)
		return;
	if (pfind->pFirstChild == NULL)
		pfind->pFirstChild = new MyTree(value);
	else
	{
		InsertBrother(pfind->pFirstChild, value);
	}
}

void MyTestTree::InsertBrother(pTreeNode pBrotherNode, dataType value) {
	if (pBrotherNode->pNextBrother != NULL)
		InsertBrother(pBrotherNode->pNextBrother, value);
	else
	{
		pBrotherNode->pNextBrother = new MyTree(value);
	}
}

pTreeNode MyTestTree::search(pTreeNode pNode, dataType value) {
	if (pNode == NULL)
		return pNode;
	if (pNode->val == value)
		return pNode;
	if (pNode->pFirstChild == NULL && pNode->pNextBrother == NULL)
		return NULL;
	else {
		if (pNode->pFirstChild!=NULL)
		{
			pTreeNode ptemp=search(pNode->pFirstChild, value);
			if (ptemp != NULL)
				return ptemp;
			else
			{
				return search(pNode->pNextBrother, value);
			}
		}
		else
		{
			return search(pNode->pNextBrother, value);
		}
	}
}

void MyTestTree::preorder(pTreeNode pNode) {
	if (pNode == NULL)
		return;
	cout << pNode->val << " ";
	preorder(pNode->pFirstChild);
	preorder(pNode->pNextBrother);
}

void MyTestTree::Inorder(pTreeNode pNode) {
	if (pNode == NULL)
		return;
	Inorder(pNode->pFirstChild);
	cout << pNode->val << " ";
	Inorder(pNode->pNextBrother);
}

void MyTestTree::postorder(pTreeNode pNode) {
	if (pNode == NULL)
		return;
	postorder(pNode->pFirstChild);
	postorder(pNode->pNextBrother);
	cout << pNode->val << " ";
}


int main() {
	MyTestTree mytree(1);
	if (mytree.pRoot == NULL)
		return 0;
	mytree.Insert(1, 2);
	mytree.Insert(1, 3);
	mytree.Insert(1, 4);
	mytree.Insert(1, 5);
	mytree.Insert(1, 6);
	mytree.Insert(1, 7);
	mytree.Insert(4, 8);
	mytree.Insert(5, 9);
	mytree.Insert(5, 10);
	mytree.Insert(6, 11);
	mytree.Insert(6, 12);
	mytree.Insert(6, 13);
	mytree.Insert(10, 14);
	mytree.Insert(10, 15);

	cout << "前序遍历：" << endl;
	mytree.preorder(mytree.pRoot);
	cout << endl;
	cout << "中序遍历：" << endl;
	mytree.Inorder(mytree.pRoot);
	cout << endl;
	cout << "后序遍历：" << endl;
	mytree.postorder(mytree.pRoot);
	system("pause");
	return 0;
}