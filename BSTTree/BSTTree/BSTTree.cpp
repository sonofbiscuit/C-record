#include <iostream>
#include <queue>
#include <algorithm>
#include <stack>
#include "BST.h"
using namespace std;


void BST::BSTCreat() {
	int value;
	cout << "输入结点值(0结束): " << endl;
	cin >> value;
	T = NULL;
	while (value) {
		bstnode p = new BSTNode();
		p->left = p->right = nullptr;
		p->val = value;
		BSTInsert(p);
		cout << "输入下一个结点的值: " << endl;
		cin >> value;
	}
	return;
}

void BST::BSTInsert(bstnode node) {	
	bstnode temp = T;
	bstnode temp1 = T;
	
	while (temp1) {//T不空
		temp = temp1;//temp存储temp1为空时候的父节点
		if (node->val < temp1->val) {
			temp1 = temp1->left;
		}
		else {
			temp1 = temp1->right;
		}
	}
	if (T == NULL) {//说明node是头
		T = node;
	}
	else {
		if (node->val < temp->val) {
			temp->left = node;
		}
		else {
			temp->right = node;
		}
	}
}

void BST::BSTDepth() {//非递归
	if (T == NULL)
		return;
	int level=0;
	int n;
	queue<bstnode> que;
	que.push(T);
	while (!que.empty()) {
		level++;
		n = que.size();
		for (int i = 0; i < n; i++) {
			bstnode temp = que.front();
			que.pop();
			if (temp->left != nullptr) {
				que.push(temp->left);
			}
			if (temp->right != nullptr) {
				que.push(temp->right);
			}
		}
	}
	cout << "树的深度为: " << level << endl;
}


int BST::BSTDepthRE(bstnode p) {
	while (p) {
		 left = BSTDepthRE(p->left);
		 right = BSTDepthRE(p->right);
	
		if (left > right) {
			return left + 1;
		}
		else
		{
			return right + 1;
		}
	}
	return 0;
}

bstnode BST::GetRoot() {
	return T;
}

void BST::BSTPre() {
	stack<bstnode> st;
	
	st.push(T);
	while (!st.empty()) {
		bstnode temp = st.top();
		st.pop();
		cout << temp->val<<" ";
		if (temp->right != NULL) {
			st.push(temp->right);
		}
		if (temp->left != NULL) {
			st.push(temp->left);
		}
	}
	cout << endl;
}

void BST::BSTIn() {
	stack<bstnode> st;
	st.push(T);
	while (!st.empty()) {
		bstnode temp = st.top();
		while (temp->left) {
			st.push(temp->left);
			temp = temp->left;
		}
		while (!st.empty()) {
			bstnode q = st.top();
			st.pop();
			cout << q->val << " ";
			if (q->right) {
				st.push(q->right);
				break;
			}
		}
	}
	cout << endl;
}
void BST::BSTPost() {
	stack<bstnode> st;
	stack<bstnode> st1;
	st.push(T);
	while (!st.empty()) {
		bstnode temp = st.top();
		st.pop();
		st1.push(temp);
		if (temp->left) {
			st.push(temp->left);
		}
		if (temp->right) {
			st.push(temp->right);
		}		
	}
	while (!st1.empty()) {
		bstnode p = st1.top();
		st1.pop();
		cout << p->val<<" ";
	}
	cout << endl;
}


bstnode BST::search(bstnode node, int val) {
	bstnode p = node;
	if (node == NULL||node->val==val)
		return node;
	else if (val < p->val) {
		search(p->left, val);
	}
	else if (val > p->val) {
		search(p->right, val);
	}
	else {
		cout << "不存在" << endl;
	}
}






int main() {
	BST bst;
	bst.BSTCreat();
	bst.BSTDepth();
	cout << "非递归求深度: ";
	cout << bst.BSTDepthRE(bst.GetRoot()) << endl;
	cout << "非递归前序: ";
	bst.BSTPre();
	cout << "非递归中序: ";
	bst.BSTIn();
	cout << "非递归后序: ";
	bst.BSTPost();

	int value;
	cout << "输入需要搜索的值: ";
	cin >> value;
	cout << bst.search(bst.GetRoot(), value);
	return 0;
}
