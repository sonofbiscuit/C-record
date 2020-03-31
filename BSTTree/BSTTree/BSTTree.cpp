#include <iostream>
#include <queue>
#include <algorithm>
#include <stack>
#include "BST.h"
using namespace std;


void BST::BSTCreat() {
	int value;
	cout << "������ֵ(0����): " << endl;
	cin >> value;
	T = NULL;
	while (value) {
		bstnode p = new BSTNode();
		p->left = p->right = nullptr;
		p->val = value;
		BSTInsert(p);
		cout << "������һ������ֵ: " << endl;
		cin >> value;
	}
	return;
}

void BST::BSTInsert(bstnode node) {	
	bstnode temp = T;
	bstnode temp1 = T;
	
	while (temp1) {//T����
		temp = temp1;//temp�洢temp1Ϊ��ʱ��ĸ��ڵ�
		if (node->val < temp1->val) {
			temp1 = temp1->left;
		}
		else {
			temp1 = temp1->right;
		}
	}
	if (T == NULL) {//˵��node��ͷ
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

void BST::BSTDepth() {//�ǵݹ�
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
	cout << "�������Ϊ: " << level << endl;
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
		cout << "������" << endl;
	}
}






int main() {
	BST bst;
	bst.BSTCreat();
	bst.BSTDepth();
	cout << "�ǵݹ������: ";
	cout << bst.BSTDepthRE(bst.GetRoot()) << endl;
	cout << "�ǵݹ�ǰ��: ";
	bst.BSTPre();
	cout << "�ǵݹ�����: ";
	bst.BSTIn();
	cout << "�ǵݹ����: ";
	bst.BSTPost();

	int value;
	cout << "������Ҫ������ֵ: ";
	cin >> value;
	cout << bst.search(bst.GetRoot(), value);
	return 0;
}
