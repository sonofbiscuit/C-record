#pragma once
typedef struct BSTNode* bstnode;

struct BSTNode {
	int val;
	bstnode left;
	bstnode right;
	
};

class BST {
public:
	int left = 0, right = 0;
	void BSTCreat();
	void BSTInsert(bstnode node);
	void BSTDepth();
	int BSTDepthRE(bstnode node);//�ݹ�
	void BSTPre();//�ǵݹ�
	void BSTIn();//�ǵݹ�
	void BSTPost();//�ǵݹ�


	bstnode search(bstnode node, int val);
	bstnode GetRoot();

private:
	bstnode T; 
};

