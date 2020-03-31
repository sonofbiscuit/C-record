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
	int BSTDepthRE(bstnode node);//µÝ¹é
	void BSTPre();//·ÇµÝ¹é
	void BSTIn();//·ÇµÝ¹é
	void BSTPost();//·ÇµÝ¹é


	bstnode search(bstnode node, int val);
	bstnode GetRoot();

private:
	bstnode T; 
};

