#include <iostream>
#include <vector>
#include <queue>
#include "MatrixSearch.h"

using namespace std;

/*
������һ��m��n�еķ��񣬴����� [0,0] ������ [m-1,n-1] ��һ�������˴����� [0, 0] �ĸ��ӿ�ʼ�ƶ�
��ÿ�ο��������ҡ��ϡ����ƶ�һ�񣨲����ƶ��������⣩��Ҳ���ܽ�������������������λ֮�ʹ���k�ĸ��ӡ�
���磬��kΪ18ʱ���������ܹ����뷽�� [35, 37] ����Ϊ3+5+3+7=18���������ܽ��뷽�� [35, 38]����Ϊ3+5+3+8=19��
���ʸû������ܹ�������ٸ����ӣ�
*/
//bfsʵ�־�����������
int main() {
	Solution sol;
	sol.movingCount(3,2,17);
}

int Solution::getDigit(int x) {//��ȡx��λ����
	int sum=0;
	for (; x; x /= 10) {
		sum += x % 10;
	}
	return sum;
}

int Solution::movingCount(int m, int n, int k) {
	int res=0;
	if (k == 0)
		return 1;
	int pair1[] = { 0,1 };//����
	int pair2[] = { 1,0 };//����
	queue<pair<int, int>> que;
	vector<vector<int>> tag(m, vector<int>(n, 0));
	que.push(make_pair(0, 0));//(0��0����)
	res = 1;
	while (!que.empty()) {
		int x = que.front().first;
		int y = que.front().second;
		//auto [x, y] = que.front();  C++17�ṹ����
		que.pop();
		for (int i = 0; i < 2; i++) {
			int fi = x + pair1[i];//fi=x+0        fi=x+1
			int se = y + pair2[i];//se=y+1//����   se=y+0//����
			int tt = getDigit(fi);
			if (fi < 0 || fi >= m || se<0 || se >= n || tag[fi][se] == 1 || getDigit(fi) + getDigit(se)>k) {
				continue;//Խ������ѷ��ʹ�����λ���ʹ���k
			}
			que.push(make_pair(fi, se));//����Ҫ��ļ���que
			tag[fi][se] = 1;//���ʱ��
			res++;//��������
		}
	}
	cout << res;
	return res;
}