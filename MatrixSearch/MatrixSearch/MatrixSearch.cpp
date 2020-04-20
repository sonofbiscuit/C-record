#include <iostream>
#include <vector>
#include <queue>
#include "MatrixSearch.h"

using namespace std;

/*
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动
它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。
请问该机器人能够到达多少个格子？
*/
//bfs实现矩阵搜索问题
int main() {
	Solution sol;
	sol.movingCount(3,2,17);
}

int Solution::getDigit(int x) {//获取x的位数和
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
	int pair1[] = { 0,1 };//右移
	int pair2[] = { 1,0 };//下移
	queue<pair<int, int>> que;
	vector<vector<int>> tag(m, vector<int>(n, 0));
	que.push(make_pair(0, 0));//(0，0出发)
	res = 1;
	while (!que.empty()) {
		int x = que.front().first;
		int y = que.front().second;
		//auto [x, y] = que.front();  C++17结构化绑定
		que.pop();
		for (int i = 0; i < 2; i++) {
			int fi = x + pair1[i];//fi=x+0        fi=x+1
			int se = y + pair2[i];//se=y+1//右移   se=y+0//下移
			int tt = getDigit(fi);
			if (fi < 0 || fi >= m || se<0 || se >= n || tag[fi][se] == 1 || getDigit(fi) + getDigit(se)>k) {
				continue;//越界或者已访问过或者位数和大于k
			}
			que.push(make_pair(fi, se));//符合要求的加入que
			tag[fi][se] = 1;//访问标记
			res++;//访问总数
		}
	}
	cout << res;
	return res;
}