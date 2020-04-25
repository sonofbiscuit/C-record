#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
#include "MatrixSearch2.h"
using namespace std;

//leetcode542
int main() {
	Matrix ma;
	vector<vector<int>> tt{ {0,0,0},{0,1,0},{1,1,1} };
	ma.BFS(tt);
	cout << endl << endl;
	vector<vector<int>> tt1{ {0,0,0},{0,1,0},{1,1,1} };
	ma.DP(tt1);
}



vector<vector<int>> Matrix::BFS(vector<vector<int>>& matrix) {
	int m = matrix.size();
	int n = matrix[0].size();
	vector<pair<int, int>> mov(4);
	mov.push_back(pair<int, int>(0, -1));
	mov.push_back(pair<int, int>(-1, 0));
	mov.push_back(pair<int, int>(1, 0));
	mov.push_back(pair<int, int>(0, 1));
	queue<pair<int, int>> que;
	vector<vector<int>> tag(matrix.size(),vector<int>(matrix[0].size(),0));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (matrix[i][j] == 0) {
				que.push(make_pair(i, j));
			}
		}
	}//找0
	while (!que.empty()) {
		auto x = que.front().first;
		auto y = que.front().second;
		que.pop();
		for (int i = 0; i < mov.size(); i++) {
			int fi = x + mov[i].first;
			int se = y + mov[i].second;
			if (fi >= 0 && fi < m&&se >= 0 && se < n && !tag[fi][se] && matrix[fi][se] == 1) {
				matrix[fi][se] += matrix[x][y];
				tag[fi][se] = 1;
				que.push(make_pair(fi, se));
			}
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << matrix[i][j]<<" ";
		}
		cout << endl;
	}
	return matrix;
}



/*
0 _ _ _ 0
_ _ _ _ _
_ _ 1 _ _
_ _ _ _ _
0 _ _ _ 0
从中心位置的 11 移动到这四个 00，就需要使用四种不同的方法：
用 f(i, j)f(i,j) 表示位置 (i, j)(i,j) 到最近的 0 的距离。
如果只能「水平向左移动」和「竖直向上移动」
那么可以向上移动一步，再移动 f(i - 1, j)f(i−1,j) 步到达某一个 00，也可以向左移动一步，再移动 f(i, j - 1)f(i,j−1) 步到达某一个 0


首先从左上角开始递推 dp[i][j]dp[i][j] 是由其 「左方」和 「左上方」的最优子状态决定的；
然后从右下角开始递推 dp[i][j]dp[i][j] 是由其 「右方」和 「右下方」的最优子状态决定的；
看起来第一次递推的时候，把「右上方」的最优子状态给漏掉了，其实不是的，因为第二次递推的时候「右方」的状态在第一次递推时已经包含了「右上方」的最优子状态了；
看起来第二次递推的时候，把「左下方」的最优子状态给漏掉了，其实不是的，因为第二次递推的时候「右下方」的状态在第一次递推时已经包含了「左下方」的最优子状态了


*/

vector<vector<int>> Matrix::DP(vector<vector<int>>& matrix) {
	int m = matrix.size();
	int n = matrix[0].size();
	vector<vector<int>> dp(m,vector<int>(n,INT_MAX));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (matrix[i][j] == 0) {
				dp[i][j] = 0;
			}
		}
	}
	
	//左上
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i - 1 >= 0) {//上
				dp[i][j] = min(dp[i][j], dp[i - 1][j]+1);
			}
			if (j - 1 >= 0) {//左
				dp[i][j] = min(dp[i][j], dp[i][j - 1]+1);
			}
		}
	}

	//左下
	for (int i = m - 1; i >= 0; i--) {
		for (int j = 0; j < n; j++) {
			if (i < m - 1) {//下
				dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1);
			}
			if (j >= 1) {//左
				dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1);
			}
		}
	}

	//右上
	for (int i = 0; i < m; i++) {
		for (int j = n - 1; j >= 0; j--) {
			if (i >= 1) {//上
				dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1);
			}
			if (j < n - 1) {//右
				dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1);
			}
		}
	}
	//右下
	for (int i = m - 1; i >= 0; i--) {
		for (int j = n - 1; j >= 0; j--) {
			if (i < m - 1) {//下
				dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1);
			}
			if (j < n - 1) {//右
				dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1);
			}
		}
	}

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << dp[i][j] << " ";
		}
		cout << endl;
	}
	return dp;
}