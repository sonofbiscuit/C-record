#include <iostream>
#include <vector>
#include <utility>
#include <cstddef>
#include "GameProblem.h"

using namespace std;

int main() {
	vector<int> vec{ 3,9,1,2 };
	Solution s;
	s.stoneGame(vec);
}

//对于pair类，由于它只有两个元素，分别名为first和second，因此直接使用普通的点操作符即可访问其成员
//first和second表示先拿和后拿所能得到的比分
bool Solution::stoneGame(vector<int>& piles) {
	int left = 0, right = 0;
	size_t n = piles.size();
	pair<int, int>pair(0, 0);
	std::vector<std::pair<int, int>> temp(n, pair);
	std::vector<vector<std::pair<int, int>>> dp(n, temp);
	for (int i = 0; i < n; i++) {
		dp[i][i].first = piles[i];
		dp[i][i].second = 0;
	}

	int m = 0;
	//斜遍历数组
	for (int i = 2; i <= n; i++) {
		for (int j = 0; j <= n - i; j++) {
			int m = i + j - 1;
			left = piles[j] + dp[j + 1][m].second;//先拿左
			right = piles[m] + dp[j][m - 1].second;//先拿右

			if (left > right) {
				dp[j][m].first = left;
				dp[j][m].second = dp[j + 1][m].first;
			}
			else {
				dp[j][m].first = right;
				dp[j][m].second = dp[j][m - 1].first;
			}
		}
	}
	return dp[0][n - 1].first - dp[0][n - 1].second;
}