#include <iostream>
#include <vector>
#include <algorithm>
#include "dynamic.h"

using namespace std;

//leetcode石头游戏3

string Solution::stoneGameIII(vector<int>& stoneValue) {
	int n = stoneValue.size();
	vector<int>f;//f[i]表示先手从i开始拿，最多能从剩余数组中获得多少领先
	f.resize(n + 1);
	for (int i = n - 1; i >= 0; i--) {
		int sum = 0;//取的累计和
		int mt = INT_MIN;//mt为从i开始取能获得的最大领先，初始化为负无穷
		for (int j = i; j < i + 3 && j < n; j++) {//1、2、3
			sum += stoneValue[j];//自己拿1、2、3个所获得的分数
			f[i] = max(mt, sum - f[j + 1]);//取完后，轮到对手取了，所以要减去对手能获得的最大分数，就是自己的得分
		}
	}
	if (f[0] > 0) {
		return "Alice";
	}
	else if (f[0] == 0) {
		return "Tie";
	}
	else {
		return "Bob";
	}
}
/*
Alice 和 Bob 用几堆石子在做游戏。几堆石子排成一行，每堆石子都对应一个得分，由数组 stoneValue 给出。

Alice 和 Bob 轮流取石子，Alice 总是先开始。在每个玩家的回合中，该玩家可以拿走剩下石子中的的前 1、2 或 3 堆石子 。比赛一直持续到所有石头都被拿走。

每个玩家的最终得分为他所拿到的每堆石子的对应得分之和。每个玩家的初始分数都是 0 。比赛的目标是决出最高分，得分最高的选手将会赢得比赛，比赛也可能会出现平局。

假设 Alice 和 Bob 都采取 最优策略 。如果 Alice 赢了就返回 "Alice" ，Bob 赢了就返回 "Bob"，平局（分数相同）返回 "Tie" 。

*/






//dp[i][j]表示从num[i......j]开始，当前操作的选手（注意，不一定是先手）与另一位选手最多的分数差
//num[i.....j]拿左边后变为num[i+1.......j],拿右边后变为num[i.......j-1]
//int a = nums[i] - dp[i+1][j];表示拿了i后，第二个人在i+1.....j中所能获得的最大分数与num[i]的差值，
//int b = nums[j] - dp[i][j - 1];表示拿了j后，第二个人在i.....j-1中所能获得的最大分数与num[j]的差值，
//如果说a>b，则说明
bool Solution::predict(vector<int>& nums) {
	int n = nums.size();
	vector<vector<int>>dp(n, vector<int>(n + 1));
	for (int i = n; i >= 0; i--) {
		for (int j = i + 1; j < n; j++) {
			int a = nums[i] - dp[i+1][j];
			int b = nums[j] - dp[i][j - 1];
			dp[i][j] = max(a, b);
		}
	}
	return dp[0][n - 1] > 0;
}
/*
给定一个表示分数的非负整数数组。 玩家1从数组任意一端拿取一个分数，随后玩家2继续从剩余数组任意一端拿取分数，然后玩家1拿，……。
每次一个玩家只能拿取一个分数，分数被拿取之后不再可取。直到没有剩余分数可取时游戏结束。最终获得分数总和最多的玩家获胜。
给定一个表示分数的数组，预测玩家1是否会成为赢家。你可以假设每个玩家的玩法都会使他的分数最大化。
如果最终两个玩家的分数相等，那么玩家1仍为赢家。
*/



