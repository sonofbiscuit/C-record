#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <typeinfo>


using namespace std;

/*======================================================================================================*/
//DP
bool PredictTheWinner(vector<int>& nums) {
	vector<vector<int>> ans(nums.size(), vector<int>(nums.size())); //store dp
	for (int i = 0; i < nums.size(); i++) {
		ans[i][i] = nums[i];
	}

	for (int i = nums.size() - 2; i >= 0; i--) {
		for (int j = i + 1; j < nums.size(); j++) {
			int l = nums[i] - ans[i + 1][j];
			int r = nums[j] - ans[i][j - 1];
			ans[i][j] = max(l, r);
		}
	}

	for (int i = 0; i < ans.size(); i++) {
		for (int j = 0; j < ans[i].size(); j++) {
			cout << ans[i][j] << "  ";
		}
		cout << endl;
	}
	return true;
}

//1690. 石子游戏 VII
int stoneGameVII(vector<int>& stones) {
	int n = (int)stones.size();
	vector<vector<int>>sum_rec(n, vector<int>(n));  // 记录区间内的和
	vector<vector<int>>dp(n, vector<int>(n)); // dp记录当前玩家和另一个玩家得分的最大差值
	// 所以 最大得分差 可以理解为此次操作之后，A 所收获的价值 - 下次B 比A的得分差的最大值。
	// 如果是 B 操作，那么就是 B 所收获的价值 - 下次A比B得分差的最大值
	for (int i = 0; i < n; i++) {
		sum_rec[i][i] = stones[i];
		for (int j = i + 1; j < n; j++) {
			sum_rec[i][j] = sum_rec[i][j - 1] + stones[j];
		}
	}

	for (int i = n - 1; i >= 0; i--) {
		for (int j = i + 1; j < n; j++) {
			if (j - i == 1) {
				dp[i][j] = max(stones[i], stones[j]);
			}
			dp[i][j] = max(sum_rec[i + 1][j] - dp[i + 1][j], sum_rec[i][j - 1] - dp[i][j - 1]);
		}
	}
	cout << dp[0][n - 1];
	return dp[0][n - 1];
}


/*===========================================================================================================*/
//dp 714
int maxProfit(vector<int>& prices, int fee) {
	int n = prices.size();
	vector<vector<int>> dp(n, vector<int>(2));
	//dp[i][0]表示第i天结束没有股票时的最大收益，dp[i][1]表示手里有股票时候的最大收益
	dp[0][0] = 0;
	dp[0][1] = -prices[0];
	for (int i = 1; i < n; i++) {
		dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
		dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
	}
	for (int j = 0; j < n; j++) {
		cout << dp[j][0] << "   " << dp[j][1] << endl;
	}
	return dp[n-1][0];
}
//贪心
int maxProfit1(vector<int>& prices, int fee) {
	int n = prices.size();
	int sum_profit = 0;
	int buy = prices[0] + fee; //默认买了第一个
	for (int i = 0; i < n; i++) {
		if (prices[i] + fee < buy) { // 更便宜，则买便宜的
			buy = prices[i] + fee; // fee加在买的时候而不放在else if的卖的操作时，理解为在跌的时候才买入
			//可以防止后续一直是上涨时候， 不断执行-fee，从而出现错误
		}
		else if (prices[i] > buy) {  //卖出去能赚，则赚
			sum_profit += prices[i] - buy;
			buy = prices[i];
		}
	}
	cout << sum_profit << endl;
	return sum_profit;
}
//买卖股票III， leetcode 123
int maxProfitIII(vector<int>& prices) {
	int n = prices.size();
	vector<int> buy1(n, 0);
	vector<int> sell1(n, 0);
	vector<int> buy2(n, 0);
	vector<int> sell2(n, 0);
	buy1[0] = -prices[0];
	buy2[0] = -prices[0];
	for (int i = 1; i < n; i++) {
		buy1[i] = max(buy1[i - 1], -prices[i]);
		sell1[i] = max(sell1[i - 1], buy1[i] + prices[i]);
		buy2[i] = max(buy2[i - 1], sell1[i] - prices[i]);
		sell2[i] = max(sell2[i - 1], buy2[i] + prices[i]);
	}
	for (auto a : sell2) {
		cout << a << endl;
	}
	return sell2[n - 1];
}





//dp746
int minCostClimbingStairs(vector<int>& cost) {
	int mincost = 0;
	int n = cost.size();
	//vector<int> dp(n + 1);
	//使用滚动数组优化，时间换空间
	vector<int> dp(3);
	dp[1] = 0, dp[2] = 0;
	for (int i = 2; i <= n; i++) {
		dp[2] = min(dp[1] + cost[i - 1], dp[0] + cost[i - 2]);
		dp[0] = dp[1];
		dp[1] = dp[2];
	}
	for (auto a : dp) {
		cout << a << endl;
	}
	return dp[2];
}


//走棋盘，左上角到右下角多少种方法，棋盘mxn大小
int uniquePaths(int m, int n) {
	vector<vector<int>> dp(m, vector<int>(n)); //dp各位置上表示的是到此位置有几种走法
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i == 0 || j == 0)
				dp[i][j] = 1;
			else
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
		}
	}
	cout << dp[m - 1][n - 1] << endl;
	return dp[m - 1][n - 1];
}


//978 dp
int maxTurbulenceSize(vector<int>& arr) {
	int n = arr.size();
	vector<vector<int>> dp(n, vector<int>(2, 1));
	dp[0][0] = dp[0][1] = 1;
	//dp[i][0]表示大于左侧的数
	//dp[i][1]表示小于左侧的数
	for (int i = 1; i < n; i++) {
		if (arr[i] > arr[i - 1]) {
			dp[i][0] += dp[i - 1][1];
		}
		else if (arr[i] < arr[i - 1]) {
			dp[i][1] += dp[i - 1][0];
		}
	}

	int max_count = 1;
	for (int i = 0; i < n; i++) {
		max_count = max(max_count, dp[i][0]);
		max_count = max(max_count, dp[i][1]);
	}
	cout << max_count << endl;
	return max_count;
}



//最长严格递增子序列
int lengthOfLIS1(vector<int>& nums) {
	int n = nums.size();
	if (n == 0) {
		return 0;
	}
	vector<int> dp(n);
	//dp代表的是当前位置最长的序列
	//状态方程  dp[i] = max{dp[i], dp[j]+1};  j为小于i的点，且nums[j]<nums[i]
	for (int i = 0; i < n; i++) {
		dp[i] = 1;
		for (int j = 0; j < i; j++) {
			if (nums[j] < nums[i]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	cout << *max_element(dp.begin(), dp.end()) << endl;
	return *max_element(dp.begin(), dp.end());
}


//Russian Doll Envelopes
//俄罗斯套娃
//使用了严格递增子序列
//先正常排序，按照[x,y]第二个元素从大到小选择，第一个元素按照严格递增子序列
int maxEnvelopes1(vector<vector<int>>& envelopes) {
	if (envelopes.empty()) {
		return 0;
	}
	int n = envelopes.size();
	vector<int> dp; //dp表示当前处最大的信封嵌套次数


	sort(envelopes.begin(), envelopes.end(), [](auto const& e1, auto const& e2) { //先由小到大
		return e1[0] < e2[0] || (e1[0] == e2[0] && e1[1] > e2[1]);
	});

	//更新dp
	dp.resize(n, 1);
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (envelopes[j][1] < envelopes[i][1]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	cout << *max_element(dp.begin(), dp.end()) << endl;
	return *max_element(dp.begin(), dp.end());
}


//背包问题

//0-1背包
int bagProblem01(vector<int> weight, vector<int> value, int num, int capcity) {
	vector<vector<int>> dp(num + 1, vector<int>(capcity + 1));
	//dp[i][j]表示将前i件物品装进限重为j的背包可以获得的最大价值
	for (int i = 0; i <= num; ++i) { //num
		for (int j = 0; j <= capcity; ++j) { //weight
			if (i == 0 || j == 0) {
				dp[i][j] = 0;
			}
			else {
				if (j >= weight[i - 1]) { //此时是可以装下物品i的
					dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i - 1]] + value[i - 1]); //选物品i与不选物品i
				}
				else {
					dp[i][j] = dp[i - 1][j];
				}
			}
		}
	}
	return dp[num][capcity];
}

//完全背包问题

//完全背包，
//如果求组合数就是外层for循环遍历物品，内层for遍历背包。   <==有次序
//如果求排列数就是外层for遍历背包，内层for循环遍历物品。   <==无次序

//每种物品有无限多个，第i（i从1开始）种物品的重量为w[i]，价值为v[i].
//在总重量不超过背包承载上限,能够装入背包的最大价值
int bagProblemcomp1(vector<int> weight, vector<int> value, int num, int capcity) {
	vector<vector<int>> dp(num + 1, vector<int>(capcity + 1)); //一共num件物品，背包容量为capcity
	//dp[i][j]表示将前i件物品装进限重为j的背包可以获得的最大价值
	for (int i = 1; i <= num; ++i) { //num
		for (int j = 0; j <= capcity; ++j) { //weight
			if (j >= weight[i - 1]) { //此时是可以装下物品i的
				dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i - 1]] + value[i - 1]); 
				//选物品i与不选物品i.物品i可以重复选取，因此选取时写dp[i][j - weight[i - 1]]而不是dp[i-1][j - weight[i - 1]]
			}
			else {
				dp[i][j] = dp[i - 1][j];
			}
		}
	}
	return dp[num][capcity];
}

int bagProblemcomp1_1(vector<int> weight, vector<int> value, int num, int capcity) {
	vector<vector<int>> dp(num + 1, vector<int>(capcity + 1)); //一共num件物品，背包容量为capcity
	//dp[i][j]表示将前i件物品装进限重为j的背包可以获得的最大价值
	for (int i = 1; i <= num; ++i) { //num
		for (int j = 0; j <= capcity; ++j) { //weight
			
		}
	}
	return dp[num][capcity];
}

//多重背包问题




//背包问题 恰好装满  、  利润至少为  、 求所有的方案  等等

/*	
int main() {
	
	vector<int> weight = { 2,2,6,5,4 };
	vector<int> value = { 6,3,5,4,6 };
	int capcity = 10;
	int num = 5;

	bagProblem01(weight, value, num, capcity);

	return 0;
}*/

	//vector<int> arr = { 9,4,2,10,7,8,8,1,9 };
	//maxTurbulenceSize(arr);

	//uniquePaths(3, 3);
	
	//vector<int> nums = { 7,90,5,1,100,10,10,2  };
	//PredictTheWinner(nums);

	//vector<int> stones = { 5,3,1,4,2 };
	//stoneGameVII(stones);

	//vector<int> price = { 3, 2, 1, 8, 4, 9 };
	//int fee = 2;
	//maxProfit1(price, fee);


	//vector<int> cost = { 10,15,20 };
	//minCostClimbingStairs(cost);




