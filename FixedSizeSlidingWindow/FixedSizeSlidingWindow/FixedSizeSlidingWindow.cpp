#include <iostream>
#include <stack>
#include <string>
#include <queue>
#include <algorithm>

using namespace std;


/*
每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 k 张卡牌。
你的点数就是你拿到手中的所有卡牌的点数之和
输入：cardPoints = [1,2,3,4,5,6,1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1 。
但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，最终点数为 1 + 6 + 5 = 12 。

*/

//for循环的嵌套会导致时间复杂度的增长
/*
int maxScore(vector<int>& cardPoints, int k) {
	int n = cardPoints.size();
	int windowsize = n - k;
	int ans = INT_MAX;
	int sum = 0;
	for (int left = 0, right = left + windowsize - 1; left <= n - windowsize && right < n; left++, right++) {
		int temp = 0;
		int l = left, r = right;
		for (l; l <= r; l++) {
			temp += cardPoints[l];
			cout << temp << endl;
		}
		ans = min(ans, temp);
		temp = 0;
	}
	for (auto c : cardPoints) {
		sum += c;
	}
	cout << endl;
	cout << sum - ans;
	return sum - ans;
}
*/

int maxScore(vector<int>& cardPoints, int k) {
	int n = cardPoints.size();
	int m = 0;
	for (auto a : cardPoints) {
		m += a;
	}
	int windowsize = n - k;
	int ans = INT_MAX;
	int sum = 0;
	int temp=0;
	for (int i = 0; i < windowsize; i++) {
		sum += cardPoints[i];//求第一个窗口的和
	}
	temp = sum;
	for (int i = 1; i < n - windowsize + 1; i++) {
		sum = sum + cardPoints[windowsize - 1 + i] - cardPoints[i-1];//窗口依次向后移动，每次加后面一个，减去前面一个
		cout << cardPoints[windowsize - 1 + i] << "   " << cardPoints[i - 1] << endl;
		temp = min(temp, sum);//存储该过程中的最小值
	}
	return m - temp;
}



int main() {
	vector<int> card = { 1,79,80,1,1,1,200,1 };
	maxScore(card, 3);
}
