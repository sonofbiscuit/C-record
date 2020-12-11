#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <typeinfo>

using namespace std;

//leetcode 842
//深度遍历加剪枝(不断遍历加回溯)    1 23456579  12 3456579 .。。。。
bool backtrack(vector<int>& vec, string s, int index, int length, long long sum, int prev) {
	if (index == length) { //Ò»ÌõÂ·½áÊø
		return vec.size() >= 3;
	}
	long long temp = 0;
	for (int i = index; i < length; i++) {
		if (i > index&&s[index] == '0') {
			break;
		}
		temp = temp * 10 + s[i] - '0'; //123456579
		if (temp > INT_MAX) {
			break;
		}
		if (vec.size() >= 2) {
			if (temp > sum) {
				break;
			}
			else if (temp < sum) {
				continue;
			}
		}
		vec.push_back(temp);
		if (backtrack(vec, s, i + 1, length, temp + prev, temp)) { // ºóÒÆÒ»
			return true;
		}
		vec.pop_back();
	}
	return false;
}

vector<int> splitIntoFibonacci(string S) {
	vector<int> ans;
	backtrack(ans, S, 0, S.length(), 0, 0);
	for (auto a : ans) {
		cout << a << endl;
	}
	return ans;
}


/**/
int main() {
	//string S = "aaaabbb";
	//reorganizeString(S);

	//vector<int> nums = { 2,2 };
	//int target = 2;
	//searchRange(nums, target);

	//countPrimes(1500000);

	//vector<char> tasks = {'A', 'A', 'A', 'B', 'B', 'B','C','C' ,'C' ,'D' ,'D' ,'E' };
	//int n = 2;
	//leastInterval(tasks, n);

	//generate(5);

	//vector<vector<int>> s = { {0, 0, 1, 1}, {1, 0, 1, 0}, {1, 1, 0, 0} };
	//matrixScore(s);

	vector<int> vec;
	string s = "123456579";
	splitIntoFibonacci(s);




	return 0;
}




