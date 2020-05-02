#include <iostream>
#include <vector>
#include <algorithm>
#include "backTracking.h"
using namespace std;

//树的图片见文档，子集/组合/排列.png
//子串
//输入一个不包含重复数字的数组，输出这些数字的所有子集
vector<vector<int>> Solution::subsets(vector<int>& nums) {
	vector<int> temp;
	backtrack(nums, 0, temp);
	for (size_t i = 0; i < ans.size(); i++) {
		for (size_t j = 0; j < ans[i].size(); j++) {
			cout << ans[i][j] << " ";
		}
		cout << endl;
	}
	return ans;
}

void Solution::backtrack(vector<int>& nums, int start, vector<int>& temp) {
	ans.push_back(temp);
	for (size_t i = start; i < nums.size(); i++) {
		temp.push_back(nums[i]);
		backtrack(nums, i + 1, temp);
		temp.pop_back();
	}
}

//组合
//输入两个数字 n, k，算法输出[1..n] 中 k 个数字的所有组合。
//k限制了树的高度，n限制了树的宽度
vector<vector<int>> Solution::combine(int n, int k) {
	vector<int> track;
	backtrack1(n, k, 1, track);
	cout << "================" << endl;
	for (size_t i = 0; i < ans1.size(); i++) {
		for (size_t j = 0; j < ans1[i].size(); j++) {
			cout << ans1[i][j] << " ";
		}
		cout << endl;
	}
	return ans1;
}
void Solution::backtrack1(int n, int k, int start, vector<int>& track) {
	if (track.size() == k) {
		ans1.push_back(track);
		return;
	}
	for (int i = start; i <= n; i++) {//类似于一个指针，遍历树上的所有结点
		track.push_back(i);
		backtrack1(n, k, i + 1, track);
		track.pop_back();
	}
}


//排列
//输入一个不包含重复数字的数组 nums，返回这些数字的全部排列。
vector<vector<int>> Solution::permute(vector<int>& nums) {
	vector<int> track;
	backtrack2(nums, track);
	cout << "================" << endl;
	for (size_t i = 0; i < ans2.size(); i++) {
		for (size_t j = 0; j < ans2[i].size(); j++) {
			cout << ans2[i][j] << " ";
		}
		cout << endl;
	}
	return ans2;
}
void Solution::backtrack2(vector<int>& nums, vector<int>& track) {
	if (track.size() == nums.size()) {
		ans2.push_back(track);
		return;
	}
	for (size_t i = 0; i < nums.size(); i++) {
		if (find(track.begin(),track.end(),nums[i]) != track.end()) {//包含nums[i]
			continue;
		}
		track.push_back(nums[i]);
		backtrack2(nums, track);
		track.pop_back();
	}
}


int main() {
	vector<int> nums = { 1,2,3 };
	Solution sol;
	sol.subsets(nums);
	sol.combine(5, 2);
	vector<int> nums1 = { 1,2,3 };
	sol.permute(nums1);
}