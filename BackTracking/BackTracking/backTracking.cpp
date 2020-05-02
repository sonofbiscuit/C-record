#include <iostream>
#include <vector>
#include <algorithm>
#include "backTracking.h"
using namespace std;

//����ͼƬ���ĵ����Ӽ�/���/����.png
//�Ӵ�
//����һ���������ظ����ֵ����飬�����Щ���ֵ������Ӽ�
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

//���
//������������ n, k���㷨���[1..n] �� k �����ֵ�������ϡ�
//k���������ĸ߶ȣ�n���������Ŀ��
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
	for (int i = start; i <= n; i++) {//������һ��ָ�룬�������ϵ����н��
		track.push_back(i);
		backtrack1(n, k, i + 1, track);
		track.pop_back();
	}
}


//����
//����һ���������ظ����ֵ����� nums��������Щ���ֵ�ȫ�����С�
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
		if (find(track.begin(),track.end(),nums[i]) != track.end()) {//����nums[i]
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