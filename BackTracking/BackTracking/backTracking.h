#pragma once
#include<vector>
using namespace std;

class Solution {
public:
	vector<vector<int>> subsets(vector<int>& nums);
	void backtrack(vector<int>& nums, int start, vector<int>& temp);

	vector<vector<int>> combine(int n, int k);
	void backtrack1(int n, int k, int start, vector<int>& track);

	vector<vector<int>> permute(vector<int>& nums);
	void backtrack2(vector<int>& nums, vector<int>&track);
private:
	vector<vector<int>> ans;
	vector<vector<int>> ans1;
	vector<vector<int>> ans2;
};