#pragma once
#include <vector>
using namespace std;
class Matrix {
public:
	vector<vector<int>> BFS(vector<vector<int>>& matrix);
	vector<vector<int>> DP(vector<vector<int>>& matrix);
};