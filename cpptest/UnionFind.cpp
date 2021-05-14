#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <typeinfo>
#include <stack>
#include <queue>
#include <deque>
#include "StlStringChange.h"

using namespace std;
using namespace strtool;



class UnionFindOp {
private:
	vector<int> roots;
	vector<int> size;//根节点下的全部的结点数
	int n;

public:
	UnionFindOp(int n) {
		roots.resize(n);
		size.resize(n, 1);
		for (int i = 0; i < n; i++) {
			roots[i] = i;
		}
	};

	int find(int i) { // 使用路径压缩
		return i == roots[i] ? i : (roots[i] = find(roots[i]));
	}

	void union_element(int i, int j) {//按秩合并
		int x = find(i);
		int y = find(j);
		if (x != y) {
			if (size[x] > size[y]) {
				roots[y] = x;
				size[x] += size[y];
			}
			else {
				roots[x] = y;
				size[y] += size[x];
			}
		}
	}

	bool isconnected(int a, int b) {
		int x = find(a);
		int y = find(b);
		return x == y;
	}
};


class Solution {
public:
	int minimumEffortPath(vector<vector<int>>& heights) {
		if (heights.empty()) {
			return -1;
		}
		int m = heights.size();
		int n = heights[0].size();
		vector<tuple<int, int, int >> edges;
		for (int i = 0; i < m; i++) { //hang
			for (int j = 0; j < n; j++) { //lie
				int index = i * n + j;
				if (i > 0) {//hang
					edges.emplace_back(index - n, index, abs(heights[i][j] - heights[i - 1][j]));
				}
				
				if (j > 0) {//lie
					edges.emplace_back(index - 1, index, abs(heights[i][j] - heights[i][j-1]));
				}//i=0,j=1,2,3,4,...        0-1, 1-2, 2-3,...
			}
		}

		sort(edges.begin(), edges.end(), [](const auto& e1, const auto& e2) {
			auto&& [x1, x2, x3] = e1;
			auto&& [y1, y2, y3] = e2;
			return x3 < y3;
		});

		UnionFindOp uf(m * n);
		int ans = 0;
		for (auto [x, y, z] : edges) {
			uf.union_element(x, y);
			if (uf.isconnected(0, m * n - 1)) {
				ans = z;
				break;
			}
		}
		cout << ans << endl;
		return ans;

	}
};

/*
int main() {
	//n x n 的矩阵 isConnected,isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
	vector<vector<int>> heights = { {1, 2, 2},{3,8,2},{5,3,5} };
	Solution sol;
	sol.minimumEffortPath(heights);

	return 0;
}*/