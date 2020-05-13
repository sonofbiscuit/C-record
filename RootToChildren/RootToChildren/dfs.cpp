#include <iostream>
#include <vector>

using namespace std;


class Solution {
public:
	int ans = 0;
	vector<int> vec;
	vector<bool> tag;
	int minTime(int n, vector<vector<int>>& edges, vector<bool>& hasApple) {
		vec.resize(n);
		tag.resize(n, false);
		for (int i = 0; i < (int)edges.size(); i++) {
			cout << edges[i][1] << "    " << edges[i][0] << endl;
			vec[edges[i][1]] = edges[i][0];//保存父结点
		}
		tag[0] = true;
		for (int i = 0; i < (int)hasApple.size(); i++) {
			if (hasApple[i]) {//从上往下找有果子的结点
				dfs(i);//编号为i的结点往回进行搜索
			}
		}
		return ans * 2;
	}
	void dfs(int vroot) {
		if (!tag[vroot]) {//看当前结点是否被访问过
			tag[vroot] = true;
			ans++;
			dfs(vec[vroot]);//从下往上
		}
	}

};


int main() {
	vector<vector<int>> nums = { {0,1},{0,2},{1,4},{1,5},{2,3},{2,6} };
	vector<bool> hasApple = { false,false,true,false,true,true,false };
	Solution sol;
	sol.minTime(7, nums, hasApple);
}