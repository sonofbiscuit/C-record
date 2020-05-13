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
			vec[edges[i][1]] = edges[i][0];//���游���
		}
		tag[0] = true;
		for (int i = 0; i < (int)hasApple.size(); i++) {
			if (hasApple[i]) {//�����������й��ӵĽ��
				dfs(i);//���Ϊi�Ľ�����ؽ�������
			}
		}
		return ans * 2;
	}
	void dfs(int vroot) {
		if (!tag[vroot]) {//����ǰ����Ƿ񱻷��ʹ�
			tag[vroot] = true;
			ans++;
			dfs(vec[vroot]);//��������
		}
	}

};


int main() {
	vector<vector<int>> nums = { {0,1},{0,2},{1,4},{1,5},{2,3},{2,6} };
	vector<bool> hasApple = { false,false,true,false,true,true,false };
	Solution sol;
	sol.minTime(7, nums, hasApple);
}