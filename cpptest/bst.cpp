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
#include <array>
#include <deque>
#include <time.h>
#include "StlStringChange.h"
#include <numeric>
#include <functional>


/*
bd
byda
wangyi


other
*/


class byda {
public:
//===================================================================BYTE DANCE 2021.9.12
/*
int count_task(int N, vector<pair<int, int>> tasks, int max_time) {  //11111
	vector<int> cover(max_time + 2, 0);
	for (auto a : tasks) {
		cover[a.first]++;
		cover[a.second+1]--;
	}

	int max_task = 0;
	int temp_sum = 0;
	for (int i = 0; i < cover.size();++i) {
		temp_sum += cover[i];
		if (max_task < temp_sum) {
			max_task = temp_sum;
		}
	}
	return max_task;
}

int main() {
	vector<pair<int, int>> tasks;

	int N, begin_time, dur_time;
	cin >> N;
	int max_time = 0;
	for (int i = 0; i < N; ++i) {
		cin >> begin_time;
		cin >> dur_time;
		int end_time = begin_time + dur_time;
		max_time = max(max_time, end_time);
		tasks.push_back(make_pair(begin_time, end_time));
	}
	cout<<count_task(N, tasks, max_time);
	return 0;
}
*/

/*
int judge(vector<int> defense, vector<int> attack) {  //////22222222
	unordered_map<int, int> rec_def;
	for (auto a : defense) {
		rec_def[a]++;
	}

}

int main() {
	vector<pair<int, int>> tasks;

	int case_num, n, m;
	cin >> case_num >> n >> m;
	int def, att;
	for (int i = 0; i < case_num; ++i) {
		vector<int> defense(n), attack(m);
		for (int j = 0; j < n; ++j) {
			cin >> def;
			defense.push_back(def);
		}
		for (int j = 0; j < m; ++j) {
			cin >> att;
			attack.push_back(att);
		}


	}

	return 0;
}
*/

/*
int choose_num(vector<int> nums) {   //////44444

}
*/

};



class bd {

public:
	//baidu   2021.9.14     笔试3
	int ini_dist = 0x3f3f3f3f;
	int dijkstra(int village, int start, int next_village, vector<vector<int>> uni, vector<vector<int>> bid) {
		vector<vector<int>> graph(village + 1, vector<int>(village + 1, ini_dist));
		vector<int> dist(village + 1, ini_dist);
		vector<int> visited(village + 1, 0);
		vector<int> pre(village + 1, 0);  //快递站从1开始 

		for (auto& a : uni) {
			graph[a[0]][a[1]] = a[2];
			if (a[0] == start) {
				dist[a[1]] = a[2];
				pre[a[1]] = start;
			}
		}

		for (auto& a : bid) {
			graph[a[0]][a[1]] = a[2];
			graph[a[1]][a[0]] = a[2];
			if (a[0] == start || a[1] == start) {
				dist[a[1]] = a[2];
				dist[a[0]] = a[2];
				pre[a[1]] = start;
				pre[a[0]] = start;
			}
		}

		dist[start] = 0;
		for (int i = 1; i <= village; ++i) {
			int temp = -1;  //初始化点temp
			for (int j = 1; j <= village; ++j) {
				if (!visited[j] && (temp == -1 || dist[j] < dist[temp])) {
					temp = j;
				}
			}
			visited[temp] = 1;

			//update
			for (int k = 1; k <= village; ++k) {
				dist[k] = min(dist[k], dist[temp] + graph[temp][k]);
			}
		}

		// final
		return dist[next_village];
	}

	int express(int village_num, vector<vector<int>> uni, vector<vector<int>> bid, vector<int> param, vector<int> target, int start) {
		//village_num
		//uni    unidirectional road
		//bid    bidirectional  road
		//param  wasted time  a(face2face)  and  b(cupboard) ,  send_num   
		//target  target village
		int ans = 0;
		int t = 0;
		int now = start;
		for (auto& v : target) {
			int need_time = dijkstra(village_num, now, v, uni, bid);
			t += need_time;
			if (t % 2) { //odd
				t += param[0];
			}
			else {
				t += param[1];
			}
			now = v;
		}
		int back_time = dijkstra(village_num, now, start, uni, bid);
		return t + back_time;
	}
};




class other {
public:
	/*
	vector<string> words{ "a","aba","abc","d","cd","bcd","abcd" };
	maxSum(words);
	*/
	int maxSum(vector<string>& words) {
		int n = words.size();
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			unordered_map<char, int> mp;
			int a_size = 0;
			for (auto& a : words[i]) {
				if (!mp.count(a)) {
					mp[a]++;
					a_size++;
				}
			}

			for (int j = i + 1; j < n; ++j) {
				int b_size = 0;
				for (auto& b : words[j]) {
					if (mp.count(b)) {
						b_size = 0;
						break;
					}
					else {
						b_size++;
					}
				}
				unordered_set<char> st = unordered_set(words[j].begin(), words[j].end());
				if (b_size == 0) {
					continue;
				}
				ans = max(ans, a_size + (int)st.size());
			}
		}
		return ans;
	}

	//==============================================================================================
	

	//===================================================网易wangyi===========================================
	int strop(string str, int m) {
		//string str;
		//int m;
		//getline(cin, str);
		//cin >> m;
		int len = str.size();
		vector<int> vec(len, 0);
		for (int i = 0; i < len - 1; ++i) {
			vec[i + 1] = abs(str[i + 1] - str[i]);
		}
		int l = 0, r = m - 1;
		int total = 0;
		for (int j = 0; j < m; ++j) {
			total += (str[j] - 'A');
		}
		int index = l;
		l++;
		r++;
		while (r < len) {
			int temp = total - (str[l] - 'A') + (str[r] - 'A');
			if (temp >= total) {
				total = temp;
				index = l;
			}
			l++;
			r++;
		}
		int tt = accumulate(vec.begin(), vec.end(), 0);
		int ans = len + (tt - total) + m;
		return ans;
	}
};


/*
int main() {
	return 0;
}
*/

/*
* baidu 2021.9.14  笔试3
*
int village_num = 6;
vector<vector<int>> uni{ {1,2,1},{4,1,2},{3,5,2},{2,3,1} };
vector<vector<int>> bid{ {6,3,1},{4,5,1},{1,3,3},{2,4,2} };
vector<int> param{ 2,3,5 };
vector<int> target{ 1,4,6,6,2 };
int start = 2;
bd b;
b.express(village_num, uni, bid, param, target, start);
*/