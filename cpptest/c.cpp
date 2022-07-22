#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
#include <typeinfo>
#include <stack>
#include <queue>
#include <array>
#include <deque>
#include <regex>
#include <random>
#include <ctime>
#include "StlStringChange.h"
#include <numeric>
#include <functional>
#include <cmath>



/* #pragma GCC optimize(2) //O2优化
*/
using namespace std;
using namespace strtool;

using PII = pair<int, int>;


//[captures] (params) mutable-> return-type{...} //lambda 表达式的完整形式


/*
string s1 = "hiya";    // 拷贝初始化
string s2("hello");    // 直接初始化
string s3(10, 'c');    // 直接初始化
*/
//多用直接初始化，少用拷贝初始化

//循环时，前自增大于后自增，前自增返回的是自增后的自己，后自增返回的是自增前自己的副本(临时变量)


template <typename Dtype>
unsigned int __builtin_popcount(Dtype u)
{
	u = (u & 0x55555555) + ((u >> 1) & 0x55555555);
	u = (u & 0x33333333) + ((u >> 2) & 0x33333333);
	u = (u & 0x0F0F0F0F) + ((u >> 4) & 0x0F0F0F0F);
	u = (u & 0x00FF00FF) + ((u >> 8) & 0x00FF00FF);
	u = (u & 0x0000FFFF) + ((u >> 16) & 0x0000FFFF);
	return u;
}


//=======================================================并查集=====================================
class UnionFindOp {
public:
	vector<int> roots;
	vector<int> size;//根节点下的全部的结点数
	//int size;

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
};


//=========================================字典树/前缀树=============================================
class TrieOp {
private:
	bool isEnd;
	TrieOp* next[26];
public:
	/** Initialize your data structure here. */
	TrieOp() {
		isEnd = false;
		memset(next, 0, sizeof(next));
	}

	/** Inserts a word into the trie. */
	void insert(string word) {
		TrieOp* node = this;
		for (auto a : word) {
			if (node->next[a - 'a'] == NULL) {
				node->next[a - 'a'] = new TrieOp();
			}
			node = node->next[a - 'a'];
		}
		node->isEnd = true;
	}

	/** Returns if the word is in the trie. */
	bool search(string word) {
		TrieOp* node = this;
		for (auto a : word) {
			if (node->next[a - 'a'] == NULL) {
				return false;
			}
			node = node->next[a - 'a'];
		}
		return node->isEnd;
	}

	/** Returns if there is any word in the trie that starts with the given prefix. */
	bool startsWith(string prefix) {
		TrieOp* node = this;
		for (auto a : prefix) {
			if (node->next[a - 'a'] == NULL) {
				return false;
			}
			node = node->next[a - 'a'];
		}
		return true;
	}
};


string reorganizeString(string S) {
	int word_maxcount = 0;
	char word_index = ' ';
	int length = int(S.size());
	vector<int> m2(26, 0);
	for (int i = 0; i < length; i++) {
		m2[S[i]-'a']++;
		word_maxcount = max(word_maxcount, m2[S[i] - 'a']);
		word_index = word_maxcount == m2[S[i] - 'a']?S[i]:word_index;
	}
	if (word_maxcount > (S.size() + 1) / 2)
		return "";
	
	vector<char> ans(length,0);
	int index = 0;
	for (int i = word_maxcount; i > 0; i--) { //排序最多的字母
		ans[index] = word_index;
		m2[word_index - 'a']--;  //更新m2中存储的信息
		index += 2;
	}
	for (int j = 0; j < 26; j++) {
		while (m2[j]-- > 0) {
			if (index >= length)
				index = 1;
			ans[index] = char('a' + j);
			index += 2;
		}
	}
	for (int i = 0; i < ans.size(); i++) {
		cout << ans[i] << endl;
	}

	string ansbuf;
	return ansbuf.assign(ans.begin(), ans.end());
}


vector<int> searchRange(vector<int>& nums, int target) {
	vector<int> ans;
	int n = nums.size();
	int low = 0;
	int high = n - 1;
	int mid = (low+high) / 2;
	int count = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (nums[mid] < target) {
			low = mid+1;
		}
		else if (nums[mid] > target) {
			high = mid - 1;
		}
		else { //找到target,得判断数目,此时nums[mid]=target
			count++;
			int opleft = 1;
			int opright = 1;
			while (mid - opleft >= 0 && nums[mid - opleft] == target) {
				count++;
				opleft++;
			}
			ans.push_back(mid - opleft + 1);
			while (mid + opright < n && nums[mid + opright] == target) {
				count++;
				opright++;
			}
			ans.push_back(mid + opright - 1);
			for (auto c : ans) {
				cout << c << endl;
			}
			return ans;
		}
	}
	ans.push_back(-1);
	ans.push_back(-1);
	return ans;
}


int countPrimes(int n) {
	int count = 0;
	vector<int> ans(n, 0);
	for (int i = 2; i < n; i++) {
		ans[i] = 1;
	}
	for (int j = 2; j < n; j++) {
		if (ans[j] == 1) { //是素数
			count++;
			for (int m = j * 2; m < n; m+=j) {
				ans[m] = 0;
			}
		}
	}
	cout << count << endl;
	return count;
}


int leastInterval(vector<char>& tasks, int n) {
	if (tasks.empty()) {
		return 0;
	}
	vector<int> task_tag(26);
	for (auto temp : tasks) {
		task_tag[temp - 'A']++;//优点调度记录
	}

	sort(task_tag.begin(), task_tag.end());//优先度排序
	
	int total_size = (n + 1) * (task_tag[25] - 1);//忽略最后一行
	for (int i = 0; i < 26; i++) {
		total_size -= min(task_tag[i], task_tag[25] - 1);
	}
	//if()
	//cout << tasks.size() + total_size << endl;
	return total_size > 0 ? tasks.size() + total_size : tasks.size();
}

vector<vector<int>> generate(int numRows) {
	vector<vector<int>> ans(numRows);
	for (int i = 0; i < numRows; i++) {
		ans[i].resize(i + 1);
		ans[i][0] = ans[i][i] = 1;
		for (int j = 1; j < i; j++) {
			ans[i][j] = ans[i - 1][j - 1] + ans[i - 1][j];
		}
	}
	for (int i = 0; i < ans.size(); i++) {
		for (int j = 0; j < ans[i].size(); j++) {
			cout << ans[i][j];
		}
		cout << endl;
	}
	return ans;
}


int matrixScore(vector<vector<int>>& A) {
	if (A.size() == 0) {
		return 0;
	}
	
	for (int i = 0; i < A.size();i++) {
		if (A[i][0] != 1) {
			for (int j = 0; j < A[i].size(); j++) {
				A[i][j] = !A[i][j];
			}
		}
	}

	for (int i = 0; i < A[0].size(); i++) {
		int count_0 = 0;
		int count_1 = 0;
		for (int j = 0; j < A.size(); j++) {
			if (A[j][i] == 0) {
				count_0 += 1;
			}
			else {
				count_1 += 1;
			}
		}
		if (count_0 > count_1) { //列变换
			for (int k = 0; k < A.size();k++) {
				A[k][i] = !A[k][i];
			}
		}
	}
	int ans = 0;
	for (int i = A.size() - 1; i >= 0; i--) {
		for (int j = A[i].size() - 1; j >= 0; j--) {
			ans += A[i][j] * pow(2, (A[i].size() - j - 1));
			//cout << A[i][j] * pow(2, (A[i].size() - j - 1)) << ' ';
		}
		//cout << endl;
	}
	//cout << ans << endl;
	return ans;
}


bool backtrack(vector<int>& vec, string s, int index, int length, long long sum, int prev) {
	if (index == length) { //一条路结束
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
		if (backtrack(vec, s, i + 1, length, temp + prev, temp)) { // 后移一
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


int wiggleMaxLength(vector<int>& nums) {
	if (nums.size() < 2) {
		return nums.size();
	}
	int prenum = nums[1] - nums[0];
	int count = prenum != 0 ? 2 : 1;
	for (int i = 2; i < nums.size(); i++) {
		if ((nums[i] - nums[i - 1] > 0 && prenum <= 0) || (nums[i] - nums[i - 1] < 0 && prenum>=0)) {
			prenum = nums[i] - nums[i - 1];
			count++;
		}
	}
	cout << count << endl;
	return count;
}



// 快速排序
bool quicksort(vector<int>& vec, int low, int high) {
	if (low >= high) {
		return false;
	}
	swap(vec[low], vec[low + rand() % (high - low + 1)]);  // 随机化版本的快排
	int pivot = vec[low], i = low, j = high;
	while (i < j) {
		while (i < j && vec[j] >= pivot) {
			if (vec[j] != pivot)
				j--;
			else
				return true;
		}
		vec[i] = vec[j];
		while (i < j && vec[i] <= pivot) {
			if (vec[i] != pivot)
				i++;
			else
				return true;
		}
		vec[j] = vec[i];
	}
	vec[i] = pivot;
	return quicksort(vec, low, i - 1) || quicksort(vec, i + 1, high);
}

bool containsDuplicate(vector<int>& nums) {
	return quicksort(nums, 0, nums.size()-1);
}

vector<vector<string>> groupAnagrams(vector<string>& strs) {
	map<string, vector<string>> cmp;
	vector<vector<string>> ans;
	for (const auto a : strs) {
		auto temp = a;
		sort(temp.begin(), temp.end());
		cmp[temp].emplace_back(a);
	}

	for (const auto a : cmp) {
		ans.emplace_back(a.second);
	}
	return ans;
}


int monotoneIncreasingDigits(int N) {
	string num = to_string(N);
	int idx = 0, pre = -1;
	for (int i = 0; i < num.size() - 1; i++) {
		if (pre < num[i]) {
			idx = i;
			pre = num[i];
		}
		if (num[i] > num[i + 1]) {
			num[idx]--;
			for (int j = idx + 1; j < num.size(); j++) {
				num[j] = '9';
			}
			break;
		}
	}
	for (int i = 0; i < num.size(); i++) {
		cout << num[i] << endl;
	}
	int ans = atoi(num.c_str());
	cout << ans;
	return ans;
}

//双向映射
bool wordPattern(string pattern, string s) {
	vector<string> str_s;
	strtool::split(s, str_s, " ");
	int n = (int)pattern.size();
	map<char, set<string>> pat1;
	map<string, set<char>> pat2;
	
	for (int i = 0; i < n; i++) {
		pat1[pattern[i]].insert(str_s[i]);
		pat2[str_s[i]].insert(pattern[i]);
	}
	for (int j = 0; j < n; j++) {
		if (pat1[pattern[j]].size() > 1 || pat2[str_s[j]].size() > 1) {
			return false;
		}
	}
	return true;
}


char findTheDifference(string s, string t) {
	int temp = 0;
	for (auto a : s)
		temp ^= a; //
	for (auto b : t)
		temp ^= b;
	cout << temp;
	return 0;
}

void rotate(vector<vector<int>>& matrix) {
	int n = matrix.size();
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			int temp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = temp;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n / 2; j++) {
			int temp = matrix[i][j];
			matrix[i][j] = matrix[i][n - 1 - j];
			matrix[i][n - 1 - j] = temp;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}


int maximumUniqueSubarray(vector<int>& nums) {
	if (nums.size() == 0)
		return 0;
	unordered_set<int> lookup;
	int left = 0;
	int sum = 0;
	int maxsum = 0;
	for (int i = 0; i < nums.size(); i++) {
		while (lookup.find(nums[i]) != lookup.end()) {//找到重复
			sum -= nums[left];
			lookup.erase(nums[left]);//删左
			left++;
		}
		sum += nums[i];
		maxsum = max(maxsum, sum);
		lookup.insert(nums[i]);
		
	}
	cout<< maxsum;
	return maxsum;
}


string reformatNumber(string number) {
	string temp = "";
	string ans_str = "";
	for (auto a : number) {
		if (a == ' ' || a == '-') {
			continue;
		}
		temp.push_back(a);
	}
	if (temp.size() < 4)
		return temp;
	else if (temp.size() == 4) {
		temp.insert(temp.begin() + 2, '-');
		return temp;
	}
	else {
		for (int i = 3; i < temp.size(); i+=3) {
			temp.insert(temp.begin() + i, '-');
			i += 1;
			if (temp.size() - i == 4) {
				temp.insert(temp.begin() + i + 2, '-');
				break;
			}
		}
		
	}
	for (auto a : temp) {
		cout << a;
	}
	return ans_str;
}


int candy(vector<int>& ratings) {
	int n = ratings.size();
	vector<int> left(n);
	for (int i = 0; i < n; i++) {
		if (i > 0 && ratings[i] > ratings[i - 1]) {
			left[i] = left[i - 1] + 1;
		}
		else {
			left[i] = 1;
		}
	}
	int right = 0, ret = 0;
	for (int i = n - 1; i >= 0; i--) {
		if (i < n - 1 && ratings[i] > ratings[i + 1]) {
			right++;
		}
		else {
			right = 1;
		}
		ret += max(left[i], right);
	}
	return ret;
}



int maximalRectangle(vector<vector<char>>& matrix) {
	int row = matrix.size();
	int col = matrix[0].size();
	vector<vector<int>> tag(row, vector<int>(col, 0));
	return 0;
}





bool canPlaceFlowers(vector<int>& flowerbed, int n) {
	//no push_front on vector but we can insert before begin()；
	flowerbed.insert(flowerbed.begin(), 0);
	flowerbed.emplace_back(0);
	int count = n;
	for (int i = 1; i < flowerbed.size() - 1; i++) {
		if (flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0 && flowerbed[i] == 0) {
			flowerbed[i] = 1;
			count--;
		}
	}
	cout << count << endl;
	return count == 0 ? true : false;
}




vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	if (k > nums.size() || nums.size() == 0) {
		return nums;
	}
	vector<int> ans;
	priority_queue<pair<int, int>> que; // pair内存储<value,index>
	for (int i = 0; i < k; i++) {
		que.emplace(nums[i], i);
	}
	ans.emplace_back(que.top().first);
	
	for (int j = k; j < nums.size(); j++) {
		que.emplace(nums[j], j);
		while (que.top().second < j - k+1) {
			que.pop(); // remove the top element
			
		}
		ans.push_back(que.top().first);
	}

	for (auto a : ans) {
		cout << a << endl;
	}
	return ans;
}




int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
	int count = 0;
	int weight = 0;
	for (int i = 0; i < boxTypes.size(); i++) {
		for (int j = i; j < boxTypes.size(); j++) {
			if (boxTypes[i][1] < boxTypes[j][1]) {
				swap(boxTypes[i], boxTypes[j]);
			}
		}
	}

	for (int k = 0; k < boxTypes.size(); k++) {
		if (count + boxTypes[k][0] <= truckSize) {
			count += boxTypes[k][0];
			weight += boxTypes[k][0] * boxTypes[k][1];
		}
		else {
			weight += (truckSize - count)*boxTypes[k][1];
			//cout << weight;
			return weight;
		}
	}
	//cout << weight;
	return weight;
}


//找出vector中有多少个两个数的和是2的幂
int countPairs(vector<int>& deliciousness) {
	int mod = 1e9 + 7;
	int ans = 0;
	unordered_map<int, int> num;
	for (auto a : deliciousness) {
		for (int i = 0; i < 22; i++) {
			if (num.count((1 << i) - a)) {//找到了
				ans = ans + num[(1 << i) - a];
			}
		}
		num[a]++;
	}
	return ans;
}



//将数组分为三个子数组，前两个
int waysToSplit(vector<int>& nums) {
	const int mod = 1e9 + 7;
	int n = nums.size();
	long long count = 0;
	vector<int> presum(n, 0);
	presum[0] = nums[0];
	for (int i = 1; i < n; i++) {//前缀和
		presum[i] = presum[i-1] + nums[i];
	}

	for (int i = 0; i < n; i++) {
		if (presum[i] * 3 > presum[n - 1]) {
			break;
		}
		int left = i+1, right = n - 2;
		while (left <= right) {//二分找左边界
			int mid = (left + right) / 2;
			if (presum[mid] - presum[i] < presum[i]) {
				left = mid + 1;
			}
			else {
				right = mid - 1;
			}
		}

		int left_2 = i+1, right_2 = n - 2;
		while (left_2 <= right_2) { //找右边界
			int mid = (left_2 + right_2) / 2;
			if (presum[n-1] - presum[mid] < presum[mid] - presum[i]) {
				right_2 = mid - 1;
			}
			else {
				left_2 = mid + 1;
			}
		}
		cout << left << " " << right << " " << left_2 << " " << right_2 << endl;
		int temp = right_2 - left + 1;
		count += temp;
	}

	cout << count << endl;
	return count%mod;
}



vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
	unordered_map<string, int> rec_word;
	int count = 0;
	int n = equations.size();
	for (int i = 0; i < n; i++) {
		if (rec_word.find(equations[i][0]) == rec_word.end()) {//没找到的话，就记录并且赋予编号
			rec_word[equations[i][0]] = count++;
		}
		if (rec_word.find(equations[i][1]) == rec_word.end()) {
			rec_word[equations[i][1]] = count++;
			//将字符串用整数替代掉，（感觉直接写string也可以）
		}
	}

	vector<vector<pair<int, double>>> matrix(count);
	//save the weight creat weight-matrix
	for (int i = 0; i < n; i++) {
		int a = rec_word[equations[i][0]],
			b = rec_word[equations[i][1]];
		matrix[a].emplace_back(make_pair(b, values[i]));
		matrix[b].emplace_back(make_pair(a, 1.00 / values[i]));
	}

	vector<double> ans;
	for (int j = 0; j < queries.size(); j++) {
		double default_answer = -1.000;
		if (rec_word.find(queries[j][0]) != rec_word.end() && rec_word.find(queries[j][1]) != rec_word.end()){
			int a = rec_word[queries[j][0]], b = rec_word[queries[j][1]];
			if (a == b) {
				default_answer = 1.00;
			}
			else {
				vector<double> ratio(count, -1.00);
				queue<int> que;
				que.push(a);
				ratio[a] = 1.00;
				while (!que.empty() && ratio[b] < 0) { //b<0说明b与a无直接关联
					int temp = que.front();
					que.pop();

					for (auto node : matrix[temp]) {//找b
						if (ratio[node.first] < 0) {
							ratio[node.first] = ratio[temp] * node.second;
							que.push(node.first);
						}
					}
				}
				default_answer = ratio[b];
			}
		}
		ans.emplace_back(default_answer);
	}
	for (auto m : ans) {
		cout << m << endl;
	}
	return ans;
}

void dfs(vector<vector<int>> connected, vector<int>& is_visited, int i) {
	for (int j = 0; j < connected[i].size(); j++) {
		if (connected[i][j] == 1 && is_visited[j] != 1) {
			is_visited[j] = 1;
			dfs(connected, is_visited, j);
		}
	}
}

int findCircleNum(vector<vector<int>>& isConnected) {//leetcode547，找相邻省份
	int n = isConnected.size();
	int count = 0;
	vector<int> is_visited(n, 0);
	for (int i = 0; i < n; i++) {
		if (is_visited[i] != 1) {
			is_visited[i] = 1;
			dfs(isConnected, is_visited, i);
			count++;
		}
		
	}
	cout << count << endl;
	return count;
}



int maximumGain(string s, int x, int y) {
	int ans = 0;
	if (x >= y) {
		while (s.find_first_of("ab") != std::string::npos) {
			ans += x;
			std::size_t found = s.find_first_of("ab");
			s.erase(found, found + 2);
			
		}
		while (s.find_first_of("ba") != std::string::npos) {
			ans += y;
			std::size_t found1 = s.find_first_of("ba");
			s.erase(found1, found1 + 2);
			
		}
	}
	else {
		while (s.find_first_of("ba") != std::string::npos) {
			ans += y;
			std::size_t found1 = s.find_first_of("ba");
			s.erase(found1, found1 + 2);
			
		}
		while (s.find_first_of("ab") != std::string::npos) {
			ans += x;
			std::size_t found = s.find_first_of("ab");
			s.erase(found, found + 2);
			
		}		
	}
	for (auto a : s) {
		cout << a << endl;
	}
	cout << ans << endl;
	return ans;
}

//===========================================并查集================================================================

int minimumHammingDistance(vector<int>& source, vector<int>& target, vector<vector<int>>& allowedSwaps) {
	int n = source.size();
	UnionFindOp un(n);
	for (auto& a : allowedSwaps) {
		un.union_element(a[0], a[1]); //下标进行并查集操作初始化
	}
	unordered_map<int, vector<int>> groups;
	for (int i = 0; i < n; i++) { //下标最大为n-1
		groups[un.find(i)].emplace_back(i); //插入根节点所有的子节点
	}
	
	int ans = 0;
	for (int i = 0; i < groups.size();i++) {
		unordered_map<int, int> source_map;
		unordered_map<int, int> target_map;
		for (auto k : groups[i]) { //groups为根节点下的子节点,也包含了根节点本身
			source_map[source[k]]++; //k为下标，source[k]为值，把同属于一个并查集的结点合并
			target_map[target[k]]++;//target中下标属于一个并查集的合并到一块
		}
		for (auto[num, freq] : target_map)//找两者之间多少个相同的
			ans += min(freq, source_map[num]); 
		//freq为target中某个并查集元素的数量，source_map[num]为source_map中属于同一个并查集的元素数量
	}
	cout << n - ans << endl;
	return n - ans;
}

string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
	int n = s.size();
	UnionFindOp un(n);
	for (auto& b : pairs) {
		un.union_element(b[0], b[1]);
	}
	unordered_map<int, vector<int>> sub_roots;
	for (int i = 0; i < n; i++) {
		sub_roots[un.find(i)].emplace_back(s[i]); //根节点index和其对应的所有子节点
	}

	for (auto& [root, sub_root] : sub_roots) {
		sort(sub_root.begin(), sub_root.end(), greater<int>()); // greater不加就错误  
	}
	for (int i = 0; i < s.size(); i++) {
		int x = un.find(i);
		s[i] = sub_roots[x].back();//从后往前，前面排序是从大到小
		sub_roots[x].pop_back();
	}
	return s;
}


//leetcode 1203
//vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
	
//}

//leetcode 684
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
	int n = edges.size();
	UnionFindOp un(n); //初始化
	vector<int> ans;
	
	//结点是从1开始，但是初始化是从0-n-1，因此后续结点记得-1;
	for (auto nodes : edges) {
		if (un.find(un.roots[nodes[0]-1]) != un.find(nodes[1]-1)) {
			un.union_element(nodes[0] - 1, nodes[1] - 1); //逐个合并，逐个判断；
		}
		else {
			ans.clear();
			ans.emplace_back(nodes[0]);
			ans.emplace_back(nodes[1]);
		}
	}

	for (auto a : ans) {
		cout << a << endl;
	}
	return ans;
}


class findUnion {
public:
	vector<int> roots;
	vector<int> size; //秩

	findUnion(int n) {
		roots.resize(n);
		size.resize(n, 1);
		for (int i = 0; i < n; i++) {
			roots[i] = i;
		}
	}

	int find(int x) {
		return x == roots[x] ? x : (roots[x] = find(roots[x]));
	}

	void unions(int x, int y) {
		int a = find(x);
		int b = find(y);
		if (a != b) {
			if (size[a] < size[b]) {
				roots[a] = b;
				size[b] += size[a];
			}
			else { // 若两个节点都为新的，则第二个归并到第一个
				roots[b] = a;
				size[a] += size[b];
			}
		}
	}
};

//class Solution {
//public:   //721账户合并
vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
	//string难以用index直接转换和并查集使用，因此进行转换。
	unordered_map<string, int> emailToIndex;
	unordered_map<string, string> emailToName;
	int email_index = 0;
	for (int i = 0; i < accounts.size(); i++) {
		string name = accounts[i][0];
		int sub_n = accounts[i].size();
		for (int j = 1; j < sub_n; j++) {
			if (!emailToIndex.count(accounts[i][j])) { //没找到才标号
				emailToIndex[accounts[i][j]] = email_index++; // 各邮件编号
				emailToName[accounts[i][j]] = name;  // 邮件对应的人
			}
		}
	}

		//编号完成后，进行属于同一个人的合并
	findUnion un(email_index);
	for (auto account : accounts) {
		string name = account[0];
		string first_email = account[1];
		for (int j = 2; j < account.size(); j++) {
			un.unions(emailToIndex[first_email], emailToIndex[account[j]]);//同一人的邮箱进行归并
		}
	}

	//找出每一个根节点下包含的所有邮箱有哪些
	map<int, vector<string>> rootAndemails;
	for (auto& [email, email_index] : emailToIndex) {
		int index = un.find(email_index);//找根节点
		rootAndemails[index].emplace_back(email);
	}


	//依次存储起来
	vector<vector<string>> ans;
	for (auto [root_index, sub_emails] : rootAndemails) {
		vector<string> temp;
		sort(sub_emails.begin(), sub_emails.end());
		string email_name = emailToName[sub_emails[0]];
		temp.emplace_back(email_name); //先插入姓名，接下来插入邮箱
		for (auto a : sub_emails) {
			temp.emplace_back(a);
		}
		ans.emplace_back(temp);
	}
	return ans;
}
//};

		


//vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
	
//}



class UnionFindOp1 {
public:
	vector<int> roots;
	vector<int> size;//根节点下的全部的结点数
	int count;
	int release;

	UnionFindOp1(int n) {
		roots.resize(n);
		size.resize(n, 1);
		for (int i = 0; i < n; i++) {
			roots[i] = i;
		}
		count = n;
		release = 0;
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
			count--;
		}
		else {
			release++;
		}
	}
};


int makeConnected(int n, vector<vector<int>>& connections) {
	if (connections.size() < n - 1) {
		return -1;
	}
	UnionFindOp1 un(n);
	for (auto temp : connections) {
		un.union_element(temp[0], temp[1]);
	}
	cout << un.release << endl;
	return un.release;
}



//===============================双指针=========================================

int characterReplacement(string s, int k) {
	vector<int> nums(26, 0);
	int n = s.size();
	int max_count = 0;
	int left = 0;
	int right = 0;
	while (right < n) {
		nums[s[right]-'A']++;
		max_count = max(max_count, nums[s[right] - 'A']); //找出出现次数最多的那个字母
		if (right - left + 1 - max_count > k) { //已经到达了k次
			nums[s[left]-'A']--;//左边移除，次数减一,直到满足
			left++;
		}
		cout << left <<"   "<< right << endl;
		right++;
	}
	cout << left << "   " << right << endl;
	cout << right - left << endl;
	return right - left;
}

int equalSubstring(string s, string t, int maxCost) {
	int n = s.size();
	vector<int> temp(n, 0);
	for (int i = 0; i < n; i++) {
		temp[i] = abs(s[i] - t[i]);
	}
	int left = 0;
	int right = 0;
	int sumup = 0;
	int countsize = 0;
	while (right < n) {
		if ((sumup + temp[right]) <= maxCost) { //未超出最大开销
			sumup = sumup + temp[right]; //存储开销
			right++;
		}
		else {
			if (left > right) {
				sumup = 0;
				left++;
				right++;
			}
			else {
				sumup = sumup - temp[left];
				left++; //否则删除左侧元素
			}
			
		}
		countsize = max(countsize, right - left);
	}
	cout << countsize << endl;
	return countsize ;
}



//lc480
vector<double> medianSlidingWindow(vector<int>& nums, int k) {
	vector<double> ans;
	multiset<double> slidingwindow;
	for (int i = 0; i < nums.size(); i++) {
		if (slidingwindow.size() > k) { //大于k个了，就把最开始的删除掉
			slidingwindow.erase(slidingwindow.find(nums[i - k]));
		}
		slidingwindow.insert(nums[i]);//逐渐插入
		if (i >= k - 1) { //滑动窗口已经满了,可以开始求均值
			auto mid = slidingwindow.begin();
			std::advance(mid, k / 2);//移动到中间位置
			ans.emplace_back((*mid + *prev(mid, 1 - k / 2)) / 2);//k为偶数的时候，mid加上了mid左移一位的数
		}
	}
	return ans;
}

//lc480改写
//vector<double> medianSlidingWindow1(vector<int>& nums, int k) {
	
//}


vector<vector<int>> matrixReshape(vector<vector<int>>& nums, int r, int c) {
	if (nums.empty())
		return nums;
	vector<vector<int>> ans(r, vector<int>(c));
	int row = nums.size();
	int col = nums[0].size();
	if ((row * col) == (r * c)) {
		for (int i = 0; i < r * c; i++) {
			ans[i / c][i % c] = nums[i / col][i % col];
		}
	}
	return ans;
}


int minKBitFlips(vector<int>& A, int K) {
	int n = A.size();
	vector<int> diff(n + 1); //差分数组
	int ans = 0;
	int differCount = 0; //differCount记录的是此处的反转次数
	for (int i = 0; i < n; i++) {
		differCount += diff[i];
		if ((A[i] + differCount) % 2 == 0) { //为偶数时
			//若A[i]处是1,1反转了奇数次为0，若此处是0，0反转了偶数次为0
			//那么需要在翻转一次，这样才能全1
			if (i + K > n) { //后面还需反转但是已经越界了
				return -1;
			}
			ans++;
			diff[i]++;
			diff[i + K]--; //差分数组变化时候只改变两端的值
		}
	}
	return ans;
}

// 二维矩阵前缀和
//f(i,j)=f(i−1,j)+f(i,j−1)−f(i−1,j−1)+matrix[i][j]
class NumMatrix {
private:
	vector<vector<int>> two_dimension_presum;
public:
	NumMatrix(vector<vector<int>>& matrix) {
		if (matrix.empty()) {
			return;
		}
		int m = matrix.size();
		int n = matrix[0].size();
		two_dimension_presum.resize(m + 1, vector<int>(n + 1));
		//two_dimension_presum[i][j]是以(i-1,j-1)为右下角的矩阵的和
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				two_dimension_presum[i][j] = two_dimension_presum[i - 1][j] + two_dimension_presum[i][j - 1] - two_dimension_presum[i - 1][j - 1] + matrix[i - 1][j - 1];
				//计算二维矩阵前缀和
			}
		}
	}

	int sumRegion(int row1, int col1, int row2, int col2) {
		return two_dimension_presum[row2 + 1][col2 + 1] - two_dimension_presum[row1][col2 + 1] - two_dimension_presum[row2 + 1][col1] + two_dimension_presum[row1][col1];
	}
};






//回溯+记忆化搜索
class BackAndMem {
private:
	vector<string> temp; //
	vector<vector<string>> ans;
	vector<vector<int>> mem; // mem[i][j]记录从i到j位置是否为回文字符串

	int n;
public:
	void dfs(string s, int index) {//判断s从index开始，之后的情况
		if (index == n) {
			ans.emplace_back(temp);
			return;
		}
		else {
			for (int i = index; i < n; i++) {
				if (isPalindrome(s, index, i) == 1) { //是回文，那么继续往后
					temp.emplace_back(s.substr(index, i - index + 1));
					dfs(s, i + 1);
					temp.pop_back();//说明dfs此条路失败了
				}
			}
		}
	}

	int isPalindrome(string s, int i, int j) { //判断字符串s的i到j是否为回文串
		if (mem[i][j]) {
			return mem[i][j];
		}
		if (i >= j) {
			return mem[i][j] = 1;
		}
		return mem[i][j] = (s[i] == s[j] ? isPalindrome(s, i+1, j-1) : -1);
	}

	vector<vector<string>> partition(string s) {
		n = s.size();
		mem.assign(n, vector<int>(n));
		dfs(s, 0);
		return ans;
		
	}

	/*
	此处的mem就是当作标记位，对于更新mem，可以使用dp,不需要isPalindrome重复判断
	vector<vector<string>> partition(string s) {
		n = s.size();
		mem.assign(n, vector<int>(n,true));

		for(int i=n-1;i>=0;i--){
			for(int j=i+1;j<n;j++){
				mem[i][j] = (s[i]==s[j])&&(mem[i+1][j-1]);
			}
		}
		dfs(s, 0);
		return ans;

	}
	*/
	
};

//分割为回文子串的最小次数
int minCut(string s) {
	int n = s.size();
	vector<vector<int>> tag(n, vector<int>(n, true)); //标记是否为回文字符串
	vector<int> f(n, INT_MAX); //记录分割的最小次数

	for (int i = n - 1; i >= 0; i--) {
		for (int j = i + 1; j < n; j++) {
			tag[i][j] = (s[i] == s[j]) && tag[i + 1][j - 1]; //dp方法更新tag,找出回文串的位置
		}
	}
	//找出位置以后，可以更新f，记录次数，但是要考虑整个字符串为回文串，此时分割次数为0
	for (int i = 0; i < n; i++) {
		if (tag[0][i]) { //为回文，往后
				f[i] = 0;
		}
		else { //i处不是回文了，此时f[i]处仍然是INT_MAX
			for (int j = 0; j < i; j++) { //从j到i的子串更新
				if (tag[j + 1][i]) { 
					//0到i已经不是回文了，在考虑一下1到i之间是否还有回文
					//若没有回文了，那么当j=i-1时候，tag[j+1][i]=1
					//f[i] = min(f[i],f[i-1]+1);
					//f[i] = min(MAX_INT, f[i-1]+1);
					f[i] = min(f[i], f[j] + 1);
				}
			}
		}
	}
	cout << f[n - 1] << endl;
	return f[n - 1];
}

int beautySum(string s) {
	int n = s.size();
	int ans = 0;

	for (int left = 0; left < n; left++) {
		vector<int> temp(26, 0);
		for (int right = left; right < n; right++) {
			int _max = 1;
			int _min = n;
			temp[s[right] - 'a']++;
			for (auto a : temp) {
				if (a > 0) {
					_max = max(_max, a);
					_min = min(_min, a);
				}
			}
			ans += _max - _min;
		}
	}
	cout << ans << endl;
	return ans;
}


//中缀表达式计算，此处只有+, -, (, ), " "(空格)
int calculate(string s) {
	int sign = 1; //初始化符号位
	int ans = 0;
	int n = s.size();
	string temp;
	stack<int> ops; //存储符号位
	ops.push(1);
	int i = 0; //index
	while (i < n) {
		if (s[i] == '(') {
			ops.push(sign);
		}
		else if (s[i] == ')') {
			ops.pop(); //只有当括号去除完毕后，才删掉（之前的符号，类似于去掉表达式的括号
		}
		else if (s[i] == '+') {
			if (!temp.empty()) {
				ans += sign * (stoi(temp)); //这个符号用完了，准备用下一个了
			}
			sign = ops.top();
			temp.clear();
		}
		else if (s[i] == '-') {
			if (!temp.empty()) {
				ans += sign * (stoi(temp));
			}
			sign = -ops.top(); //符号的改变类似于去掉括号
			temp.clear();
		}
		else {
			if (s[i] != ' ') { //表达式中是存在空格的
				temp.push_back(s[i]); //说明此时是数字，加入到temp中
			}
		}
		i++; //后移
	}
	//考虑最后一个数字
	if (!temp.empty()) {
		ans += sign * (stoi(temp));
	}
	return ans;
}


//中缀表达式计算，只有 +,-,*,/ 四种符号 和 ' '空格 
int calculate2(string s) { //注意先计算的是*, /, 可以把减法取相反数变为加法
	stack<int> nums; //存储数字
	int ans = 0;//最后的结果
	int temp = 0; //临时存储
	char sign = '+'; //保存前一个符号位
	int n = s.size();
	int i = 0;
	while (i < n) {
		if (isdigit(s[i])) { //s[i]是数字的时候
			temp = temp * 10 + (s[i] - '0'); //数字不一定只有一位
		}
		if ((!isdigit(s[i]) && s[i] != ' ') || i == n - 1) { //s[i]为符号且不为空,temp已经保存了这个符号之前的数字
			//考虑最后一位数字，此时i=n-1
			if (sign == '+') {
				nums.push(temp);
			}
			else if (sign == '-') {
				nums.push(-temp);
			}
			else if (sign == '*') {  //先算乘除法，在保存到nums中，最后一起加起来
				nums.top() = nums.top() * temp;
			}
			else {
				nums.top() = nums.top() / temp;
			}
			sign = s[i]; //更新符号
			temp = 0; //重置temp，为保存s[i]之后的数字做准备
		}
		i++;
	}
	while (!nums.empty()) {
		ans += nums.top();
		nums.pop();
	}
	cout << ans << endl;
	return ans;
}


string removeDuplicates(string S) {
	string str;
	for (auto a : S) {
		if (str.empty() || a != str.back()) {
			str.push_back(a);
		}
		else{
			str.pop_back();
		}
	}
	return str;
}

//入度出度来解决树的问题 331
bool isValidSerialization(string preorder) { 
	stack<int> rec;
	int i = 0;
	int n = preorder.size();
	rec.push(1);
	while (i < n) {
		if (isdigit(preorder[i])) { //为数字时
			while (isdigit(preorder[i])) {
				i++; //数字不一定是一位
			}
			if (rec.empty()) {
				return false;
			}
			rec.top() -= 1;
			if (rec.top() == 0) {
				rec.pop();
			}
			rec.push(2);
			i++;
		}
		else {//为符号时
			if (rec.empty()) {
				return false;
			}
			if (i < n && preorder[i] == ',') {
				i++;
			}
			else { //#
				rec.top() -= 1;
				if (rec.top() == 0) {
					rec.pop();
				}
				i++;
			}
		}
	}
	cout << rec.empty() << endl;
	return rec.empty();
}


// 最长上升子序列问题    LIS
// 300. 最长递增子序列
int lengthOfLIS(vector<int>& nums) {
	int n = nums.size();
	if (n == 0) {
		return 0;
	}
	vector<int> dp(n);
	//dp代表的是当前位置最长的序列
	//状态方程  dp[i] = max{dp[i], dp[j]+1};  j为小于i的点，且nums[j]<nums[i]
	for (int i = 0; i < n; i++) {
		dp[i] = 1;
		for (int j = 0; j < i; j++) {
			if (nums[j] < nums[i]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	cout << *max_element(dp.begin(),dp.end()) << endl;
	return *max_element(dp.begin(), dp.end());
}


//case2 使用贪心算法+二分查找
int lengthOfLIS_CASE2(vector<int>& nums) {
	int n = nums.size();
	if(n == 0){
		return 0;
	}
	int len = 1;
	vector<int> b(n+1);
	b[len] = nums[0];
	for (int i = 1; i < n; ++i) {
		if (nums[i] > b[len]) {
			b[++len] = nums[i];
		}
		else {
			int l = 0, r = len;
			while (l <= r) {
				int mid = l + (r - l) / 2;
				if (b[mid] < nums[i]) {
					l = mid + 1;
				}
				else {
					r = mid - 1;
				}
			}
			b[l] = nums[i];
		}
	}
	return len;
}




//Russian Doll Envelopes
//俄罗斯套娃
// 首先我们将所有的信封按照 w 值第一关键字升序、h 值第二关键字降序进行排序；
//随后我们就可以忽略 w 维度，求出 h 维度的最长严格递增子序列，其长度即为答案。
int maxEnvelopes(vector<vector<int>>& envelopes) {
	if (envelopes.empty()) {
		return 0;
	}
	int n = envelopes.size();
	vector<int> dp; //dp表示当前处最大的信封嵌套次数
	

	sort(envelopes.begin(), envelopes.end(), [](auto const& e1, auto const& e2) { //第一个元素先由小到大比较进行总体排序
		return e1[0] < e2[0] || (e1[0]==e2[0]&&e1[1]>e2[1]); //按照 w 值第一关键字升序、h 值第二关键字降序进行排序
	});

	//更新dp
	dp.resize(n, 1);
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (envelopes[j][1] < envelopes[i][1]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	cout << *max_element(dp.begin(), dp.end()) << endl;
	return *max_element(dp.begin(), dp.end());
}



//螺旋形输出矩阵
vector<int> spiralOrder(vector<vector<int>>& matrix) {
	if (matrix.empty()||matrix[0].empty()) {
		return {};
	}
	int m = matrix.size();
	int n = matrix[0].size();
	vector<vector<int>> visited(m, vector<int>(n, 0));
	vector<vector<int>> way = { {0,1},{1,0},{0,-1},{-1,0} }; //右，下，左，上

	int row = 0, column = 0;
	int total = m * n;
	int countIndex = 0;
	vector<int> ans(total);
	
	for (int i = 0; i < total; i++) {
		ans[i] = matrix[row][column];
		visited[row][column] = 1;
		int nextrow = row + way[countIndex][0];
		int nextcol = column + way[countIndex][1]; //更新位置
		if (nextrow < 0 || nextrow >= m || nextcol < 0 || nextcol >= n || visited[nextrow][nextcol]) { //越界了
			countIndex = (countIndex + 1) % 4;
		}
		row = row + way[countIndex][0];
		column = column + way[countIndex][1]; //未越界时，nextrow = row ，。。。。
		//越界后，更新
	}
	return ans;
}



//下一个最大元素I
//找出 nums1 中每个元素在 nums2 中的下一个比其大的值
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
	if (nums1.empty() || nums2.empty()) {
		return {};
	}
	int n = nums1.size();
	stack<int> st;
	map<int, int> mp;
	vector<int> ans;

	//找出nums2中每个元素右边第一个大于他的数
	for (int i = nums2.size() - 1; i >= 0; --i) {
		int temp = nums2[i];
		while (!st.empty() && st.top() < temp) {
			st.pop();  // 出栈小于temp的数， 此时栈里最终留下的数，即为第一个大于temp的数
		}
		mp[temp] = st.empty() ? -1 : st.top();
		st.push(temp);
	}

	for (int i = 0; i < n; ++i) {
		if (mp.count(nums1[i])) {
			ans.emplace_back(mp[nums1[i]]);
		}
		else {
			ans.emplace_back(-1);
		}
	}

	return ans;
}

//下一个最大元素II
//给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素
//和I相比，此处的数组可以循环,通过取余完成循环
vector<int> nextGreaterElements(vector<int>& nums) {
	if (nums.empty()) {
		return {};
	}
	int n = nums.size();
	vector<int> ans(n,-1);
	stack<int> st;

	for (int i = 0; i < 2*n-1; i++) { //i<2n-1时候,i%n能完整覆盖 0~(n-1)。 n-1 < i < 2n-1时，开始循环搜索
		while (!st.empty() && (nums[st.top()] < nums[i % n])) { 
			ans[st.top()] = nums[i % n];
			st.pop();
		}
		st.push(i % n); //存下标
	}
	return ans;
}

//下一个最大元素III
/*给你一个正整数 n ，请你找出符合条件的最小整数(最小的大于n的整数)，其由重新排列 n 中存在的每位数字组成，并且其值大于 n 。如果不存在这样的正整数，则返回 -1 。*/
int nextGreaterElement(int n) {
	//先将n转为可操作的vector或者string
	string s = to_string(n);
	//从右往左找到第一个不满足从左到右为降序的数字
	//如： 645321，则为4
	int len = s.size();
	int first_min = -1;
	for (int i = len - 1; i >= 1; i--) {
		if (s[i - 1] < s[i]) {
			first_min = i - 1;
			break;
		}
	}
	if (first_min == -1)
		return -1;

	//在first_min到len-1之间，找到满足大于s[first_min]的最小数字
	int first_big = -1;
	int cmp_temp=INT_MAX;
	for (int j = first_min; j < len; j++) {
		if (s[j] > s[first_min]) {//找到大的了
			if (cmp_temp >= s[j]) { // s[first_min]<cmp_temp,同时还存在比cmp_temp更小的数，也满足s[first_min]<cmp_temp
				cmp_temp = s[j]; //大于s[first_min]的最小数字
				first_big = j;
			}
		}
	}
	if (first_big == -1) {
		return -1;
	}
	swap(s[first_min], s[first_big]);
	//若有158476531
	/*
		首先找到了4，为s[first_min]
		再找到5为s[first_big]
		交换后变为158576431
		为了满足1585后的  76431有最小值，此时76431为降序
		将其转换为升序即可得到最小值
	*/
	int i = first_min + 1; //交换
	int j = len - 1;
	while (i < j) {
		swap(s[i], s[j]);
		i++;
		j--;
	}	
	//12222333 = > 12223332
	//12223332
	//12223233
	int ans = atoi(s.c_str());
	cout << ans << endl;
	return ans;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*========================================字符串  DP    DFS   DFS+MEM====================================================*/

//不同的子序列
//s为父串
//给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
class dpAndMemory {
private:
	vector<vector<int>> dfs_vec;
	vector<vector<int>> dp_vec;
public:
	int dfs(string s, int s_index, string t, int t_index) {
		if (s_index == s.size()) {
			return 0;
		}
		if (t_index = t.size()) {
			return 1;
		}
		if (s[s_index] == t[t_index]) {
			return dfs(s, s_index + 1, t, t_index) + dfs(s, s_index + 1, t, t_index + 1);
			//位置相同的时候，可以选择父串和子串全都往后移动，或者父串往后移动，子串不动
		}
		else {
			return dfs(s, s_index + 1, t, t_index);
			//不相同的时候，父串往后移动，子串不动
		}
	}

	//dfs记忆化搜索
	int dfs_mem(string s, int s_index, string t, int t_index) {
		//边界条件
		if (t_index == t.size()) { //子串匹配成功
			return 1;
		}
		if (s_index == s.size()) { //父串走完了，字串没走完，匹配失败
			return 0;
		}
		
		if (dfs_vec[s_index][t_index] != -1) //访问过了
			return dfs_vec[s_index][t_index];


		if (s[s_index] == t[t_index]) { //判断一下当前位置的的字符是否相同
			dfs_vec[s_index][t_index] = dfs_mem(s, s_index + 1, t, t_index + 1) + dfs_mem(s, s_index + 1, t, t_index);
		}
		else {
			dfs_vec[s_index][t_index] = dfs_mem(s, s_index + 1, t, t_index);
		}
		
		return dfs_vec[s_index][t_index];
	}

	//dp
	int dp(string s, string t) { //t_len x s_len大小
		int s_len = s.size();
		int t_len = t.size();
		if (s_len < t_len) {
			return 0;
		}
		dp_vec.resize(t_len + 1, vector<int>(s_len + 1));

		for (int i = 0; i <= s_len; ++i) {
			dp_vec[t_len][i] = 1;
		}

		for (int i = t_len - 1; i >= 0; i--) {
			for (int j = s_len - 1; j >= 0; j--) {
				if (t[i] == s[j]) {
					dp_vec[i][j] = dp_vec[i + 1][j + 1] + dp_vec[i][j + 1]; 
				}
				else {
					dp_vec[i][j] = dp_vec[i][j + 1];//主串动，子串不动
				}
			}
		}
		return dp_vec[0][0];
	}

	int numDistinct(string s, string t) {
		int s_len = s.size();
		int t_len = t.size();

		int dfs_ans = dfs(s, 0, t, 0);


		dfs_vec.resize(s_len, vector<int>(t_len, -1)); 
		//dfs_mem[i][j]是s[0:i] t[0:j]中匹配的子串的个数
		int def_mem_ans = dfs_mem(s, 0, t, 0);
		cout << def_mem_ans << endl;

		int dp_ans = dp(s, t);

		return 0;
	}
};

/**/


//两个字符串的最长公共子序列的长度
int longestCommonSubsequence(string text1, string text2) {
	int l1 = text1.size();
	int l2 = text2.size();
	if (!(l1 && l2)) {
		return 0;
	}
	vector<vector<int>> dp(l1 + 1, vector<int>(l2 + 1, 0));
	for (int i = 1; i <= l1; i++) {
		for (int j = 1; j <= l2; j++) {
			if (text1[i-1] == text2[j-1]) {
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			else {
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
	}
	cout << dp[l1][l2] << endl;
	return dp[l1][l2];
}


bool isNum(string a) {
	return (a != "+" && a != "-" && a != "*" && a != "/");
}

int evalRPN(vector<string>& tokens) {
	if (tokens.empty()) {
		return 0;
	}
	int n = tokens.size();
	stack<int> st;
	int ans = 0;
	for (string a : tokens) {
		if (isNum(a)) { //数字
			st.push(atoi(a.c_str()));
		}
		else {
			int e2 = st.top();
			st.pop();
			int e1 = st.top();
			st.pop();
			if (a == "+") {
				st.push(e1 + e2);
			}
			else if (a == "-") {
				st.push(e1 - e2);
			}
			else if (a == "*") {
				st.push(e1 * e2);
			}
			else {
				st.push(e1 / e2);
			}
		}
	}
	cout << st.top();
	return st.top();
}

//5710 订单采购
int getNumberOfBacklogOrders(vector<vector<int>>& orders) {
	priority_queue<pair<int, int>> buy_list;  //别人的购买价格，从最高开始看
	priority_queue<pair<int, int>> sale_list; //别人的销售价格，从最低开始看，因为优先队列默认顶部为最大，因此加符号颠倒栈内元素
	int ans = 0;
	for (auto a : orders) {
		if (a[2]) { //sale销售订单,查看其他人当前 ==最高的== 采购订单，若高于我的sale价格，则卖出
			sale_list.push(make_pair(-a[0], a[1]));
		}
		else {//buy采购订单，差看其他人当前的  ==最低的==  销售价格， 若低于 我的采购价格，则买入
			buy_list.push(make_pair(a[0], a[1]));
		}
		//就是需要sale<buy 才成立
		ans += a[1]; //统计所有的数量

		while (!buy_list.empty()) {
			if (sale_list.empty() || -sale_list.top().first>buy_list.top().first) { //价格不满足
				break;
			}
			//有合适的
			auto buy_temp = buy_list.top();
			buy_list.pop();
			auto sale_temp = sale_list.top();
			sale_list.pop();
			
			//更新数量
			int sale_amount = min(buy_temp.second, sale_temp.second);
			buy_temp.second -= sale_amount;
			sale_temp.second -= sale_amount;
			ans = ans - 2 * sale_amount; //减去卖出的和采购的订单
			if (buy_temp.second) { //销售量不满足采购量
				buy_list.push(buy_temp);
			}
			else{ //采购完成，但是销售还有
				sale_list.push(sale_temp);
			}

		}
	}
	cout << ans;
	return ans;
}


//leetcode 1802
//类似于三角形的台阶
int maxValue(int n, int index, int maxSum) {
	if (maxSum < n) {
		return 1;
	}
	//取index位置
	int l = index, r = index;
	int ans = 1; //index处的值
	int surplus = maxSum - n; //剩余值
	while (l > 0 || r < n - 1) {//还未到边界
		int len = r - l + 1;
		if (surplus < len) { //剩余值不够len内分了
			break;
		}
		ans++;
		surplus -= len; //len内均匀加1
		l = max(0, l - 1);
		r = min(n - 1, r + 1); //l和r不会满足越界条件，即使l=0,r=n-1，只有surplus不够分，才会break
	}
	//surplus不够窗口内分了
	ans += surplus / n; //均分一下
	return ans;

	/*
	其实while之后也可以写成这种形式
	while(l>=0||r<=n-1){//还未到边界
		int len = r-l+1;
		if(surplus<len){ //剩余值不够len内分了
			break;
		}
		ans++;
		surplus -=len; //len内均匀加1
		l = max(0,l-1);
		r = min(n-1,r+1); //l和r不会满足越界条件，即使l=0,r=n-1，只有surplus不够分，才会break
	}
	//surplus不够窗口内分了
	return ans;


	这样写的不好就是当len的长度是窗口时候，我的surplus还够窗口内分很多次，那样的话就会循环非常多次的while
	影响了时间复杂度
	因此，当len等于窗口长度时，就跳出，剩余的值直接均分，这样避免了特殊情况下大量的循环
	比如用例  2 1 21
	会额外循环9次
	*/
}


//132模式
// i < j < k   nums[i]<nums[k]<nums[j]  
bool find132pattern(vector<int>& nums) {
	if (nums.empty()) {
		return false;
	}
	priority_queue<int,vector<int>,greater<int>> sto; //从小到大排序
	int temp = INT_MIN; //temp来标记当前的最小值
	for (int i = nums.size() - 1; i >= 0; i--) {
		if (nums[i] < temp) { //找位置1
			return true;
		}
		while (!sto.empty() && sto.top() < nums[i]) {//右边最小的数小于当前数字，此时已经满足了该数大于它右侧的一个数
			temp = sto.top();  //temp记录的是位置2的数
			cout << temp << endl;
			sto.pop(); //把栈内小于当前数字的全部出栈，此时栈顶是位置3的数字
		}
		sto.push(nums[i]);
	}
	return false;
}


class Solutionnew22 {
private:
	vector<vector<int>> step = { {-1,0},{1,0},{0,-1},{0,1} };
	vector<vector<int>> visited;
public:
	bool dfs(vector<vector<char>>& board, vector<vector<int>>& step, int index, string word, int i, int j) {
	//step不加&会导致程序运行时间大幅度增长，因为&step引用step就是传入step本身
	//而直接写vector<vector<int>> step是传值，相当于对step进行了一次拷贝
		if (board[i][j] != word[index]) {
			return false;
		}
		if (index == word.size() - 1) {
			return true;
		}
		visited[i][j] = true;
		for (auto& a : step) {
			int newrow = i + a[0], newcol = j + a[1];
			if (newrow >= 0 && newrow < board.size() && newcol >= 0 && newcol < board[0].size()) {
				if (!visited[newrow][newcol]) {
					bool looptemp = dfs(board, step, index + 1, word, newrow, newcol);
					if (looptemp) {
						return true;
						break;
					}
				}
			}
		}
		visited[i][j] = false;
		return false;
	}

	bool exist(vector<vector<char>>& board, string word) {
		int index = 0; //记录word当前在第几位
		int m = board.size(), n = board[0].size();
		visited.resize(m, vector<int>(n, false));
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				bool ans = dfs(board, step, index, word, i, j);
				if (ans) {
					return ans;
				}
			}
		}
		return false;
	}
};


//数组中两个数异或的最大值
class trie_ex { //字典树
private:
	trie_ex* next[2] = {nullptr}; //不设置nullptr的话，后续对于补位的那些数字，指针会报错
public:
	trie_ex() {}
	void insert(int x, int n) {
		//从高位到低位依次，因为对于二进制而言，高位为1最大
		trie_ex* node = this;
		for (int i = n-1; i >= 0; i--) { 
			int temp = (x >> i) & 1; //二进制下标从0开始，因此只需要移动i位，从1开始则移动i-1位
			if (node->next[temp] == nullptr) {
				node->next[temp] = new trie_ex();
			}
			node = node->next[temp];
		}
	}

	int tr_find(int x,int n) {
		trie_ex* node = this;
		int ans_find = 0; //根据x的值，所查找到的能保证最大异或值的数
		for (int i = n - 1; i >= 0; i--) { //全部以最长的位数为准，少的补齐
			int temp = (x >> i) & 1; //取第i位
			if (node->next[!temp]) { //存在,那么走反方向,反方向在的话
				node = node->next[!temp];
				ans_find = ans_find * 2 + !temp;
			}
			else { //反方向为空，走同向
				node = node->next[temp];
				ans_find = ans_find * 2 + temp;
			}
		}
		ans_find = ans_find ^ x;
		return ans_find;
	}
};
int findMaximumXOR(vector<int>& nums) {
	trie_ex tr;
	int ans = 0;
	int max_ele = *max_element(nums.begin(), nums.end());
	int n = 0;
	while (max_ele) {
		n++;
		max_ele = max_ele >> 1;
	}
	for (auto c : nums) {
		tr.insert(c,n);
	}
	for (auto a : nums) {
		ans = max(ans, tr.tr_find(a,n));
	}
	cout << ans << endl;
	return ans;
}

//子集I
class Perm {
private:
	vector<vector<int>> ansI;
	vector<int> temp;
public:
	void dfs(vector<int>& nums, int level, vector<vector<int>>& ans, vector<int> temp) {
		if (level == nums.size()) {
			ans.emplace_back(temp);
			return;
		}
		//level+1 表示第level个位置上的元素是否被选中
		dfs(nums, level + 1, ans, temp); // 此时不将当前元素放入temp
		temp.emplace_back(nums[level]);
		dfs(nums, level + 1, ans, temp);
	}

	vector<vector<int>> subsets(vector<int>& nums) {
		dfs(nums, 0, ansI, temp);
		for (auto a : ansI) {
			for (auto b : a) {
				cout << b << " ";
			}
			cout << endl;
		}
		return ansI;
	}
};


int numRabbits(vector<int>& answers) {
	if (answers.empty()) {
		return 0;
	}
	sort(answers.begin(), answers.end());
	int total = answers[0] + 1; //记录总数
	int temp_count = answers[0]; //记录当前颜色的数目
	int temp_color = answers[0]; //记录当前类别
	for (int i = 1; i < answers.size(); ++i) {
		if (answers[i] == temp_color) {
			if (temp_count == 0) { //当前颜色计算完了,即使数量相同，那也是另一种颜色，比如2,2,2,2,最后一个2为其他颜色
				temp_count = answers[i] + 1;
				total += temp_count;
			}
			temp_count--;
		}
		else {
			temp_color = answers[i];
			temp_count = answers[i];
			total += temp_count + 1;
		}

	}
	return total;
}

//   81搜索旋转排序数组
bool search(vector<int>& nums, int target) {
	int l = 0, r = nums.size() - 1;
	while (l <= r) {
		while (l < r && nums[l] == nums[l + 1]) //l<r保证了l后至少还有一位，等号不成立
			l++;
		while (l < r && nums[r - 1] == nums[r])
			r--;
		int mid = (l + r) / 2;
		if (nums[mid] == target) {
			return true;
		}
		//不重复
		if (nums[l] <= nums[mid]) { //左侧有序了
			if (target >= nums[l] && target <= nums[mid]) { //target在左侧
				r = mid - 1;
			}
			else {
				//左侧有序但是target在右侧
				l = mid + 1;
			}
		}else if (nums[mid] <= nums[r]) { //右侧有序了
			if (target >= nums[mid] && target <= nums[r]) {
				l = mid + 1;
			}
			else { //右侧有序但是在左侧
				r = mid - 1;
			}
		}
	}
	return false;
}


int findMin(vector<int>& nums) {
	//如何判断在最小值的左侧、右侧
	int n = nums.size();
	int l = 0, r = n - 1;
	while (l <= r) {
		int mid = (l + r) / 2;
		if (nums[mid] < nums[r]) { //右侧有序,不能写等号，有重复元素
			r = mid;
		}
		else if (nums[mid] > nums[r]) { //右侧无序minimum在mid右
			l = mid + 1;
		}
		else {
			r--; //重复时右侧挨个减掉
		}
	}
	return nums[l];

}


//474 1和0
/*
给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。
如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
*/
pair<int, int> zeroandone(string str) {
	int count0 = 0, count1 = 0;
	for (auto a : str) {
		a == '0' ? count0++ : count1++;
	}
	return make_pair(count0, count1);
}

int findMaxForm(vector<string>& strs, int m, int n) {
	int len = strs.size();
	vector<pair<int, int>> count_word;
	for (auto a : strs) {
		count_word.push_back(zeroandone(a));
	}
	vector<vector<vector<int>>> dp(len+1,vector<vector<int>>(m+1,vector<int>(n+1)));
	
	for (int k = 1; k <= len; ++k) {
		for (int i = 0; i <= m; ++i) { 
			for (int j = 0; j <= n; ++j) {  //此时使用i个0,j个1
				if (i >= count_word[k - 1].first && j >= count_word[k - 1].second) {
					dp[k][i][j] = max(dp[k - 1][i][j], dp[k - 1][i - count_word[k - 1].first][j - count_word[k - 1].second] + 1);
				}
				else {
					dp[k][i][j] = dp[k - 1][i][j];
				}
			}
		}
	}


	return dp[len][m][n];
}

//约瑟夫环
int findTheWinner(int n, int k) {
	vector<int> nums(n);
	for (int i = 0; i < n; i++) {
		nums[i] = i + 1;
	}

	int begin = 0;
	while (n > 1) {
		int step = (begin + k - 1) % n; //移动到的位置
		int del_ele = nums[step];
		for (int j = step; j < nums.size() - 1; j++) {
			nums[j] = nums[j + 1];
		}
		
		nums[nums.size() - 1] = del_ele;
		nums.pop_back();
		n--;
		begin = (step+1)%n;
		begin--;

	}
	cout << nums[0];
	return nums[0];
}



int min_ele(int a, int b, int c) {
	int m = a <= b ? a : b;
	int n = a <= c ? a : c;

	return (m <= n ? m : n);
}

//最少侧跳次数
int minSideJumps(vector<int>& obstacles) {
	int n = obstacles.size();
	vector<vector<int>> dp(n, vector<int>(4));
	int begin = 2;
	int ans = 0;
	for (int i = 0; i < n; i++) {
		if (i == 0) {
			dp[0][1] = 1, dp[0][3] = 1, dp[0][2] = 0;
		}
		//dp[i][1] = 1, dp[i][3] = 1;
		if (obstacles[i] != 0) {
			dp[i][obstacles[i]] = n;
		}

	}

	//dp[n][3]
	//dp[i][1] = min(dp[i-1][1],dp[i][2],dp[i][3])
	for (int j = 1; j < n; ++j) {
		for (int i = 1; i <= 3; ++i) {
			if (obstacles[j] == i) {
				dp[j][i] = n;
			}
			else {
				if (i == 1) {
					dp[j][i] = min_ele(dp[j - 1][1], dp[j - 1][2] + 1 + (dp[j][2] >= n), dp[j - 1][3] + 1 + (dp[j][3] >= n));
				}
				else if (i == 2) {
					dp[j][i] = min_ele(dp[j - 1][2], dp[j - 1][1] + 1 + (dp[j][1] >= n), dp[j - 1][3] + 1 + (dp[j][3] >= n));
				}
				else {
					dp[j][i] = min_ele(dp[j - 1][3], dp[j - 1][1] + 1 + (dp[j][1] >= n), dp[j - 1][2] + 1 + (dp[j][2] >= n));
				}
			}
		}
	}
	return min_ele(dp[n - 1][1], dp[n - 1][2], dp[n - 1][3]);
}


//打家劫舍II   213
int rob(vector<int>& nums) {
	//房间i偷或者不偷
	// dp[i]表示到达房间i时，所获得的利润
	//dp[i] = max(dp[i-1], dp[i-2]+dp[i]);
	int n = nums.size();
	if (n == 1) {
		return nums[0];
	}
	else if (n == 2) {
		return max(nums[0], nums[1]);
	}

	vector<int> dp(n - 1, 0); //0-n-2
	vector<int> dp1(n, 0); //1-n-1  //
	dp[0] = nums[0];
	dp[1] = max(nums[0], nums[1]); //注意这里的初始化
	dp1[1] = nums[1];
	dp1[2] = max(nums[2], nums[1]);
	for (int i = 2; i < n - 1; i++) {
		dp[i] = max(dp[i - 1], (dp[i - 2] + nums[i]));
	}
	for (int i = 3; i < n; i++) {
		dp1[i] = max(dp1[i - 1], (dp1[i - 2] + nums[i]));
	}
	return max(dp[n - 2], dp1[n - 1]);
}



//474 一和零  1和0
class ZeroAndOne {
public:
	vector<int> countZeroOne(string s) {
		vector<int> count(2,0);
		for (auto a : s) {
			count[a - '0']++;
		}
		return count;
	}

	int findMaxForm(vector<string>& strs, int m, int n) { //最多m个0，n个1
		int lens = strs.size();
		vector<vector<vector<int>>> dp(lens + 1, vector<vector<int>>(m + 1,vector<int>(n+1)));

		//dp[k][i][j]表示在0-k之间，使用i个0 j个1的最大子串数目
		for (int k = 1; k <= lens;++k) {
			auto count = countZeroOne(strs[k-1]);
			for (int i = 0; i <= m; i++) {
				for (int j = 0; j <= n; j++) {
					//不能直接将i==0||j==0时，dp[i][j]=0,因为可能子串为“000”，“111”等
					if (i >= count[0] && j >= count[1]) {
						dp[k][i][j] = max(dp[k - 1][i][j], dp[k - 1][i - count[0]][j - count[1]] + 1);
					}
					else {//装不下
						dp[k][i][j] = dp[k - 1][i][j];
					}
					
				}
			}
		}
		return dp[lens][m][n];
	}
};


//单线程CPU   短作业调度1834
vector<int> getOrder(vector<vector<int>>& tasks) {
	int n = tasks.size();
	vector<int> index_rec(n);
	iota(index_rec.begin(), index_rec.end(), 0);
	sort(index_rec.begin(), index_rec.end(), [&](int a, int b) { //[&]    lambda以引用方式捕获
		return tasks[a][0] < tasks[b][0]; //用大小为n的vector记录下标
	});

	priority_queue<pair<int, int>,vector<pair<int, int>>,greater<pair<int, int>>> que;
	int time_rec = 0;
	int p = 0; //记录当前进入了多少个任务
	vector<int> ans;

	for (int i = 0; i < n; ++i) {
		if (que.empty()) {
			time_rec = max(time_rec, tasks[index_rec[p]][0]);
		}
		while (p < n && tasks[index_rec[p]][0] <= time_rec) {
			que.emplace(make_pair(tasks[index_rec[p]][1], index_rec[p]));
			//priority_queue先排序第一个参数，在排序第二个参数
			//因此，存储时，要将执行时间放在第一个参数，不能将index放在第一个
			++p;
		}

		auto [ length, index] = que.top();
		ans.emplace_back(index);
		time_rec += length;
		que.pop();

	}
	return ans;

}


//363 矩形区域不超过 K 的最大数值和
int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
	//先拿二维矩阵前缀和暴力
	if (matrix.empty()) {
		return 0;
	}
	int m = matrix.size();
	int n = matrix[0].size();
	vector<vector<int>> matrix_presum(m + 1, vector<int>(n + 1));

	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			matrix_presum[i][j] = matrix_presum[i - 1][j] + matrix_presum[i][j - 1] - matrix_presum[i - 1][j - 1] + matrix[i - 1][j - 1];
		}
	}

	long ans = INT_MIN;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) { //左上角
			for (int a = i + 1; a <= m; a++) {
				for (int b = j + 1; b <= n; b++) { //右下角
					long sumup = matrix_presum[a][b] + matrix_presum[i][j] - matrix_presum[a][j] - matrix_presum[i][b];
					if (sumup < k) {
						ans = max(ans, sumup);
					}
				}
			}
		}
	}
	return ans;
}


//368,最大整除子集
vector<int> largestDivisibleSubset(vector<int>& nums) {
	int n = nums.size();
	sort(nums.begin(), nums.end());
	vector<int> dp(n + 1, 1);
	//dp[i]表示[0-i]最长子序列个数
	int max_length = 1;
	int max_index = 0;
	int max_value = 0;

	for (int i = 1; i < n; ++i) {
		for (int j = 0; j < i;++j) {
			if (nums[i] % nums[j] == 0) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
		if (dp[i] > max_length) {
			//保存下标、长度、最大值很重要
			max_index = i;
			max_value = nums[i];
			max_length = dp[i];
		}
	}

	vector<int> ans;
	for (int j = max_index; j >= 0; --j) {   //倒序查找
		if (max_value % nums[j] == 0&&dp[j]==max_length) {
			ans.emplace_back(nums[j]);
			max_value = nums[j];
			max_length--;
		}
	}
	return ans;

}




//1011. 在 D 天内送达包裹的能力
// 找出满足D天内送达的最小船负载能力
//包裹的顺序不能改变
int shipWithinDays(vector<int>& weights, int D) {
	//船的负载范围在  (max(单个货物重量)) ~ (所有货物重量之和)  之间
	int lower_bound = *max_element(weights.begin(), weights.end());
	int upper_bound = accumulate(weights.begin(), weights.end(), 0);
	//二分查找在此区间验证重量是否满足
	
	int ans = 1;
	while (lower_bound <= upper_bound) {
		int sum = 0;
		int count = 1;
		int mid = (lower_bound + upper_bound) / 2;
		for (auto weight : weights) {
			sum += weight;
			if (sum > mid) {
				count++;
				sum = weight;
			}
		}
		if (count <= D) { //船大了,时间比预计时间短,船也有可能能继续缩小，同时count满足D
			upper_bound = mid - 1;
		}
		else { //船小了，count>D,规定时间装不完
			lower_bound = mid + 1;
		}
	}
	return lower_bound;
}


//判断是否存在a^2+b^2 = c
bool judgeSquareSum(int c) {
	for (long a = 0; a * a <= c; ++a) {
		double b = sqrt(c - a * a);
		if (b == int(b)) {
			return true;
		}
	}
	return false;
}




int deleteAndEarn(vector<int>& nums) {
	int n = nums.size();
	int max_ele = 0;
	for (int i = 0; i < n; ++i) {
		max_ele = max(max_ele, nums[i]);
	}

	vector<int> all(max_ele + 1);

	for (auto a : nums) {
		all[a] += a;
	}
	vector<int> dp(max_ele + 1);
	dp[0] = 0;
	dp[1] = all[1];

	for (int i = 2; i <= max_ele; ++i) {
		dp[i] = max(dp[i - 1], dp[i - 2] + all[i]);
	}
	cout << dp[max_ele] << endl;
	return dp[max_ele];
}


//137 只出现一次的数字II
//给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素
int singleNumber(vector<int>& nums) {
	//找到一个方法，使得出现三次的数字位操作后变为0
	//将二进制中的count方法延展到32位int,二进制中出现一次的方法
	int one = 0;
	int two = 0;
	for (auto num : nums) {
		one = (num ^ one) & (~two);
		two = (num ^ two) & (~one);
	}
	return one;   // 出现一次的数字。  one位置为1， 最后记录结果只用返回one
}

//case 2
int singleNumber2(vector<int>& nums) {
	//遍历，统计每一位二进制的数量
	int ans = 0;
	for (int i = 0; i < 32; ++i) {
		int index_count = 0;
		for (auto num : nums) {
			index_count += (num >> i) & 1; //取最低位
		}
		if (index_count % 3 != 0) {
			ans |= (1 << i); //构造ans
		}
	}
	return ans;
}


// 260. 只出现一次的数字 III
/*
* 一个整数数组 nums，其中恰好有两个元素只出现一次,其余所有元素均出现两次.
* 找出只出现一次的那两个元素
*/
vector<int> singleNumberIIcase1(vector<int>& nums) {
	//先将所有的数字进行异或，可以得到出现一次的两个元素的异或值
	int num_xor = 0;
	for (auto& a : nums) {
		num_xor ^= a;
	}
	// 对于num_xor , 其二进制为1的位置，说明两个数对应位置不同， 为0的位置，说明相同
	// 取出num_xor的二进制中最低位1（比较好取）
	num_xor = (num_xor == INT_MIN ? num_xor : num_xor & (-num_xor));
	// -num_xor 的最低位1不变，其他位取反
	// INT_MAX不改变，是因为，INT_MAX的最高位为1， 其余位为0，最低位1就是其本身
	int case1 = 0, case2 = 0;
	for (auto& a : nums) {
		if (a & num_xor) {  //最低位也为1的那一类数  
			case1 ^= a;
		}
		else {  //最低位为0的那一类数  
			case2 ^= a;
		}
	}
	vector<int> ans{ case1,case2 };
	return ans;
}


//554 砖墙
//使得直线穿过的砖块数量最少
int leastBricks(vector<vector<int>>& wall) {
	unordered_map<int, int> un;
	int m = wall.size();
	for (int i = 0; i < m; ++i) {
		int sum = 0;
		int n = wall[i].size();   //每行的砖块数量不一样
		for (int j = 0; j < n - 1; ++j) {
			sum += wall[i][j];
			un[sum]++;
		}
	}
	int count = 0;
	for (auto [_, cnt] : un) {
		count = max(count, cnt);
	}
	cout << m - count << endl;
	return m - count;
}



//1723 完成所有工作所需要的最短时间
class Jobs {
private:
	int ans = 0x3f3f3f3f;
	int n = 0;
	int k_ = 0;
	vector<int> sums;
public:
	void dfs(int index,vector<int> sums, vector<int>& jobs,int max_ele) {
		if (max_ele >= ans) { //剪枝
			return;
		}
		if (index == n) { //工作完了
			ans = max_ele;
			return;
		}
		for (int i = 0; i < k_; ++i) { //遍历人
			sums[i] += jobs[index]; //分配了当前工作
			dfs(index + 1, sums, jobs, max(sums[i], max_ele));
			sums[i] -= jobs[index];
		}
	}

	//优化防止超时
	//改变了分配次序，优先分配给空闲的工人
	void dfs2(vector<int>& jobs, vector<int>& nums, int job_index, int worker_count, int max_ele) {
		//job_index表示当前的工作编号
		//worker_count表示当前用了多少个工人
		if (max_ele >= ans) {
			return;
		}
		if (job_index == n) {
			ans = max_ele;
			return;
		}

		//优先分配空闲工人
		if (worker_count < k_) {
			sums[worker_count] = jobs[job_index];
			dfs2(jobs, nums, job_index + 1, worker_count + 1, max(max_ele, sums[worker_count]));
			sums[worker_count] = 0;
		}

		//空闲工人分配完毕了
		//没有空闲工人了，从所有工人中正常dfs
		for (int i = 0; i < worker_count; ++i) {
			sums[i] += jobs[job_index];
			dfs2(jobs, nums, job_index + 1, worker_count, max(max_ele, sums[i]));
			sums[i] -= jobs[job_index];
		}

	}

	int minimumTimeRequired(vector<int>& jobs, int k) {
		if (k == jobs.size()) {
			return *max_element(jobs.begin(), jobs.end());
		}
		n = jobs.size();
		k_ = k;
		int job_index = 0;
		sums.resize(k);
		//dfs(0, sums, jobs,0);
		dfs2(jobs, sums, 0, 0, 0);
		cout << ans << endl;
		return ans;
	}
};

//403 青蛙过河
class Frog {
private:
	unordered_set<int> stone_rec;
	vector<unordered_map<int, int>> mem;
	int n;
public:
	//普通的dfs，会超时
	void dfs(vector<int>& stones,int index, int last_step, bool& tag) { //要记录上次跳了多少步,index为当前位置
		if (index == n) {
			tag = true;
			return;
		}
		//上次为last_step,这次为last_step-1 ~ last_step+1
		//要保证大于0
		for (int this_step = last_step - 1; this_step <= last_step + 1 && this_step<=n; ++this_step) {
			if (this_step > 0) {
				//有石头则继续
				int trans = index + this_step;
				if (stone_rec.count(trans)) {
					dfs(stones, trans, this_step,tag);
				}
			}
		}
		
	}

	//记忆化搜索
	//mem[stone][step] 记录在当前石头，上次走了step步时，能否到达下一块石头,step不要当数组，不然会越界,当值存入unordered_map
	//mem中存入stone的下标，存值会越界
	//vector<unordered_map<int,int>> mem
	//记忆化搜索每次更新的就是mem，通过mem判断是否能到达
	bool dfs2(vector<int>& stones, vector<unordered_map<int, int>>& mem, int index, int last_step) {
		if (index == n-1) { //走到了最后一块石头
			cout << "done" << endl;
			return true;
		}
		//index不会是没有石头的位置，下面if除去了这种可能
		if (mem[index].count(last_step)) { //当前位置走得通
			//有
			return mem[index][last_step];
		}

		for (int this_step = last_step - 1; this_step <= last_step + 1; ++this_step) {
			if (this_step > 0) { //走几步
				//按石头来找，不按照每次的步数来找
				int find_stone = lower_bound(stones.begin(), stones.end(), stones[index] + this_step) - stones.begin();
				if (find_stone < n && stone_rec.count(stones[index] + this_step) && dfs2(stones, mem, find_stone, this_step)) {
					return mem[index][last_step] = true;
				}
			}
		}
		return mem[index][last_step] = false;
	}


	bool canCross(vector<int>& stones) {
		//n = *max_element(stones.begin(),stones.end());
		stone_rec = unordered_set<int>(stones.begin(), stones.end()); //记录了石头的位置
		//bool tag = false;
		//dfs(stones, 0, 0, tag);
		//return tag;

		//若n记录的是所有石头数目，当用例为
		//[0,2147483647]  会越界
		//n用来只记录石头位置，不采用所有数组
		
		n = stones.size();
		mem.resize(n);
		return dfs2(stones, mem, 0, 0);
	}
};


//7 整数反转
// 123=》321   -123=》-321
// 不使用64位整数，即long
int reverse(int x) {
	//-2^31 = -2147483648 =>-147483648
	//2^31-1 = 2147483647 =>147483647
	return 0;
}


//1482 制作m束花需要的最少天数
//每束花需要使用相邻的k朵花，bloomDay[i]表示第i多花的开放时间，返回m束需要的最少天数，无法制作返回-1
class Flowers {
public:
	bool can_make(vector<int>& bloomDay, int m, int k, int mid) { //在mid情况下，看有多少满足的
		int done_flowers = 0; //已经做的花束
		int flower_count = 0; //记录当前花束已经用的花朵
		for (int i = 0; i < bloomDay.size(); ++i) {
			if (bloomDay[i] <= mid) { //可以开放
				flower_count++;
				if (flower_count == k) {
					done_flowers++; //完成数加1
					flower_count = 0; //为下次做准备
				}
			}
			else { //开不了，那么把之前已经计算的花朵清0
				flower_count = 0;
			}
		}
		cout<< (done_flowers >= m);
		return done_flowers >= m;
	}

	int minDays(vector<int>& bloomDay, int m, int k) {
		if (bloomDay.size() < m * k) {
		return -1;
		}
		//题目可理解为在长度为n的数组中，找出m个长度为k且不相交的区间，找出区间中所有元素的最大值中的最小值
		//看到最大值的最小值，应想到二分
		int max_day = *max_element(bloomDay.begin(), bloomDay.end());
		int min_day = *min_element(bloomDay.begin(), bloomDay.end());
	
		while (min_day <= max_day) {
			int mid = (min_day + max_day) / 2;
			if (can_make(bloomDay, m, k, mid)) { //当前值可以完成,那么再把它缩小一些
				max_day = mid;
			}
			else {//不能完成，那增大一点
				min_day = mid+1;
			}
		}
		return min_day;
	}

};





//5751. 下标对 中的最大距离
int maxDistance(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size();
	int n = nums2.size();
	if (nums1[m - 1] > nums2[0]) { //最小大于最大，nums1[i]<nums2[j]不可能成立
		return 0;
	}
	int ans = 0;
	int i = 0;
	for (int j = 0; j < n; ++j) {
		while (nums1[i] > nums2[j] && i < m) {
			i++; //找到满足nums1[i]>nums2[j]的第一个数
		}
		if (i < m) {
			//不能倒序求解，倒序求解的话，i初始在m-1，j初始在n-1.若此时也有m>n
			/*while(i>=0&&nums1[i]<nums2[j]){
				i--;
			}*/
			//若nums1[m-1] == nums2[0],nums[m-2]<nums[0]...
			//此时while没执行，返回的是i-j，并非0
			ans = max(ans, j - i);
		}
	}
	
	return ans;
}




//5750. 人口最多的年份
int maximumPopulation(vector<vector<int>>& logs) {
	vector<int> count(101);
	for (auto a : logs) {
		for (int i = a[0]; i < a[1];++i) {
			count[i-1950]++; //把活着的那些年全部标记
		}
	}

	int ans = 0;
	for (int i = 0; i < 101;++i) {
		if (count[i] > count[ans]) {
			ans = i;
		}
	}
	return ans+1950;
}

//1856. 子数组最小乘积的最大值
/*
数组 [3,2,5] （最小值是 2）的最小乘积为 2 * (3+2+5) = 2 * 10 = 20 。
给你一个正整数数组 nums ，请你返回 nums 任意 非空子数组 的最小乘积 的 最大值
*/
int maxSumMinProduct(vector<int>& nums) {
	//当前子数组的最小值没有变时，子数组长度越长，最小乘积越大
	//倘若我存储当前位置左右两侧比它小的数的位置时候
	//[2,3,3,1,2]可以得到 =>>>rec = [-1,3],[0,3],[0,3],[-1,-1],[3,-1]
	//得到前缀和为=>>>[2,5,8,9,11]
	//最小乘积:  0-2 2*8=16  3 1*11=11 

	nums.emplace_back(0); //头尾加0便于后续处理
	nums.insert(nums.begin(),0);
	int n = nums.size();
	//存储前缀和
	vector<int> pre_sum(n + 1);
	for (int i = 1; i <= n; ++i) {
		pre_sum[i] = pre_sum[i - 1] + nums[i - 1];
	}
	vector<pair<int, int>> smaller_ele(n); //存左右侧比当前位置小的下标
	//拿栈+循环找下标
	stack<int> st; //存储下标
	

	for (int i = 0; i < n; ++i) {
		while (!st.empty() && nums[i] < nums[st.top()]) {
			smaller_ele[st.top()].second = i;
			st.pop(); //存储过最小值的下标全部弹出
			
		}
		st.push(i);
	}
	
	stack<int> st2; //找左侧第一个比它小的
	for (int j = n-1; j >=0 ; --j) {
		while (!st2.empty() && nums[st2.top()] > nums[j]) {
			smaller_ele[st2.top()].first = j;
			st2.pop();
		}
		st2.push(j);
	}
	
	int ans = 0;
	for (int i = 1; i < n; ++i) {
		int l = smaller_ele[i].first;
		int r = smaller_ele[i].second; //左右的最小值都不计算(l,r)中的值
		ans = max(ans, (pre_sum[r] - pre_sum[l + 1]) * nums[i]);
	}
	cout << ans << endl;
	return ans;
}



//1734.解码异或后的排列
vector<int> decode(vector<int>& encoded) {
	//前n个正整数的排列，n为奇数
	//观察可知，encoded的所有数的异或值等于perms[0]^perms[end]
	//可以得到 perms[0]^encoded[所有下标位奇数] = encoded[所有下标偶数]^perms[end] = perms[all]^
	/*
	perms是前n个正整数的排列，因此，perms[all]^的值 为1~n的异或值
	*/
	int n = encoded.size();
	int perms_all = 0;
	for (int i = 1; i <= n + 1; ++i) {
		perms_all ^= i;
	}
	int p_0 = 0; //perms[0]
	int encoded_allodd_xor = 0;
	for (int i = 0; i < n; ++i) {
		if (i % 2 != 0) { //odd
			encoded_allodd_xor ^= encoded[i];
		}
		//xor_all^=encoded[i];
	}
	p_0 = perms_all ^ encoded_allodd_xor;
	vector<int> perms(n + 1);
	perms[0] = p_0;
	for (int i = 0; i < n; i++) {
		perms[i + 1] = perms[i] ^ encoded[i];
	}
	return perms;
}



//87 扰乱字符串 hard
class scramble {
private:
	vector<vector<vector<int>>> mem;
	int n;
	string s1, s2;
public:
	//先写爆搜
	bool parameter_same(string s1, string s2) { //判断二者所含的各种字母的数量是否相同
		vector<int> str(26);
		for (auto a : s1) {
			str[a - 'a']++;
		}

		for (auto b : s2) {
			str[b - 'a']--;
		}
		if (any_of(str.begin(), str.end(), [](const auto& a) {
			return a != 0;  //存在不等于0，说明不完全相同
		})) {
			return false;
		}
		return true;
	}

	//递归的边界条件，字符串相等了， 字符串元素不相等
	bool isScramble(string s1, string s2) {
		if (s1 == s2) {
			return true;
		}
		if (!parameter_same(s1, s2)) {
			return false;
		}
		int n = s1.size();

		for (int i = 1; i < s1.size(); ++i) {  //[0~i) [i~n),开始划分
			string a = s1.substr(0, i), b = s1.substr(i);
			string c = s2.substr(0, i), d = s2.substr(i);
			//先计算不交叉的
			if (isScramble(a, c) && isScramble(b, d)) {
				return true;
			}

			//若交叉了,s1的前i个字符实际上是s2的后i个字符
			string c1 = s2.substr(s2.size() - i);//取s2的后s2.size()-i个字符
			string d1 = s2.substr(0, s2.size() - i); //取s2的前s2.size()-i个字符
			if (isScramble(a, c1) && isScramble(b, d1)) {
				return true;
			}
		}
		return false;
	}

	//使用记忆化搜索
	//mem[i][j][k]表示s1在位置i，s2在位置j，长度为k时是否匹配
	//mem 0表示没访问过，-1表示不成立，1表示成立
	bool dfs(int s1_index, int s2_index, int len) { //更新mem和使用mem判断
		if (mem[s1_index][s2_index][len] != 0) {
			return mem[s1_index][s2_index][len] == 1; //当前位置走过了，直接返回mem所记录的值
		}
		
		string a = s1.substr(s1_index, len);
		string b = s2.substr(s2_index, len); //取长度为len的子串
		
		if (a == b) {
			mem[s1_index][s2_index][len] = 1;
			return true;
		}
		if (!parameter_same(a, b)) { //false
			mem[s1_index][s2_index][len] = -1;
			return false;
		}

		for (int i = 1; i < len; i++) {
			if (dfs(s1_index, s2_index, i) && dfs( s1_index + i, s2_index + i, len - i)) { //不交换
				mem[s1_index][s2_index][len] = 1; //不交换走的通
				return true;
			}
			//交换
			//len+s2_index 表示从s2_index开始取len长的子串
			//len + s2_index - i表示取len长的子串的后i位，即交换了
			if (dfs(s1_index, s2_index + len - i, i) && dfs(s1_index + i, s2_index, len - i)) {//前部和后部
				mem[s1_index][s2_index][len] = 1;
				return true; //交换后走的通
			}
		}
		mem[s1_index][s2_index][len] = -1;
		return false;
	}

	bool isScramble2(string _s1, string _s2) {
		s1 = _s1, s2 = _s2;
		if (s1 == s2) {
			return true;
		}
		if (s1.size() != s2.size()) {
			return false;
		}
		n = s1.size();
		mem.resize(n, vector<vector<int>>(n, vector<int>(n+1, 0)));
		return dfs(0, 0, n);
	}

};




//组合总数
class Combination_Sum {
private:
	vector<pair<int, int>> frequency;
	vector<vector<int>> comb2_ans;
	vector<int> comb2_temp;

	vector<int> temp3;
	vector<vector<int>> ans3;

	vector<int> mem4;
	int temp4;
	int n4;
	int n4_2;

public:
	void dfs1(vector<int>& candidates, int target, vector<vector<int>>& ans, vector<int>& temp_str, int index) {
		if (index == candidates.size()) {
			return;
		}
		if (target == 0) {
			ans.emplace_back(temp_str);
			return;
		}
		//不选择当前位置
		dfs1(candidates, target, ans, temp_str, index + 1);
		if (target - candidates[index] >= 0) { //当前位置可以选择
			temp_str.emplace_back(candidates[index]); //加入
			dfs1(candidates, target - candidates[index], ans, temp_str, index); //因为每一位可以重复选取，所以此处的index可以直接写index
			temp_str.pop_back(); //这路走完了
		}
	}

	/*
	* 组合总和I  39
	* 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
	  candidates 中的数字可以无限制重复被选取
	*/
	//递归+回溯
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		vector<vector<int>> ans;
		vector<int> temp_str;
		dfs1(candidates, target, ans, temp_str, 0);
		return ans;
	}


	/*
	* 组合总和II  40
	* 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
	  candidates 中的每个数字在每个组合中只能使用一次。
	* candidates包含了重复元素，[1,1,2]与[2,1,1]、[1,2,1]这些为同一种组合
	*/


	//因为有重复元素，使用递归+回溯，会造成结果重复，因此可以提前排序，然后相同的数字统一处理，for(..;i<frequency;..)来控制选择重复元素的个数
	//将包含重复元素的candidates转换为包含不重复数字和其出现频率的frequency

	//case 2 : 或者类似于组合总数I一样处理，最后set去重
	void dfs2(vector<pair<int, int>>& frequency, int target, int index) {
		if (target == 0) {
			comb2_ans.emplace_back(comb2_temp);
			return;
		}
		if (index == frequency.size()) {
			return;
		}

		//不选当前位置
		dfs2(frequency, target, index + 1);

		//选当前位置，考虑一下重复元素一起递归
		int max_time = min(target / frequency[index].first, frequency[index].second); //当前数字最多的选取次数
		for (int j = 1; j <= max_time; ++j) {
			comb2_temp.emplace_back(frequency[index].first);
			dfs2(frequency, target - frequency[index].first * j, index + 1);
			//选取1~most个重复元素，逐次便利
		}
		for (int k = 1; k <= max_time; ++k) {
			//所有的重复元素都执行完了,最后一次一共塞入了max_time个元素，删除掉他们
			comb2_temp.pop_back();
		}
	}

	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
		sort(candidates.begin(), candidates.end());
		int n = candidates.size();
		for (int i = 0; i < n; ++i) {
			if (frequency.empty() || frequency.back().first != candidates[i]) {
				frequency.emplace_back(candidates[i], 1);
			}
			else {
				frequency.back().second++; //重复
			}
		}
		dfs2(frequency, target, 0);
		return comb2_ans;
	}

	/*
	* 组合总和III  216
	* 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
		说明：
			 所有数字都是正整数。
			 解集不能包含重复的组合。

	*/

	void dfs3(int k, int n, int num) {
		if (n == 0 && k == 0) {
			ans3.emplace_back(temp3);
			return;
		}
		if (num > 9 || k == 0) {
			return;
		}
		temp3.emplace_back(num);
		dfs3(k - 1, n - num, num + 1);
		temp3.pop_back();
		dfs3(k, n, num + 1);
	}
	vector<vector<int>> combinationSum3(int k, int n) {
		dfs3(k, n, 1);
		return ans3;
	}


	/*
	* 组合总和IV  377
	* 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
	* 注意：
		   顺序不同的序列被视作不同的组合
	*/

	void dfs4(vector<int> nums, int target, int& ans) {
		if (target == 0) {
			ans++;
			return;
		}
		if (target < 0) {
			return;
		}
		for (auto num : nums) {
			dfs4(nums, target - num, ans);
		}
	}

	int combinationSum4(vector<int>& nums, int target) {
		n4 = nums.size();
		temp4 = 0;
		dfs4(nums, target,temp4);
		return temp4;
	}   //暴力会超时


	//记忆化搜索
	//mem[value]表示target为value时有几个方案数
	int mem_search(vector<int>& nums, int target) {
		if (mem4[target] != -1) {
			return mem4[target]; //走过了，直接返回数目
		}
		//没走过
		int res = 0;
		for (auto num : nums) {
			if (target - num >= 0) {
				res += mem_search(nums, target - num);
			}
		}
		mem4[target] = res;
	}

	int combinationSum4_2(vector<int>& nums, int target) {
		n4_2 = nums.size();
		mem4.resize(target + 1, -1);
		mem_search(nums, target);
		mem4[0] = 1;
		return mem4[target];
	}


	//dp
	int combinationSum4_3(vector<int>& nums, int target) {
		int ans = 0;
		vector<vector<unsigned long long>> dp(target + 1, vector<unsigned long long> (target+1));
		dp[0][0] = 1;
		for (int i = 1; i <= target; ++i) { //选择多少个,因为最小数为1，因此最大长度为taregt
			for (int j = 1; j <= target; ++j) { //当前值
				for (auto num : nums) { //从头开始取
					if (j - num >= 0) {
						dp[i][j] += dp[i - 1][j - num];
					}
				}
			}
			ans += dp[i][target];
		}
		return ans;
	}
};



//12. 整数转罗马数字
//给你一个整数，将其转为罗马数字。
string intToRoman(int num) {
	vector<int> nums = { 1,4,5,9,10,40,50,90,100,400,500,900,1000 };
	vector<string> rep = { "I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M" };

	string ans("");
	for (int i = 12; i >= 0; --i) {
		while (num >= nums[i]) {
			ans += rep[i];
			num -= nums[i];
		}
	}
	return ans;
}



//8. 字符串转换整数 (atoi)
int myAtoi(string s) {
	auto a = s.find_first_not_of(' ');
	int pre_zero = 1;
	int pos = 1;
	long long ans = 0;
	if (s[a] == '-') {
		pos = -1;
		a++;
	}
	else if (s[a] == '+') {
		pos = 1;
		a++;
	}
		
	for (int i = a; i < s.size(); ++i) {
		if (s[i] == '0' && pre_zero) {

			continue;
		}
		if (s[i] != 0) {
			if (s[i] < '0' && s[i]>9) {
				break;
			}
			else {
				ans = ans * 10 + s[i] - '0';
				pre_zero = 0;
			}
		}
	}
	cout << pos*ans << endl;
	return ans;
}

//1827. 最少操作使数组递增
int minOperations(vector<int>& nums) {
	int opt = 0;
	for (int i = 1; i < nums.size(); ++i) {
		if (nums[i] <= nums[i - 1]) {
			opt += nums[i - 1] + 1 - nums[i];
			nums[i] = nums[i - 1] + 1;
		}
	}
	cout << opt << endl;
	return opt;
}


//455. 分发饼干
int findContentChildren(vector<int>& g, vector<int>& s) {
	sort(g.begin(), g.end());
	sort(s.begin(), s.end());
	int sum = 0;
	int g_index = 0, s_index = 0;
	while (g_index < g.size() && s_index < s.size()) {
		if (s[s_index] >= g[g_index]) {
			g_index++;
			//sum++;
		}
		s_index++;
	}
	cout << g_index << endl;
	return g_index;
}

//52双周赛 1
string sortSentence(string s) {
	string temp("");
	vector<string> str(10);
	for (int i = 0, pre_index = 0; i < s.size(); ++i) {
		if ('0' <= s[i] && s[i] <= '9') {
			temp = s.substr(pre_index, i - pre_index);
			str[s[i] - '0'] = temp;
			pre_index = i + 2;
			temp = "";
		}
	}

	string ans("");
	for (auto a : str) {
		if (a != "") {
			ans += a;
			ans += " ";
		}
	}
	ans.erase(ans.size() - 1);
	return ans;
}




vector<vector<string>> rotateTheBox(vector<vector<string>>& box) {
	int m = box.size(); //row
	int n = box[0].size(); //col
	for (int i = 0; i < m; ++i) {
		int empty_count = 0;
		for (int j = n - 1; j >= 0; --j) {
			if (box[i][j] == ".") {
				empty_count++;
			}
			else if (box[i][j] == "*") {
				empty_count = 0;
			}
			else {
				string temp = box[i][j + empty_count];
				box[i][j + empty_count] = box[i][j];
				box[i][j] = temp;
			}
		}
	}
	return box;
}


//5759. 找出所有子集的异或总和再求和
void dfssubset(vector<int> nums, int& ans,int temp,int index) {
	if (index == nums.size()) {
		ans += temp;
		return;
	}
	dfssubset(nums, ans, temp, index+1);
	dfssubset(nums, ans, temp^= nums[index],index+1);

}

//使用二进制表示法，遍历所有情况
int subsetXORSum(vector<int>& nums) {
	//首先考虑所有的子集数量，用n位长的二进制数来表示nums的每一位取或不取
	//如1010表示取第一位和第三位
	//爆搜

	int ans = 0;
	int n = nums.size();
	int total_sub = 1 << n;
	for (int i = 0; i < total_sub; i++) { //i穷举所有情况
		int temp = 0;
		for (int j = 0; j < n; ++j) { //j取位
			if ((1 & (i >> j)) != 0) { //取出二进制i的末位，若这位是1，则挑选出nums[j]
				temp ^= nums[j];
			}
		}
		ans += temp;
	}
	return ans;

	////dfs爆搜
	//int ans = 0;
	//dfssubset(nums, ans, 0, 0);
	//cout << ans << endl;
	//return ans;
}



//5760. 构成交替字符串需要的最小交换次数
int minSwaps(string s) {
	/*int count_1 = 0;
	int count_0 = 0;
	int ans = 0;
	for (auto a : s) {
		if (a == '0') {
			count_0++;
		}
		else {
			count_1++;
		}
	}
	if (abs(count_1-count_0) > 1) {
		return -1;
	}

	if (count_1 == count_0) {
		int rep1 = 0;
		for (int i = 0; i < s.size(); i+=2) {
			if (s[i] != '1')
				rep1++;
		}
		int rep0 = 0;
		for (int i = 1; i < s.size(); i += 2) {
			if (s[i] != '0')
				rep0++;
		}
		ans = min(rep1, rep0);
	}
	else if (count_1 > count_0) {
		for (int i = 0; i < s.size(); i += 2) {
			if (s[i] != '1')
				ans++;
		}
	}
	else {
		for (int i = 0; i < s.size(); i += 2) {
			if (s[i] != '0')
				ans++;
		}
	}
	cout << ans << endl;
	return ans;*/


	//第一步：统计1和0的个数
	int count_1 = count(s.begin(), s.end(), '1'); // algorithm count
	int count_0 = count(s.begin(), s.end(), '0');
	//count_if(a.begin(),a.end(),pred)
	/*
	bool IsOdd (int i) { return ((i%2)==1); }
	int mycount = count_if (myvector.begin(), myvector.end(), IsOdd);
	*/

	//s的形式只能是1010...   0101...
	int n = s.size();
	int ans = INT_MAX;
	int c1 = 0;

	int c2 = 0;
	if (count_1 == (n + 1) / 2 && count_0 == (n / 2)) { //满足条件说明1的个数和0相等或者比0多一个,此时形式为1010...
		for (int i = 0; i < n; i += 2) {
			if (s[i] != '1') {
				c1++;
			}
		}
		ans = min(ans, c1);
	}


	//0101...
	
	if (count_0 == (n + 1) / 2 && count_1 == n / 2) { //当0和1的个数相同或者0比1多一个时，满足条件
		for (int i = 0; i < n; i += 2) {
			if (s[i] != '0') {
				c2++;
			}
		}
		ans = min(ans, c2);
	}
	if (ans == INT_MAX) {
		return -1;
	}
	else {
		return ans;
	}
}



//5761. 找出和为指定值的下标对
//FindSumPairs(int[] nums1, int[] nums2) 使用整数数组 nums1 和 nums2 初始化 FindSumPairs 对象。
//void add(int index, int val) 将 val 加到 nums2[index] 上，即，执行 nums2[index] += val 。
//int count(int tot) 返回满足 nums1[i] + nums2[j] == tot 的下标对(i, j) 数目。
class FindSumPairs {
private:
	vector<int> _nums1;
	vector<int> _nums2;
	unordered_map<int, int> mp1, mp2;
public:
	FindSumPairs(vector<int>& nums1, vector<int>& nums2) {
		_nums1 = nums1;
		_nums2 = nums2;
		for (auto a : _nums1) {
			mp1[a]++;
		}
		for (auto b : _nums2) {
			mp2[b]++; //统计每种数字的个数
		}
	}

	void add(int index, int val) {
		mp2[_nums2[index]]--;
		_nums2[index] += val;
		mp2[_nums2[index] + val]++;
	}

	int count(int tot) {
		int ans = 0;
		for (auto a : _nums1) {
			int gap = tot - a;
			if (mp2.count(gap)) {
				ans += mp2[gap];
			}
		}
		return ans;
	}
};

//5762. 恰有 K 根木棍可以看到的排列数目
//有 n 根长度互不相同的木棍，长度为从 1 到 n 的整数。
//请你将这些木棍排成一排，并满足从左侧 可以看到 恰好 k 根木棍。从左侧 可以看到 木棍的前提是这个木棍的 左侧 不存在比它 更长的 木棍。

int rearrangeSticks(int n, int k) {
	int MOD = 1e9 + 7;
	vector<vector<long long>> dp(n + 1,vector<long long>(k+1));
	//dp[i][j]表示i根木棍里，可以看到j根木棍
	//最后一根再放最短的
	dp[1][1] = 1;
	for (int i = 2; i <= n; ++i) {
		for (int j = 1; j <= k; ++j) {
			dp[i][j] = (dp[i - 1][j] * (i-1) + dp[i - 1][j - 1] * 1)%MOD;
			//前i-1根木头已经看到了j根，我最后一根是最短的，那么放在任何一个木头之后都看不到我dp[i-1][j]*(i-1)
			//前i-1根木头已经看到了j-1根，差最后一根，我最后一根最短，那么只能放在第一根，为dp[i-1][j-1]*1
		}
	}
	cout << dp[n][k];
	return dp[n][k];
}


//1442. 形成两个异或相等数组的三元组数目
int countTriplets(vector<int>& arr) {
	int n = arr.size();
	vector<int> pre_xor(n + 1);
	for (int i = 1; i <= n; ++i) {
		pre_xor[i] = pre_xor[i - 1] ^ arr[i - 1];
	}
	int ans = 0;
	int temp1 = 0, temp2 = 0;
	for (int i = 1; i < n; i++) {
		for (int j = i + 1; j <= n; j++) {
			for (int k = i + 1; k <= j; ++k) {
				if (pre_xor[i - 1] == pre_xor[j]) {
					ans++;
				}
			}
		}
	}
	return ans;
}

//1738. 找出第 K 大的异或坐标值
int kthLargestValue(vector<vector<int>>& matrix, int k) {
	//前缀和
	int m = matrix.size();
	int n = matrix[0].size();
	vector<vector<int>> pre_xor(m + 1, vector<int>(n + 1));

	vector<int> ans;
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			pre_xor[i][j] = matrix[i - 1][j - 1] ^ pre_xor[i - 1][j] ^ pre_xor[i][j - 1] ^ pre_xor[i - 1][j - 1];
			ans.push_back(pre_xor[i][j]);
		}
	}

	sort(ans.begin(), ans.end(), greater<int>());
	return ans[k - 1];
}


//快速选择算法
// 第K大的数
class quickSelect {
public:
	int findKthLargest(vector<int>& nums, int k) {
		int n = nums.size();
		return quickselect(nums, 0, n-1, n - k);
	}

	//quickselect判断下标
	//random 产生随即下标
	//partion分离，进行每次子排序
	int quickselect(vector<int>& nums, int low, int high, int k) {
		int r_index = random_index(low, high);
		swap(nums[r_index], nums[high]);
		int proper_index = partion(nums, low, high);
		if (proper_index == k) {
			cout << nums[proper_index];
			return nums[proper_index];
		}
		else if (proper_index < k) {
			quickselect(nums, proper_index + 1, high, k);
		}
		else {
			quickselect(nums, low, proper_index - 1, k);
		}

	}

	int random_index(int left, int right) {
		return rand() % (right - left + 1) + left; //generate index between left and right
	}
	
	int partion(vector<int>& nums, int left, int right) {
		int i = left - 1;
		for (int j = left; j < right; ++j) {
			if (nums[j] <= nums[right]) {
				swap(nums[++i], nums[j]); //很巧妙， 统计了有多少个比pivot（即提前交换过去的right）大的
			}
		}
		swap(nums[right], nums[i+1]);
		return i+1;
	}

};

//692. 前K个高频单词
vector<string> topKFrequent(vector<string>& words, int k) {
	unordered_map<string, int> mp;
	for (auto a : words) {
		mp[a]++;
	}
	

	auto cmp = [](const auto& a, const auto& b) {
		return a.second == b.second ? a.first<b.first : a.second>b.second; //小根堆，出现次数相同则字典序优先
	};
	priority_queue<pair<string, int>, vector<pair<string, int>>, decltype(cmp)> que(cmp);
	
	//将单词存入优先队列
	for (auto& a : mp) {
		que.push(a);
		if (que.size() > k) {
			que.pop();
		}
	}

	vector<string> ans;
	for (int i = 0; i < k;++i) {
		ans.emplace_back(que.top().first);
		que.pop();
	}
	return ans; 

}


//1035. 不相交的线
//1143. 最长公共子序列的变种题
int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size();
	int n = nums2.size();
	vector<vector<int>> dp(m+1, vector<int>(n+1));
	//dp[i][j]表示 nums1[0~i)，nums2[0~j)的最大连线长度
	dp[0][0] = 0;
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (nums1[i - 1] == nums2[j - 1]) { //相等了，可以选
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			else { //不相等，不选
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
	}
	cout << dp[m][n] << endl;
	return dp[m][n];
}


//15. 三数之和
//先确定a，再寻找b+c = -a， b从前往后，c从后往前寻找
vector<vector<int>> threeSum(vector<int>& nums) {
	sort(nums.begin(), nums.end()); //先排序，便于后续去重
	vector<vector<int>> ans;
	for (int i = 0; i < nums.size() - 1; ++i) {
		if (i > 0 && nums[i] == nums[i - 1]) { //防止重复元素
			continue;
		}
		int temp = nums[i];
		int right = nums.size() - 1; //right只用走一次
		for (int j = i + 1; j < nums.size(); ++j) {
			if (nums[j] == nums[j - 1]) {
				continue;
			}
			while (j < right && nums[j] + nums[right] + temp>0) {
				right--;
			}
			if (j == right) {
				break;
			}
			if (nums[j] + nums[right] + temp == 0) {
				ans.push_back({ nums[j],nums[right],temp });
			}
		}
	}
	return ans;
}



bool checkZeroOnes(string s) {
	int ele_0 = 0;
	int ele_1 = 0;
	int temp0 = 0, temp1 = 0;
	int i = 0, j = 0;
	while (j < s.size()) {
		if (s[j] == s[i]) {
			if (s[j] == '1') {
				
				j++;
				temp1++;
			}
			else {

				j++;
				temp0++;
			}
		}
		else {
			ele_0 = max(ele_0, temp0);
			ele_1 = max(ele_1, temp1);
			temp0 = 0;
			temp1 = 0;
			i = j;
			j = j + 1;
			if (s[i] == '1') {
				temp1 = 1;
			}
			else {
				temp0 = 1;
			}
		}
	}
	ele_0 = max(ele_0, temp0);
	ele_1 = max(ele_1, temp1);
	return ele_1 > ele_0 ? true : false;
}


//二分
bool can_arive(vector<int>& dist, double hour, int mid) {
	double hour_count = 0;
	for (int i = 0; i < dist.size() - 1; ++i) {
		int temp = 0;
		if (dist[i] % mid == 0) {
			temp = dist[i] / mid;
		}
		else {
			temp = dist[i] / mid + 1;
		}
		hour_count += (double)temp;
	}
	hour_count += dist[dist.size() - 1] / (double)mid;
	return hour_count <= hour;
}


//1870. 准时到达的列车最小时速
int minSpeedOnTime(vector<int>& dist, double hour) {
	int n = dist.size();
	if ((int)hour < n - 1) {
		return -1;
	}

	int min_v = 1;

	int max_v = *max_element(dist.begin(), dist.end())*100;

	while (min_v < max_v) {
		int mid = (min_v + max_v) / 2;
		if (can_arive(dist, hour, mid)) {
			max_v = mid;
		}
		else {
			min_v = mid + 1;
		}
	}
	return min_v;
}


//1871. 跳跃游戏 VII
bool canReach(string s, int minJump, int maxJump) {
	int n = s.size();
	vector<int> pre(n+1); //前缀和
	pre[1] = 1;
	vector<int> dp(n,0);
	dp[0] = 1;
	for (int i = 1; i < n; ++i) {
		if (s[i] == '0') {
			int left = max(0, i - maxJump);
			int right = i - minJump;
			if (right >= 0 && left <= right && pre[right + 1] - pre[left] > 0) {//前缀和差大于0，说明区间内有可以到达的石头
				dp[i] = 1; //能到达
			}
		}
		pre[i + 1] = pre[i] + dp[i]; //dp[i]表示可以到达此处
	}
	return dp[n - 1];
}



//810. 黑板异或游戏
//数学解法
bool xorGame(vector<int>& nums) {
	return false;

}

//1707. 与数组中元素的最大异或值
//vector<int> maximizeXor(vector<int>& nums, vector<vector<int>>& queries) {
//
//}


//664. 奇怪的打印机
int strangePrinter(string s) { //和最长回文子串的思想很相似
	int n = s.size();
	vector<vector<int>> dp(n, vector<int>(n));
	//dp[i][j]  表示从s[i]~s[j]的区间[i,j]最小打印次数
	
	for (int i = n - 1; i >= 0; --i) {
		dp[i][i] = 1;
		for (int j = i+1; j < n; ++j) {
			if (s[j] == s[i]) {
				dp[i][j] = dp[i][j - 1];  
			}
			else {
				//两端字母不相同时，需要分别打印左右两边的打印。
				//记两部分分别为区间 [i,k][i,k] 和区间 [k+1,j][k+1,j]（其中 i <= k < j）
				//dp[i][j] = min(dp[i][k]+dp[k+1][j])   k=[i,j) 
				int min_print = INT_MAX;
				for (int k = i; k < j; ++k) {
					min_print = min(min_print, dp[i][k] + dp[k + 1][j]);
				}
				dp[i][j] = min_print;
			}
		}
	}
	cout << dp[0][n - 1];
	return dp[0][n - 1];
}





string reverseParentheses(string s) {
	stack<char> st;
	string temp = "";
	for (auto a : s) {
		if (a == ')') {
			while (st.top() != '(') {
				temp += st.top();
				st.pop();
			}
			st.pop();
			for (int i = 0; i < temp.size();++i) {
				st.push(temp[i]);
			}
			temp = "";
		}
		else {
			st.push(a);
		}
	}
	string ans = "";
	while (!st.empty()) {
		ans.push_back(st.top());
		st.pop();
	}
	return ans;
}

//238. 除自身以外数组的乘积
vector<int> productExceptSelf(vector<int>& nums) {
	//不使用除法，输出数组中除了当前位置数的其他所有数的乘积
	int  n = nums.size();
	/*vector<int> left(n+1,1);
	vector<int> right(n+1,1);
	for (int i = 0; i < n; ++i) {
		left[i + 1] = left[i] * nums[i];
		right[n - i - 1] = right[n - i] * nums[n - i - 1];
	}
	vector<int> ans(n);
	for (int i = 0; i < n; ++i) {
		ans[i] = left[i] * right[i + 1];
	}
	return ans;*/
	vector<int> ans(n,1);
	int left = 1, right = 1;
	for (int i = 0; i < n; ++i) {
		ans[i] *= left;
		left *= nums[i];

		ans[n - i - 1] *= right;
		right *= nums[n - i - 1];
	}
	return ans;
}



//477. 汉明距离总和
//计算一个数组中，任意两个数之间汉明距离的总和。
int totalHammingDistance(vector<int>& nums) {
	if (nums.empty()) {
		return 0;
	}
	int n = nums.size();
	int ans = 0;
	for (int i = 0; i < 32; ++i) {
		int cnt[1][2];
		memset(cnt, 0, sizeof(cnt));
		for (auto num : nums) {
			int temp = num >> i;
			cnt[0][temp & 1]++;
		}
		ans += cnt[0][0] * cnt[0][1];
	}
	return ans;
}



//5774. 使用服务器处理任务
//vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
//	int n = servers.size(), m = tasks.size();
//	//记录服务器下标
//	vector<int> ser_index(n);  //将其按照权重大小排序后，记录每个位置对应的原先的下标
//	iota(ser_index.begin(), ser_index.end(), 0);
//	sort(ser_index.begin(), ser_index.end(), [&](const auto& a, const auto& b) {
//		return servers[b] > servers[a];
//	});
//	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> que;   //按照结束时间排序，结束时间相同则按照服务器权重排序
//	vector<int> ans(m);
//	vector<int> ser_rec(n, 0);
//	queue<int> delay;
//
//	for (int i = 0; i < m; ++i) {
//		while (!que.empty() && que.top().first <= i) {
//			ser_rec[ser_index[que.top().second]] = 0;
//			que.pop();
//		}
//		if (que.size() == n) {
//			delay.push(i);
//			continue;
//		}
//		
//		for (int j = 0; j < n;++j) {
//			if (ser_rec[ser_index[j]] == 0) {
//				if (delay.empty()) {
//					que.emplace(make_pair(i + tasks[i], j)); //按照权重从小到大取
//					ans[i] = ser_index[j];
//					ser_rec[ser_index[j]] = 1;
//					break;
//				}
//				else {
//					que.emplace(make_pair(delay.front() + tasks[i], j));
//					ans[delay.front()] = ser_index[j];
//					ser_rec[ser_index[j]] = 1;
//					delay.pop();
//					break;
//				}
//			}
//		}
//		
//	}
//	int time_rec = 0;
//	vector<pair<int,int>> delay_ser;
//	while (!delay.empty()) {
//		time_rec = max(time_rec, que.top().first); //目前队列中第一个任务的结束时间
//		delay_ser.push_back(que.top());
//		que.pop();
//		while (que.top().first == time_rec) {  //若多个服务器同时完成
//			delay_ser.push_back(que.top());
//			que.pop();
//		}
//		sort(delay_ser.begin(), delay_ser.end(), [](const auto& a, const auto& b) {
//			return a.second > b.second; //按照服务器优先级倒序排序
//		});
//		
//		int task_cnt = min(delay.size(), delay_ser.size());
//		while(task_cnt) {
//			ans[m-delay.size()]= ser_index[delay_ser[delay_ser.size()-1].second];
//			delay_ser.pop_back();
//			delay.pop();
//			task_cnt--;
//		}
//	}
//	
//	return ans;
//}

class TasksProcessing {
private:
	using PII = pair<int, int>;
public:
	vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
		int n = servers.size(), m = tasks.size();
		//server 
		priority_queue<PII, vector<PII>, greater<PII>> busy_que; //结束时间，所使用服务器下标
		priority_queue<PII, vector<PII>, greater<PII>> idle_que; //未使用的server
		for (int i = 0; i < n; ++i) {
			//初始化按照服务器权值排序，权值相同，按照下标排序
			idle_que.emplace(make_pair(servers[i], i)); 
		}
		
		int time_stamp = 0; // 当前时间
		//当busy队列中的任务满足t <= time_stamp ,则说明执行完成，可转移到idle

		//[captures] (params) mutable-> type{...} //lambda 表达式的完整形式
		auto release = [&]() {
			while (!busy_que.empty() && busy_que.top().first <= time_stamp) {
				auto&& [_, index] = busy_que.top(); //释放的服务器下标
				idle_que.emplace(servers[index], index); //加入到空闲服务器
				busy_que.pop();
			} // && universal reference 万能引用
		};

		vector<int> ans;
		for (int i = 0; i < m; ++i) { //遍历任务
			time_stamp = max(time_stamp, i);
			release();
			if (idle_que.empty()) { //空闲服务器为空
				time_stamp = busy_que.top().first; //为空则看一下正在运行的最小结束时间
				release();//当timestamp为busyque里的最小结束时间时，释放一下任务
			}

			auto&& [_, index] = idle_que.top();
			busy_que.emplace(make_pair(tasks[i] + time_stamp, index)); //该任务的结束时间和使用的服务器编号
			ans.push_back(index); //加入所使用的服务器编号，为空也可以
			idle_que.pop(); //将已经使用的服务器弹出
		}
		return ans;
	}
};


//1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？
vector<bool> canEat(vector<int>& candiesCount, vector<vector<int>>& queries) {
	//前缀和
	int n = candiesCount.size();
	vector<int> pre_sum(n+1,0);
	for (int i = 1; i <= n; ++i) {
		pre_sum[i] = pre_sum[i - 1] + candiesCount[i - 1];
	}
	vector<bool> ans(queries.size());
	//能否吃到，判断糖果范围在不在 pre_sum[queries[][0]]~pre_sum[queries[][0]+1]。吃的糖果的范围在  day*1~day*maxcandiesnum
	for (int i = 0; i < queries.size(); ++i) {
		int candies_low = queries[i][1] * 1;
		int candies_upper = queries[i][1] * queries[i][2];
		if (candies_upper<pre_sum[queries[i][0]] || candies_low>pre_sum[queries[i][0]+1]) {
			ans[i] = false;
		}
		else {
			ans[i] = true;
		}
	}
	return ans;   ////结果错了，下标错误
}


//523. 连续的子数组和
//前缀和+同余定理
bool checkSubarraySum(vector<int>& nums, int k) {
	int pre_sum = 0;
	int pre_mod = 0;
	unordered_map<int, int> mp;
	mp[0] = -1;
	for (int i = 0; i < nums.size();++i) {
		pre_sum += nums[i];
		pre_mod = pre_sum % k;
		if (mp.count(pre_mod)) {
			if (i - mp[pre_mod] >= 2) {
				return true;
			}
		}
		else {
			mp[pre_mod] = i;
		}
	}
	return false;
}



//525. 连续数组
//给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。
//使用前缀和
int findMaxLength(vector<int>& nums) {
	unordered_map<int, int> mp;
	int cnt = 0;
	int n = nums.size();
	vector<int> renums(n);
	for (int i = 0; i < n; ++i) {
		if (nums[i] == 0) {  //0变为-1
			renums[i] = -1;
		}
		else {
			renums[i] = nums[i];
		}
	}

	int maxlength = 1;
	mp[0] = -1;
	for (int i = 0; i < n; ++i) {
		cnt = cnt + renums[i];
		/*
			if(nums[i]==1){  //简写前缀和
            cnt++;
        }else{
            cnt--;
        }
		*/
		if (mp.count(cnt)) {
			int index = mp[cnt];
			maxlength = max(maxlength, i - index);
		}
		else {
			mp[cnt] = i;
		}
	}
	cout << maxlength << endl;
	return maxlength;
}




int reductionOperations(vector<int>& nums) {
	unordered_map<int, int> mp;
	for (auto a : nums) {
		mp[a]++;
	}
	set<int> st = set<int>(nums.begin(), nums.end());
	int cnt = 0, index = 0;
	for (auto a : st) {
		if (index == 0) {
			index++;
			continue;
		}
		cnt += mp[a] * index;
		index++;
	}
	return cnt;
}




//494. 目标和
/*
向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
*/

class targetsum {
private:
	int ans;
public:
	void dfs(vector<int>& nums, int target, int index, int temp_sum) {
		if(index==nums.size()){
			if (temp_sum == target) {
				ans++;
				return;
			}
			else {
				return;
			}
		}
		//当前位置为+
		dfs(nums, target, index + 1, temp_sum + nums[index]);
		//当前位置为-
		dfs(nums, target, index + 1, temp_sum - nums[index]);
		return;
	}

	int findTargetSumWays(vector<int>& nums, int target) {
		ans = 0;
		dfs(nums, target, 0, 0);
		return ans;
	}


	//使用dp
	/*
		当总和为target时，添加正号的和为non_neg，添加负号的和为neg, 则有non_neg = sum - neg。
		target = non_neg - neg=> target = sum-2 * neg => neg = (sum-target)/2
		将问题转化为，在数组nums中选取一定数目的数字，使得他们的和为neg
		dp[i][j]表示再前i个数字里，选取若干个数字，他们的和为j的方法
	*/
	int findTargetSumWays_dp(vector<int>& nums, int target) {
		int n = nums.size();
		int sum = accumulate(nums.begin(), nums.end(), 0);
		int gap = sum - target;
		int neg = (sum - target) / 2;
		if (gap < 0 || gap % 2 != 0) {
			return 0;
		}

		vector<vector<int>> dp(n + 1, vector<int>(neg + 1));
		dp[0][0] = 1;
		for (int i = 1; i <= n; ++i) {  //选取数
			for (int j = 0; j <= neg; ++j) { //值
				dp[i][j] = dp[i - 1][j];
				if (j >= nums[i - 1]) { //可选
					dp[i][j] += dp[i - 1][j - nums[i - 1]];
				}
				
				
			}
		}
		return dp[n][neg];
	}
};


//1049. 最后一块石头的重量 II
/*
每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。
*/

/*
按照  494.目标和  的方法， 给每块石头加上正号和负号，最后选出绝对值离0最近的结果 

方法一：回溯法 （不写）
方法二：dfs+mem
方法三：dp
*/
int lastStoneWeightII(vector<int>& stones) {
	//执行相减的石头，记为non
	//non的和最大为sum/2
	int n = stones.size();
	int sum = accumulate(stones.begin(), stones.end(), 0);
	int acc = sum / 2;
	vector<vector<int>> dp(n + 1, vector<int>(acc+1));  //dp[i][j]表示i个石头下，能装物品的最大值
	dp[0][0] = 0;
	for (int i = 1; i <= n; ++i) {
		for (int j = 0; j <= acc; ++j) {
			if (j >= stones[i - 1]) {
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - stones[i - 1]] + stones[i - 1]);
			}
			else {
				dp[i][j] = dp[i - 1][j];
			}
		}
	}
	for (int j = acc;; --j) {
		if (dp[n][j]) {
			return abs(sum - 2 * dp[n][j]);
		}
	}
}


//1888. 使二进制字符串字符交替的最少反转次数
//滑动窗口
int minFlips(string s) {
	string s_sum = s + s;
	int n = s.size();
	string s0 = "01";
	string s1 = "10";
	int cnt0 = 0, cnt1 = 0;
	int ans = 0x3f3f3f3f;

	for (int i = 0; i < 2 * n; ++i) {
		if (s_sum[i] != s0[i%2]) {
			cnt0++;
		}
		if (s_sum[i] != s1[i%2]) { //当前位不同，则修改次数+1
			cnt1++;
		}
		if (i >= n) {
			if (s0[(i - n) % 2] != s_sum[i - n]) {  //删除第一位，若第一位和字符串第一位不同，则说明不用再修改那一位，便把当时已经加上的修改次数减1
				cnt0--;
			}
			if (s1[(i - n) % 2] != s_sum[i - n]) {
				cnt1--;
			}
		}
		if (i >= n - 1) {
			ans = min({ ans,cnt0,cnt1 });
		}
	}
	return ans;
}

//879. 盈利计划
/*
n 名员工
第 i 种工作会产生 profit[i] 的利润，它要求 group[i] 名成员共同参与。
如果成员参与了其中一项工作，就不能参与另一项工作。
工作的任何至少产生 minProfit 利润的子集称为 盈利计划 。并且工作的成员总数最多为 n 。
有多少种计划可以选择

0-1背包，特殊情况，即要求价值最小
*/
int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit) {
	int MOD = (int)1e9 + 7;
	int m = group.size();
	vector<vector<vector<int>>> dp(m + 1, vector<vector<int>>(n + 1, vector<int>(minProfit + 1)));
	//dp[i][j][k]表示  i项工作，j个人，利润至少为k的方案数
	dp[0][0][0] = 1;
	for (int i = 1; i <= m; ++i) {
		for (int j = 0; j <= n; ++j) {
			for (int k = 0; k <= minProfit; ++k) {
				if (j >= group[i - 1]) { //人够
					dp[i][j][k] = dp[i - 1][j][k] + dp[i-1][j - group[i - 1]][max(0, k - profit[i - 1])];  //当前利润为k
					/*
					第一种是当前工作利润 >=k，此时该背包的容量直接为 0（也就是说当前的工作利润已经满足了这个条件了，以后的工作利润至少为 0 就可以了）；

					第二种是当前工作利润 < k，此时该背包的容量减少了 profit[i]
					（也就是说在遍历下一个工作时，满足的工作利润至少为 k - profit[i]，二者取较大的。

					*/
				}
				else { //人不够，无法开启
					dp[i][j][k] = dp[i - 1][j][k];
				}
			}
		}

	}
	int sum = 0;
	for (int j = 0; j <= n; j++) {
		sum = (sum + dp[m][j][minProfit]) % MOD;
	}
	return sum;
}

// 518. 零钱兑换 II
// 给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 
// 完全背包问题   组合问题
int change(int amount, vector<int>& coins) {
	int n = coins.size();
	vector<int> dp(amount + 1);
	dp[0] = 1;
	for (auto coin : coins) { //选硬币 =>找物品
		for (int i = coin; i <= amount; ++i) { //算价值=>算背包容量
			dp[i] += dp[i - coin]; //i从coin开始选，避免重复
		}
	}
	return dp[amount];
}


int bagProblemcomp(vector<int> weight, vector<int> value, int num, int capcity) {
	vector<vector<int>> dp(num + 1, vector<int>(capcity + 1)); //一共num件物品，背包容量为capcity
	//dp[i][j]表示将前i件物品装进限重为j的背包可以获得的最大价值
	for (int i = 1; i <= num; ++i) { //num
		for (int j = 0; j <= capcity; ++j) { //weight
			if (j >= weight[i - 1]) { //此时是可以装下物品i的
				dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i - 1]] + value[i - 1]); //选物品i与不选物品i.物品i可以重复选取，因此选取时写dp[i][...]
			}
			else {
				dp[i][j] = dp[i - 1][j];
			}
		}
	}
	return dp[num][capcity];
}

//279. 完全平方数
//给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
int numSquares(int n) {
	//1<=n<=10000
	//1<=sqrt(n)<=100
	//类似于 硬币面值为1~sqrt(n)
	//硬币数量最大为n
	vector<int> dp(n + 1,0x3f3f3f3f);
	dp[0] = 0;
	for (int i = 1; i*i <= n; ++i) {
		for (int j = (pow(i,2)); j <= n; j++) {
			dp[j] = min(dp[j], dp[j - (pow(i, 2))] + 1);
		}
	}
	return dp[n];
}

//bfs求解完全平方数
//若有结点值为12， 找出小于等于12的平方数，有1，4，9。 那么12的子节点分别为12-1  12-4  12-9
//按照此规律，可构建出多叉树 
 /*
		          12
	    11         8            3
   10    7    2    7    4       2
 9 6 1  6 3   1   6  3  3  0    1    //12-4-4-4 = 0

 所使用的最少的平方数，即为找到0时候的层数
 */     




//1449. 数位成本和为目标值的最大数字
/*
动态规划
目标值（容量）恰好为target
数字1~9的cost为cost[i-1]，每位可以重复选取
*/
string largestNumber(vector<int>& cost, int target) {
	//先使用dp计算出总cost为target的最长长度
	vector<int> dp(target + 1, -0x3f3f3f3f);
	dp[0] = 0;
	for (auto a : cost) {
		for (int i = a; i <= target; ++i) {
			dp[i] = max(dp[i],dp[i-a]+1);
		}
	}

	if (dp[target] < 0) {
		return "0";
	}//未找到恰巧等于target的组合
	
	int max_length = dp[target]; //最大长度 ，接着进行状态倒退
	//当dp[i]和dp[i-cost[j]+1]相等时。说明选取了第j个数
	string ans = "";
	for (int i = 8, sum_cost = target; i >= 0; --i) {  //找出在满足最长长度的情况下，最大的数，当然从大到小遍历
		for (int c = cost[i]; c <= sum_cost && dp[sum_cost] == dp[sum_cost - c] + 1; sum_cost -= c) {
			ans += ('1' + i);  //下标i对应的数字为i+1
		}
	}

	return ans;
}

//54双周赛
//5767. 检查是否区域内所有整数都被覆盖
//已知区间 ranges[i] = [starti, endi] ，如果整数 x 满足 starti <= x <= endi ，那么我们称整数x 被覆盖了
bool isCovered(vector<vector<int>>& ranges, int left, int right) {
	/*int tag = false;
	for (int i = left; i <= right; ++i) {
		tag = false;
		for (auto a : ranges) {
			if (i >= a[0] && i <= a[1]) {
				tag = true;
				break;
			}
		}
		if (tag == false) {
			return tag;
		}
	}
	return true;*/
	//暴力法，时间复杂度太高

	//使用差分数组
	/*对于差分数组，当[i,j]区间内全部+1，那么差分数组的i位置+1，j+1位置-1*/
	vector<int> dff(52, 0);
	for (auto range : ranges) {
		dff[range[0]]++;
		dff[range[1] + 1]--;
	}
	int pre_sum = 0;
	for (int i = 0; i <= 50; ++i) {
		pre_sum += dff[i];
		if (i >= left && i <= right && pre_sum <= 0) {
			return false;
		}
	}
	return true;
}


//5768. 找到需要补充粉笔的学生编号
class Solution_chalk {
private:
	using LL = long long;
public:
	int chalkReplacer(vector<int>& chalk, int k) {
		LL sum = accumulate(chalk.begin(), chalk.end(), 0LL);  //0为 long long，  只写0为int，累加会越界
		int p = k % sum; //
		int index = -1;
		for (int i = 0; i < chalk.size(); ++i) {
			if (p - chalk[i] >= 0) {
				p -= chalk[i];
			}
			else {
				index = i;
				break;
			}
		}
		return index;
	}
};


//1857. 有向图中最大颜色值
/*
路径的 颜色值 是路径中 出现次数最多 颜色的节点数目。
请你返回给定图中有效路径里面的 最大颜色值
colors[i] 是小写英文字母
edges[j] = [aj, bj] 表示从节点 aj 到节点 bj 有一条 有向边 。
*/
// 拓扑排序判断图中是否有环， bfs统计最大颜色数
int largestPathValue(string colors, vector<vector<int>>& edges) { //n == colors.length   m == edges.length  =》n个结点，m条边
	int node_size = colors.size();
	//邻接表
	vector<vector<int>> Graph(node_size);
	//入度
	vector<int> indeg(node_size,0);

	for (auto edge : edges) {
		Graph[edge[0]].push_back(edge[1]);
		indeg[edge[1]]++;
	}
	//初始化拓扑排序结点
	stack<int> st;
	for (int i = 0; i < node_size; ++i) {
		if (!indeg[i]) {
			st.push(i); // 入度为0
		}
	}


	int node_found = 0;  //node_found !=node_size  有环
	vector<vector<int>> color_rec(node_size, vector<int>(26));
	while (!st.empty()) {
		node_found++;
		int p = st.top();
		st.pop();
		color_rec[p][colors[p] - 'a']++;
		//开始bfs p的所有后继节点q并统计颜色数目
		for (auto q : Graph[p]) {
			--indeg[q]; //入度减1
			for (int i = 0; i < 26; ++i) {
				color_rec[q][i] = max(color_rec[p][i], color_rec[q][i]); //更新后继结点的路径颜色最大值
			}
			if (!indeg[q]) { //in==0 ?
				st.push(q);
			}
		}
	}
	if (node_found != node_size) {
		return -1;
	}
	int ans = 0;
	for (int i = 0; i < node_size; ++i) {
		ans = max(ans, *max_element(color_rec[i].begin(), color_rec[i].end()));
	}
	return ans;
}

//==================================================石子游戏=======================================
class Stone {
public:
//877. 石子游戏
bool stoneGame(vector<int>& piles) {
	//pair<先手拿能获得的最大分数，后手拿能获得的最大分数>
	int n = piles.size();
	vector<vector<PII>> dp(n, vector<PII>(n));
	//初始化dp
	for (int i = 0; i < n; ++i) {
		dp[i][i] = { piles[i], 0 };
	}

	for (int i = n - 2; i >= 0; --i) {
		for (int j = i + 1; j < n; ++j) {
			//先手选择左边或者右边，可以获得的分数
			int left = piles[i] + dp[i + 1][j].second;  //先拿了最左侧的i，那么只能获得[i+1][j]后手的分数
			int right = piles[j] + dp[i][j - 1].second;
			
			//先手肯定会选取最大的
			if (left > right) {
				dp[i][j].first = left;
				dp[i][j].second = dp[i + 1][j].first;
			}
			else {
				dp[i][j].first = right;
				dp[i][j].second = dp[i][j-1].first;
			}
		}
	}
	return max(dp[0][n - 1].first,dp[0][n-1].second);
}

bool stoneGame1_1(vector<int>& piles) {
	//dp[i][j]表示在[i,j]范围，拿石头，先手拿与后手拿的差值
	int n = piles.size();
	vector<vector<int>> dp(n, vector<int>(n));
	//初始化dp
	for (int i = 0; i < n; ++i) {
		dp[i][i] = piles[i];
	}
	//更新dp
	for (int i = n - 2; i >= 0; ++i) {
		for (int j = i + 1; j < n; ++j) {
			dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
			//当前的一方，拿走左侧石头i 或者 拿走右侧石头j ， 另一方在剩下石头里选取最优解
		}
	}
	return dp[0][n - 1] > 0;
}

};


//1895. 最大的幻方
//预先处理每行、每列的前缀和（将对角线的处理放到遍历中
int largestMagicSquare(vector<vector<int>>& grid) {
	//嗯算
	int m = grid.size(); //行
	int n = grid[0].size(); //列
	vector<vector<int>> pre_sum_row(m, vector<int>(n));  //行前缀和
	vector<vector<int>> pre_sum_col(m, vector<int>(n)); //列前缀和

	for (int i = 0; i < m; ++i) {
		pre_sum_row[i][0] = grid[i][0];
		for (int j = 1; j < n; ++j) {
			pre_sum_row[i][j] = pre_sum_row[i][j - 1] + grid[i][j];
		}
	}
	for (int j = 0; j < n; ++j) { //列
		pre_sum_col[0][j] = grid[0][j];
		for (int i = 1; i < m; ++i) { //行
			pre_sum_col[i][j] = pre_sum_col[i - 1][j] + grid[i][j];
		}
	}

	int edge_size = min(m, n);//最大的边长
	for (int i = 2; i <= edge_size; i++) {
		for (int j = 0; j + i <= m; ++j) {
			for (int k = 0; k + i <= n; ++k) { //枚举左上角(j,k)
				//计算第一行的和
				int reference_value = pre_sum_row[j][k + i - 1] - (k ? pre_sum_row[j][k-1] : 0);
				bool flag = false;
				//验证后续行，与第一行做比对
				for (int jj = j + 1; jj < j + i; ++jj) {
					int temp_sum = pre_sum_row[jj][k + i - 1] - (k ? pre_sum_row[jj][k-1] : 0);
					if (temp_sum != reference_value) {
						flag = true;
						break;
					}
				}
				if (flag) {
					continue;
				}
				//验证每一列是否和第一行的和相等
				for (int kk = k; kk < k + i; kk++) {
					int temp_sum = pre_sum_col[j + i - 1][kk] - (j ? pre_sum_col[j-1][kk] : 0);
					if (temp_sum != reference_value) {
						flag = true;
						break;
					}
				}
				if (flag) {
					continue;
				}

				//验证对角线
				int diag_temp_sum = 0, diag_temp_sum2 = 0;
				for (int c = 0; c < i; ++c) {
					diag_temp_sum += grid[j + c][k + c];
					diag_temp_sum2 += grid[j + c][k + i - 1 - c];
				}
				if (diag_temp_sum == reference_value && diag_temp_sum2 == reference_value) {
					return i;
				}
			}
		}
	}
	return 1;
}

//65. 有效数字
bool isNumber(string s) {
	int n = s.size();
	bool dot_flag = false; //.只能出现一次，因此要做标记
	for (int i = 0; i < n; ++i) {
		if (s[i] == '+' || s[i] == '-') {
			if (!((i < n - 1) && (s[i+1] - '0' >= 0 && s[i+1] - '0' <= 9) || s[i + 1] == '.')) { //+ - 号后为 数字 或者 .
				return false;
			}
		}
		else if (s[i] - '0' >= 0 && s[i] - '0' <= 9) { //为数字
			if (i < n - 1 && (s[i + 1] == '+' || s[i + 1] == '-')) {//数字后面不能跟+ 或 -
				return false;
			}
		}
		else if (s[i] == '.') { //遇到了.
	   //.的前或后必须有一个数字 ， 同时.也不应该出现两次
			if (!((i > 0 && s[i-1] - '0' >= 0 && s[i-1] - '0' <= 9) || (i < n - 1 && s[i + 1] - '0' >= 0 && s[i +1] - '0' <= 9)) || dot_flag) {
				return false;
			}
			dot_flag = true;
		}
		else if (s[i] == 'e' || s[i] == 'E') { //为e或者E 。 E/e的后面不能有小数
			if (i == 0 || i == n - 1) {
				return false;
			}
			else {
				for (int j = i + 1; j < n; ++j) {
					if ((s[j] == '+' || s[j] == '-') && !(j == i + 1 && j != n - 1)) {
						return false;
					}
					if ((s[j] == '.') || (s[j] - 'a' >= 0 && s[j] - 'a' < 26) || (s[j] - 'A' >= 0 && s[j] - 'A' < 26)) {
						return false;
					}
				} //遍历结束
				break;
			}

		}
		else {
			return false;
		}
	}
	return true;
}


//65. 有效数字
//有限状态机
class Legal_Number {
public:
	enum State { //枚举出所有状态
		STATE_INITIAL,
		STATE_INT_SIGN,
		STATE_INTEGER,
		STATE_POINT,
		STATE_POINT_WITHOUT_INT,
		STATE_POINT_FRACTION,  //小数部分
		STATE_EXP,
		STATE_EXP_SIGN,
		STATE_EXP_INT,
		STATE_END
	};

	enum CharType {
		CHAR_NUMBER,
		CHAR_SIGN,
		CHAR_POINT,
		CHAR_EXP,
		CHAR_ILLEGAL
	};

	CharType toCharType(char ch) { //string[i]->const char
		if (ch >= '0' && ch <= '9') {
			return CHAR_NUMBER;
		}
		else if (ch == '+' || ch == '-') {
			return CHAR_SIGN;
		}
		else if (ch == 'e' || ch == 'E') {
			return CHAR_EXP;
		}
		else if (ch == '.') {
			return CHAR_POINT;
		}
		else {
			return CHAR_ILLEGAL;
		}
	}

	bool isNumber(string s) {
		unordered_map<State, unordered_map<CharType, State>> transfer{
			//状态->数据类型->下一个状态
			//根据状态机的图来推导
			{
				STATE_INITIAL,{
					{CHAR_SIGN,STATE_INT_SIGN},
					{CHAR_POINT,STATE_POINT_WITHOUT_INT},
					{CHAR_NUMBER,STATE_INTEGER}
				}
			},
			{
				STATE_INT_SIGN,{
					{CHAR_NUMBER,STATE_INTEGER},
					{CHAR_POINT,STATE_POINT_WITHOUT_INT}
				}
			},
			{
				STATE_POINT_WITHOUT_INT,{
					{CHAR_NUMBER,STATE_POINT_FRACTION}
				}
			},
			{
				STATE_INTEGER,{
					{CHAR_NUMBER,STATE_INTEGER},
					{CHAR_EXP,STATE_EXP},
					{CHAR_POINT,STATE_POINT}
				}
			},
			{
				STATE_EXP,{
					{CHAR_NUMBER,STATE_EXP_INT},
					{CHAR_SIGN,STATE_EXP_SIGN}
				}
			},
			{
				STATE_POINT,{
					{CHAR_NUMBER,STATE_POINT_FRACTION},
					{CHAR_EXP,STATE_EXP}
				}
			},
			{
				STATE_EXP_INT,{
					{CHAR_NUMBER,STATE_EXP_INT}
				}
			},
			{
				STATE_EXP_SIGN,{
					{CHAR_NUMBER,STATE_EXP_INT}
				}
			},
			{
				STATE_POINT_FRACTION,{
					{CHAR_NUMBER,STATE_POINT_FRACTION},
					{CHAR_EXP,STATE_EXP}
				}
			}
		};

		int n = s.size();
		State st = STATE_INITIAL;
		for (int i = 0; i < n; ++i) {
			if (transfer[st].find(toCharType(s[i])) != transfer[st].end()) {
				st = transfer[st][toCharType(s[i])];
			}
			else {
				return false;
			}
		}
		return st == STATE_INTEGER || st == STATE_POINT || 
			   st == STATE_POINT_FRACTION || st == STATE_EXP_INT || st == STATE_END || st == STATE_INT_SIGN;
	}
};




//483. 最小好进制
using LL = long long;
int check(LL max_length, LL mid, LL N) {  //最大长度，当前进制，目标值
	LL sum = 0;
	for (int i = 0; i < max_length; ++i) {
		if (sum > (N - 1) / mid) {  //防止溢出
			return 1;
		}
		sum = sum * mid + 1;
	}
	if (sum == N) {
		return 0;
	}
	return sum > N ? 1 : -1;
}

string smallestGoodBase(string n) {
	//设数字n转化为a进制后全1
	//可以得到，n=a^0+a^1+...+a^(length(n)-1)
	//因为k>=2， 所以可以得到，全1的最大长度为log2(n)+2
	//穷举最大长度，二分合适每次长度的最小进制
	LL N = stoll(n);
	int max_length = (int)(log(N) / log(2) + 1); //转换为其他进制的最大长度

	for (int i = max_length; i >= 3; --i) {
		LL l = 2, r = N-1;	//找进制
		while (l <= r) {
			LL mid = l + (r - l) / 2;
			if (check(i, mid, N)>=0) {
				//减小进制
				r = mid - 1;
			}
			else {
				l = mid + 1;
			}
		}
		//验证是否等于
		if (check(i, l, N) == 0) {
			return to_string(l);
		}
	}
	return to_string(N - 1);
}


//1239. 串联字符串的最大长度
class StrLength {
private:
	int max_l;
public:
	int backtrack(vector<string>& arr, vector<int>& cnt, int index) {
		if (index == arr.size()) {
			return 0;
		}
		vector<int> temp = cnt;  //保存当前值，留给下次不选当前位置用
		for (auto a : arr[index]) {
			if (cnt[a - 'a']>0) {
				return backtrack(arr, temp, index + 1);
			}
			else {
				cnt[a - 'a']++; //出现			
			}
		}


		return max(backtrack(arr, temp, index + 1), (backtrack(arr, cnt, index + 1) + (int)arr[index].size()));
		
	}

	int maxLength(vector<string>& arr) {
		vector<int> cnt(26,0);
		int ans = backtrack(arr,cnt,0);
		return ans;
	}


	//不使用哈希表
	//使用二进制表示法来判断是否有重复数字
	int maxLength_1(vector<string>& arr) {
		vector<int> unique_subarr;
		//筛选出没有重复字母的子串
		for (auto& a : arr) {
			int m = 0;
			for (auto b : a) {
				//看m中是否包含重复字符串
				if ((m >> (b - 'a')) & 1) {
					m = 0; 
					break;
				}
				else {
					m |= (1 << (b - 'a'));
				}
			}
			if (m > 0) {
				unique_subarr.push_back(m);
			}
		}

		int ans = 0;
		//对本身没有重复字符串的子串进行匹配
		//[captures] (params) mutable-> type{...} //lambda 表达式的完整形式
		std::function<void(int, int)> backtrack = [&](int index, int rec) {  // index表示当前位置, rec记录字符使用情况便于判重
			if (index == (int)unique_subarr.size()) { //边界条件
				ans = max(ans, (int)__builtin_popcount(rec));
				return;
			}
			//选
			if ((unique_subarr[index] & rec) == 0) { //=0表示无重复元素
				backtrack(index + 1, rec | unique_subarr[index]);
			}
			//不选
			backtrack(index + 1, rec);
		};
		
		backtrack(0, 0);
		return ans;
	}

};


class BinaryWatch {
private:
	vector<string> ans;
public:
	string transfer(int num) {
		int hour = (num & 0b1111000000)>>6;
		int minute = (num & 0b111111);
		if (hour >= 12) {
			return "";
		}
		if (minute >= 60) {
			return "";
		}

		string time;
		time += to_string(hour);
		if (minute < 10) {
			time += ":0";
			time += to_string(minute);
		}
		else {
			time += ":";
			time += to_string(minute);
		}
		return time;
	}

	void backtrack(int choose, int index, int length, int allDigit) {
		if (length == 0) {
			string temp = transfer(choose);
			if (temp != "") {
				ans.push_back(temp);
			}
			return;
		}
		
		if (allDigit - index<length) {
			return;
		}

		//不选与选
		backtrack(choose, index + 1, length, allDigit);
		choose |= (1 << index);
		backtrack(choose, index + 1, length-1, allDigit);
		return;
	}

	vector<string> readBinaryWatch(int turnedOn) { //hour 0-11  minute 0-59
		//turnedOn表示亮灯的个数
		//vector<int> vec = { 1,2,4,8,1,2,4,8,16,32 };
		if (turnedOn > 8) {
			return ans;
		}
		backtrack(0, 0, turnedOn, 10);
		return ans;
	}
};


// 剑指 Offer 38. 字符串的排列
class Perms {
private:
	vector<string> ans;
	vector<int> visited;
public:
	void backtrack(string& s, int index, int n, string& temp) {
		if (index == n) {
			ans.emplace_back(temp);
			return;
		}

		for (int i = 0; i < n; ++i) {
			if (visited[i] || (i > 0 && visited[i - 1] && s[i] == s[i - 1])) {  //当前位置计算过，或者当前位置和前一个数位置一样
				continue;
			}
			visited[i] = 1;
			temp.push_back(s[i]);
			backtrack(s, index + 1, n, temp);
			temp.pop_back();
			visited[i] = 0;
		}
	}

	vector<string> permutation(string s) {
		int n = s.size();
		visited.resize(n, 0);
		string temp = "";
		sort(s.begin(), s.end());  //将字符串排序后，相同的字符都相邻，为了去除重复的情况，每种重复字符，只计算最左侧出现的一个
		backtrack(s, 0, n, temp);
		return ans;
	}
};


//31. 下一个排列
void nextPermutation(vector<int>& nums) {

	//=====此块测试STL函数 std::next_permutation
	vector<int> temp = { 1,2,3 };
	do {
		for (auto a : temp) {
			cout << a << "\t";
		}
		cout << endl;
	} while (std::next_permutation(temp.begin(),temp.end()));

	int temp2[] = { 1,2,3 };
	do {
		cout << temp2[0] << "\t" << temp2[1] << "\t" << temp2[2] << endl;
	} while (next_permutation(temp2, temp2 + 3));
	//===========================================

	int n = nums.size();
	int first_less = nums.size() - 1;
	int first_large = nums.size() - 1;
	while (first_less > 0 && nums[first_less] <= nums[first_less - 1]) {
		first_less--;
	}
	if (first_less >= 1) {  //当序列为5 4 3 2 1   first_less为0
		while (first_large >= first_less && nums[first_large] <= nums[first_less - 1]) {
			first_large--;
		}
		swap(nums[first_large], nums[first_less-1]);
		//swap很重要，当顺序为1 2 3 4 5 ， swap直接调整为1 2 3 5 4
		//解决初始问题
	}
	reverse(nums.begin() + first_less, nums.end());
}

//1906. 查询差绝对值的最小值
//   1 <= nums[i] <= 100
vector<int> minDifference(vector<int>& nums, vector<vector<int>>& queries) {
	//考虑一个数组中的差的最小值，当一个数组有序，那么相邻的两个数字有最小的差值
	//使用pre_sum[i][j]表示在nums[0-i]中j出现的数目
	int n = nums.size();
	int max_num = 100;
	vector<vector<int>> pre_sum(n + 1, vector<int>(101,0));

	for (int i = 0; i < n; i++) {
		copy_n(pre_sum[i].begin(), 101, pre_sum[i + 1].begin());
		++pre_sum[i + 1][nums[i]];
	}

	int m = queries.size();
	vector<int>ans;
	for (int i = 0; i < m; ++i) {
		int l = queries[i][0], r = queries[i][1];
		int temp = INT_MAX;
		int last_ele = 0;
		for(int j=1;j<=100;++j){
			if (pre_sum[r + 1][j] != pre_sum[l][j]) {
				if (last_ele) {
					temp = min(temp, j - last_ele);
				}
				last_ele = j;
			}
		}
		if (temp == INT_MAX) {
			temp = -1;
		}
		ans.emplace_back(temp);
	}
	return ans;
}


//1905. 统计子岛屿
// 图论搜索
//dfs / bfs
class ssss {
private:
	vector<vector<int>> move = { {0, -1}, {-1, 0}, {1, 0}, {0, 1} };
	vector<vector<int>> visited;
public:
	void que_clear(queue<pair<int, int>>& que) {
		queue<pair<int, int>> empty;
		swap(que, empty);
	}

	int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
		int m = grid1.size(), n = grid1[0].size();
		visited.resize(m, vector<int>(n));

		int ans = 0;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (visited[i][j] || grid2[i][j] == 0) {
					continue;
				}
				visited[i][j] = 1;
				queue<pair<int, int>> que;
				que.push(make_pair(i, j));
				int count = 1;

				while (!que.empty()) {
					auto temp = que.front();
					que.pop();
					if (grid1[temp.first][temp.second] == 0) {
						count = 0;
					}
					for (int k = 0; k < 4; ++k) {
						int new_x = temp.first + move[k][0];
						int new_y = temp.second + move[k][1];
						if (new_x < 0 || new_x >= m || new_y < 0 || new_y >= n || 
							visited[new_x][new_y] == 1 || grid2[new_x][new_y]==0) {
							continue;
						}
						visited[new_x][new_y] = 1;
						que.push(make_pair(new_x, new_y));
					}
				}
				ans += (count == 0 ? 0 : 1);
			}
		}
		return ans;
	}
};

//X-Y坐标系里，求最多有多少个点在同一条直线上。
int gcd(int x, int y) {
	return y == 0 ? x : (gcd(y, x % y));
}

int maxPoints(vector<vector<int>>& points) {
	int n = points.size();
	if (n <= 2) {
		return n;
	}
	int ans = 0;
	for (int i = 0; i < n; ++i) {
		if (ans > n / 2) {
			return ans;
		}
		unordered_map<string, int> mp;
		//斜率写为delta_y/delta_x，数据类型为float，会出现精度问题
		//将其改写为  delta_x/gcd(delta_x,delta_y)  delta_y/gcd(delta_x,delta_y)  
		for (int j = i + 1; j < n; ++j) {
			int del_y = points[j][1] - points[i][1];
			int del_x = points[j][1] - points[i][0];
			int temp = gcd(del_x, del_y);
			del_x = del_x / temp;
			del_y = del_y / temp;
			string str = to_string(del_x) + "_" + to_string(del_y);
			mp[str]++;
			ans = max(ans, mp[str]);
		}
	}
	return ans;
}

//752. 打开转盘锁
using PSI = pair<string, int>;
int openLock(vector<string>& deadends, string target) {
	if (target == "0000") {
		return 0;
	}
	unordered_set<string> deadcodes(deadends.begin(), deadends.end());
	if (deadcodes.count("0000")) {
		return -1;
	}

	//前移
	auto num_pri = [=](char x)->char {
		return x == '9' ? '0' : x + 1;
	};
	//后移
	auto num_pos = [=](char x)->char {
		return x == '0' ? '9' : x - 1;
	};

	//遍历所有的情况并存储
	auto gets = [&](string& s)->vector<string> {
		vector<string> temp;
		for (int i = 0; i < 4; ++i) { //每一位
			auto a = s[i];
			s[i] = num_pri(a);
			temp.push_back(s);
			s[i] = num_pos(a);
			temp.push_back(s);
			s[i] = a;
		}
		return temp;
	};

	queue<PSI> que;
	unordered_set<string> visited = {"0000"};
	que.emplace("0000", 0);
	while (!que.empty()) {
		auto [status, step] = que.front();
		que.pop();
		vector<string> vec = gets(status);
		for (auto a : vec) {
			if (!visited.count(a) && !deadcodes.count(a)) {
				if (a == target) {
					cout << step + 1 << endl;
					return step + 1;
				}
				visited.insert(a);
				que.emplace(a, step + 1);
			}
		}
	}
	return -1;
}


//773. 滑动谜题
// 少一块的拼图
/*
	bfs进行计算， 保存每一步可能走向的位置。   拼图大小  2 x 3
*/
/*
	要求结果为    1  2  3    ===》 编号   0  1  2
				4  5  0                3  4  5
			可以看到，每个位置能移动到的下一个位置为   
			0 -> 1  3
			1 -> 0  2  4
			2 -> 1  5
			3 -> 0  4
			4 -> 1  3  5 
			5 -> 2  4
	 因此，当找到初始值0的位置，那么便可得到下一次的位置
	*/
vector<vector<int>> next_step{ {1,3},{0,2,4},{1,5},{0,4},{1,3,5},{2,4} };
int slidingPuzzle(vector<vector<int>>& board) {
	//next step
	string target = "123450";
	auto trans = [&](string& s)->vector<string> {
		vector<string> vec{};
		int zero_index = s.find('0');
		for (auto a : next_step[zero_index]) {
			string ss = s;
			swap(s[a], s[zero_index]);
			vec.emplace_back(s);
			s = ss;
		}
		return vec;
	};

	string initial = "";
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			initial += (board[i][j] + '0');
		}
	}
	if (initial == target) {
		return 0;
	}

	queue<pair<string,int>> que;
	que.emplace(initial,0);
	unordered_set<string> us{ initial };
	while (!que.empty()) {
		auto [begin, step] = que.front();
		que.pop();
		vector<string> temp_vec = trans(begin);
		for (auto aa : temp_vec) {
			if (!us.count(aa)) {
				if (aa == target) {
					return step + 1;
				}
				que.emplace(aa, step + 1);
				us.insert(move(aa));  //直接拷贝，节省时间
				//move(a) 之后，a会变为空
				//一定要先存到que再使用move存储
			}
		}
	}

	return -1;
}



//5799. 最美子字符串的数目      ////////字符串类前缀和，没掌握
/*
1.只考虑每个字母频率的奇偶性，就只考虑模2后的结果即可

只有0和1两种情况

2.只有10个字母，又只有0和1两种情况

用状态压缩

3.如果是区间dp的思想，就可能做不出
用前缀dp的思路，就ok

4.前缀和加速查找当前位置，前缀区间，某个状态的值

5.题目最多只有一个字母的频率为奇数====只有一个字母的奇偶性不同====state只有一位不同
（1）0个字母的频率为奇数
（2）10个字母中，某个字母的频率为奇数
共11种情况

6.初始化
presum_state[0] = 1
一个字母都不选的时候，state = 0000000000，种类数为1
*/
long long wonderfulSubstrings(string word) {    //《===这个方法没理解     类似dp？
	int n = word.size();
	//区间问题，考虑前缀和
	vector<int> pre(1<<10); //保存所有的情况
	pre[0] = 1;
	int state = 0;
	int ans = 0;
	for (auto w : word) {
		int t = 1 << (w - 'a'); 
		state = state ^ t; //当前前缀和
		ans += pre[state];//所有字符均出现偶数次
		//有一个字母出现奇数次
		for (int i = 1; i < (1 << 10); i = i<<1) {
			ans += pre[state ^ i];  //枚举第i个字母出现的奇偶性
		}
		pre[state]++;
	}
	return ans;
}


//909. 蛇梯棋
/*
bfs
queue存储的是每一次所能到达的下一步，期间遇到梯子或者蛇，下一步直接变为蛇或梯子的目标点
普通走的话，每次只能向前走1~6步

虽然棋盘的编号是反的，但是假装左下角为（0，0）， 从下到上顺序递增就行
*/
using TIII = tuple<int, int, int>;
int snakesAndLadders(vector<vector<int>>& board) {
	int n = board.size();
	//穷举下一步
	auto nextStep = [&](int& a)->vector<TIII> {
		vector<TIII> vec;
		for (int i = 1; i <= 6; ++i) {
			if (a + i > n * n) {
				continue;
			}
			int x = (a - 1 + i) / n, y = (a - 1 + i) % n;  // 新的行 列 下标
			if (x % 2 == 1) { //从下往上，奇数行逆向 最下面一行编号为0
				y = n - 1 - y;
			}
			vec.emplace_back(n - 1 - x, y, a + i); //下一步的横纵坐标, 和编号
			//行号变为 n-1-x
		}
		reverse(vec.begin(), vec.end());
		return vec;
	};
	vector<int> visited(n * n);
	visited[1] = 1;
	queue<PII> que;
	unordered_set<int> us{ 1 };
	que.emplace(1,0);
	while (!que.empty()) {
		auto [begin,step] = que.front(); //初始为1
		que.pop();
		auto temp = nextStep(begin);
		for (auto a : temp) {
			int next = get<2>(a);
			if (board[std::get<0>(a)][get<1>(a)] > 0) { //存在蛇或者梯子，那么直接跳跃
				next = board[std::get<0>(a)][get<1>(a)];
			}
			if (next == n * n) {
				return step + 1;
			}
			if (!visited[next]) {
				visited[next] = 1;
				que.emplace(next, step + 1);
			}
		}
	}
	return -1;
}

//LCP 07. 传递信息 easy
int numWays(int n, vector<vector<int>>& relation, int k) {
	//要求经过k轮传递到最后一个人
	//dp[i][j]表示经过i轮传到j的方案数
	//dp[0][0]=1
	vector<vector<int>> dp(k+1, vector<int>(n));
	dp[0][0] = 1;
	for (int i = 1; i <= k; ++i) {
		for (auto a : relation) {
			dp[i][a[1]] += dp[i - 1][a[0]];
		}
	}
	return dp[k][n-1];
}


//1418. 点菜展示表
vector<vector<string>> displayTable(vector<vector<string>>& orders) {
	//保存菜品
	//保存桌号以及每桌所点的菜品和数量
	unordered_map<int, unordered_map<string, int>> table;
	unordered_set<string> food;

	//提取orders的内容
	for (auto& a : orders) {
		table[atoi(a[1].c_str())][a[2]]++;
		food.emplace(a[2]);
	}

	vector<string> food_name{ "Table" };
	for (auto& a : food) {
		food_name.emplace_back(a);
	}

	sort(food_name.begin() + 1, food_name.end());

	vector<vector<string>> ans;

	ans.emplace_back(food_name);

	for (int i = 1; i <= 500; ++i) {
		if (!table[i].empty()) {
			vector<string> temp;
			temp.emplace_back(to_string(i));
			for (int j = 1; j < food_name.size(); ++j) {
				//if(table[i][a]!=0){
				temp.emplace_back(to_string(table[i][food_name[j]]));
				//}//else{
					//temp.emplace_back("0");
				//}
			}
			ans.emplace_back(temp);
		}
	}
	return ans;
}


//726. 原子的数量
/*
给定一个化学式formula（作为字符串），返回每种原子的数量。

原子总是以一个大写字母开始，接着跟随0个或任意个小写字母，表示原子的名字。
如果数量大于 1，原子后会跟着数字表示原子的数量。如果数量等于 1 则不会跟数字。H2O 和 H2O2 是可行的，但 H1O2 这个表达是不可行的。

两个化学式连在一起是新的化学式
一个括号中的化学式和数字（可选择性添加）也是化学式
*/
string countOfAtoms(string formula) {
	int n = formula.size();
	int index = 0;
	auto findAtom = [&]()->string {
		string temp = "";
		temp += formula[index++]; //保存第一个大写字母
		while (index < n && islower(formula[index])) {
			temp += formula[index];   //小写的也加上
			index++;
		}
		//原子统计完了
		return temp;
	};

	//原子之后为数字，数字>=1    ，为1时，不显示
	auto findNum = [&]()->int {
		if (index == n || !isdigit(formula[index])) {
			//结束了或者原子的后一位不为数字
			//那么说明原子只有一个
			return 1;
		}
		//原子后有数字
		int ans = 0;
		while (index < n && isdigit(formula[index])) {
			ans = ans * 10 + (formula[index] - '0');
			index++;
		}
		return ans;
	};

	//因为要处理括号，所以选择stack
	stack<unordered_map<string, int>> st;
	st.push({});
	while (index < n) {
		if (formula[index] == '(') {
			st.push({});  //存储当前的原子
			index++;
		}
		else if (formula[index] == ')') {
			//当前原子已经走完了
			//看看括号外有没有数字
			index++;
			int right_num = findNum();
			auto temp = st.top();
			//去掉括号，加到上一层
			st.pop();
			for (auto& [atom,count] : temp) {
				st.top()[atom] += count * right_num;
			}
		}
		else {
			string atom = findAtom();
			int num = findNum();
			//加到当前原子的哈希表中
			st.top()[atom] += num;
		}
	}

	//st中为所有的原子数
	vector<pair<string, int>> vec;
	auto temp = st.top();
	for (auto& [atom, num] : temp) {
		vec.emplace_back(atom, num);
	}

	sort(vec.begin(), vec.end());

	string ans = "";
	for (auto a : vec) {
		ans += a.first;
		if (a.second > 1) {
			ans += (to_string(a.second));
		}
	}
	return ans;

}

//981. 基于时间的键值存储
class TimeMap {
public:
	unordered_map<string, vector<pair<int, string>>> mp;
	/** Initialize your data structure here. */
	TimeMap() {

	}

	void set(string key, string value, int timestamp) {
		mp[key].emplace_back(timestamp, value);
	}
	//所有的键/值字符串都是小写的。
	string get(string key, int timestamp) {
		auto& a = mp[key];
		auto index = upper_bound(a.begin(), a.end(), make_pair(timestamp, string({127})));
		//string({127})为最大的字符串
		//string之间的比较，  c与abc    c为99  a为97  .  当c与abc之间比较时，按最短的长度来比较，即只比较 a 97  与  c 99  =>> abc<c
		//string({123-127})均可以，因为c为122
		if (index > a.begin()) {
			return (index - 1)->second;
		}
		return "";
	}
};




class AAA {
public:
	int ans = 0;
	unordered_set<string> st;
	int isPalindrome(string str) {
		int n = str.size();
		int l = 0, r = n - 1;
		while (l <= r) {
			if (str[l] != str[r]) {
				return 0;
			}
			l++;
			r--;
		}
		return 1;
	}

	void bfs(int count, string& s, string& temp, int judge) {
		if (judge == s.size()) {
			if (count ==3 && isPalindrome(temp)) {
				st.insert(temp);
			}
			return;
		}

		if (judge < s.size()&&count==3) {
			if (isPalindrome(temp)) {
				st.insert(temp);
			}
			return;
		}

		//选
		
		bfs(count + 1, s, temp += s[judge], judge + 1);
		temp.pop_back();
		
		//不选
		bfs(count, s, temp, judge + 1);
	}

	int countPalindromicSubsequence(string s) {
		// 长度为3
		string temp = "";
		bfs(0, s, temp, 0);
		ans = st.size();
		return ans;
	}
};


//930. 和相同的二元子数组
//前缀和+map
int numSubarraysWithSum(vector<int>& nums, int goal) {
	// mp里保存的是sum-goal
	int ans = 0;
	int sum = 0;
	unordered_map<int, int> mp;
	for (auto& a : nums) {
		mp[sum]++;  //sum == goal的时候，为0，所以0要记一次
		sum += a;
		ans += mp[sum - goal];  //<--逆向思维
	}
	return ans;
}


//218. 天际线问题
//1、扫描线
//2、线段树
vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
	auto cmp = [&](const pair<int, int>& a, pair<int, int>& b)->bool {return a.second < b.second; };
	priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> que(cmp);
	//priority_queue的第三个参数为class类型，decltype自动推断类型

	vector<int> index;
	for (auto& a : buildings) {
		index.emplace_back(a[0]);
		index.emplace_back(a[1]);
	}
	sort(index.begin(), index.end());  //所有左右端点排序 

	vector<vector<int>>ans;
	int n = buildings.size();
	int idx = 0;
	//按照所有的坐标，划分成的矩形，对于每个矩形，找到其中的最大值（因为可能同一位置有多栋建筑）
	for (auto& a : index) {  //遍历矩形
		while (idx < n && buildings[idx][0] <= a) {   //while防止了同一左端点上有多栋建筑的情况
			que.emplace(buildings[idx][1], buildings[idx][2]);  //存入右端点和高度
			idx++;
		}
		//若所有建筑左端点都不重叠的话，那么while可以写为if
		while (!que.empty() && que.top().first <= a) {
			que.pop();  //<==说明已经计算过
		}
		int highest = que.empty() ? 0 : que.top().second;
		if (ans.size() == 0 || highest != ans.back()[1]) {
			ans.push_back({ a,highest });
		}
	}
	return ans;
}

//1818. 绝对差值和
//对于取MOD后的操作，很巧妙
static constexpr int MOD = 1000000007;
int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
	//排完序再找对于每个nums[i]最接近的nums[j]
	//使得| |nums1[i]-nums2[i]| - |nums1[j]-nums2[j]| |有最大值
	vector<int> vec(nums1);
	sort(vec.begin(), vec.end());

	int max_gap = 0;
	int n = nums1.size();
	int sum = 0;
	for (int i = 0; i < n; ++i) {
		int i_gap = abs(nums1[i] - nums2[i]);
		sum = (sum + i_gap) % MOD;
		int j = lower_bound(vec.begin(), vec.end(), nums2[i]) - vec.begin();
		//找到nums1中距离nums2[i]最近的一个数
		if (j < n) {
			max_gap = max(max_gap, i_gap - abs(vec[j] - nums2[i]));
		}
		if (j > 0) {  //试j的前一个
			max_gap = max(max_gap, i_gap - abs(vec[j - 1] - nums2[i]));
		}
	}
	//sum因为取了mod，因此会出现没取模前sum>max_gap， 取模后sum<max_gap，实际上少了一个MOD
	//因此需要加上mod
	return (abs(sum - max_gap + MOD)) % MOD;
}


//1846. 减小和重新排列数组后的最大元素
int maximumElementAfterDecrementingAndRearranging(vector<int>& arr) {
	int n = arr.size();
	vector<int> count(n + 1);  //存储1~n的数出现的次数
	int have = 0;
	for (auto a : arr) {
		count[min(a, n)]++; //<==min(a,n)的考虑了当a大于n的情况,很巧妙
	}
	int miss = 0;
	int tag = 1;
	for (int i = 1; i <= n; ++i) {
		if (count[i] == 0) {
			miss++;
		}
		else {//要注意只能变为比当前值小的值
			miss -= min(miss, count[i] - 1);
			//在当前位置之前有miss个空缺，最多只能消除miss个，因为只能变为比当前值小的值
		}
	}
	return n - miss;
	//一共有n个数，若miss=1即缺失了1个，那么当前的最大值肯定为4，又因为只能填充比自己小的数
	//所以最大值为4
}


//815. 公交路线
//因为数据量太大，若存储每个路线对应的站点，那么遍历的数量级太大
//因此存储每个站点对应的路线，牺牲一点空间换取大量的时间
int numBusesToDestination(vector<vector<int>>& routes, int source, int target) {
	if (target == source) {
		return 0;
	}
	int n = routes.size();
	unordered_map<int, vector<int>> rec;
	//存储每个站点包含哪些路线
	for (int i = 0; i < n; ++i) {
		for (auto& a : routes[i]) {
			rec[a].push_back(i);
		}
	}
	//存储完毕后，准备进行bfs
	vector<int> visited(n, 0);//记录线路的访问情况
	queue<int> que;
	for (auto& a : rec[source]) {
		visited[a] = 1;
		que.push(a);
	}
	int step = 1;
	while (!que.empty()) {
		int m = que.size();
		for (int i = 0; i < m; ++i) {
			int tp = que.front();
			que.pop();
			for (auto& a : routes[tp]) {   //路线里有哪些点
				if (a == target) {
					return step;
				}
				for (auto& b : rec[a]) { //存储每个点对应的新路线					
					if (!visited[b]) {
						visited[b] = 1;
						que.push(b);
					}
				}
			}
		}
		step++;
	}
	return -1;
}


//线段树的思想递归求解
//53. 最大子序和     (数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。)
//同 剑指 Offer 42. 连续子数组的最大和 
class SegmentTree_exp1 {
public:
	struct Treeproperty {
		int lsum, rsum, isum, msum;
		//lSum 表示 [l,r] 内以 l 为左端点的最大子段和
		//rSum 表示 [l,r] 内以 r 为右端点的最大子段和
		//msum 表示 [l,r] 的最大子段和
		//iSum 表示 [l,r] 内区间和
	};

	/*
		假设[l,m] 为 [l,r] 的「左子区间」，[m+1,r] 为 [l,r] 的「右子区间」
		对于长度为 1 的区间 [i,i]，四个量的值都和 nums[i] 相等。对于长度大于 1 的区间：
			最好维护的是iSum，区间[l,r]的isum等于左区间的isum加上右区间的isum
			对于[l,r]的lsum，存在两种可能，一是可以等于左区间的lsum，二是可以等于左区间的isum + 右区间的lsum。 二者取大
			对于[l,r]的rsum，一是可以等于右区间的rsum，二是可以等于左区间的rsum + 右区间的isum。  二者取大
			对于[l,r]的msum， 当msum不跨越m时候， msum等于左区间的msum,或者右区间的msum, 若跨越了m， 
		  那么 msum 等于 左区间的rsum 加上 右区间的 lsum，三者取最大
	*/

	Treeproperty pullUp(Treeproperty l, Treeproperty r) { //区间长度变为1后，向上合并
		int isum = l.isum + r.isum;
		int lsum = max(l.lsum, l.isum + r.lsum);
		int rsum = max(r.rsum, l.rsum + r.isum);
		int msum = max(max(l.msum, r.msum), l.rsum + r.lsum);
		//两个区间合并已经完成
		return Treeproperty{ lsum,rsum,isum,msum };
	}

	Treeproperty getValue(vector<int>& nums, int left, int right) {  //迭代到区间长度为1
		if (left == right) {
			return Treeproperty{nums[left], nums[left], nums[left], nums[left] };
		}
		int mid = left + (right - left) / 2;
		Treeproperty lSub = getValue(nums, left, mid);
		Treeproperty rSub = getValue(nums, mid + 1, right);
		//划分完成
		return pullUp(lSub, rSub);
	}

	int maxSubArray(vector<int>& nums) {
		int n = nums.size();
		Treeproperty ans = getValue(nums, 0, n - 1);
		return ans.msum;
	}
};


//1838. 最高频元素的频数
//在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。
//执行**最多 k 次**操作后，返回数组中最高频元素的最大可能频数(即使得数组中最多出现多少个相同的元素)
int maxFrequency(vector<int>& nums, int k) {
	sort(nums.begin(), nums.end());
	int n = nums.size();
	int l = 0, r = 0;
	int ans = 1;
	int count = k;
	while (l <= r && r < n) {
		int gap = l==r?0:nums[r] - nums[r-1];
		int len = r - l;
		long long need = gap * len;
		if (need <= count) {
			ans = max(ans, r - l + 1);			
			r++;
			count -= need;
		}
		else { //不够了，从最左侧开始删
			l++;
			count += (nums[r-1] - nums[l - 1]); //恢复补充的
		}		
	}
	return ans;
}


//5815. 扣分后的最大得分
//间隔行得同一列不扣分，不同列i,j。扣分为j-i
long long maxPointss(vector<vector<int>>& points) {
	//hint 1 : dp
	//dp[i][j]表示选取points[i][j]后所获得的最大分数
	int m = points.size();
	int n = points[0].size();
	vector<vector<int>> dp(m, vector<int>(n));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0) {
				dp[i][j] = points[i][j];
			}
			else {
				/*
				int max_val = 0;
				for (int k = 0; k < n; ++k) {   //<==遍历上一行的所有dp值，会造成超时
					max_val = max(max_val, dp[i - 1][k] + points[i][j] - abs(j - k));
				}
				dp[i][j] = max_val;
				*/
			}
		}
	}
	return *max_element(dp[m - 1].begin(), dp[m - 1].end());
}


//138 复制带随机指针的链表
//构造这个链表的深拷贝
//因为有随机指针，因此不能像普通链表复制进行操作，因为random指向的结点可能还没构建
struct Node {
	int val;
	Node* next;
	Node* random;
	Node(int _val):
		val(_val),
		next(NULL),
		random(NULL){};
};

class Randomptr {
public:
	Node* copyRandomList(Node* head) {
		//因为有random指针，先使用map保存下来所有的指针，便于得到新旧指针的对应关系
		unordered_map<Node*, Node*> mp;
		Node* p = head;
		while (p != nullptr) {
			mp[p] = new Node(p->val);  //保存新的结点，只保存了值
			p = p->next;
		}
		//进行next和random的拷贝
		p = head;
		while (p) {
			Node* newnode = mp[p];
			if (p->next)
				newnode->next = mp[p->next];
			if (p->random)
				newnode->random = mp[p->random];
			p = p->next;
		}
		return mp[head];
	}
};


//1743. 从相邻元素对还原数组
vector<int> restoreArray(vector<vector<int>>& adjacentPairs) {
	int n = adjacentPairs.size() + 1;
	unordered_map<int, vector<int>> mp;
	for (auto& a : adjacentPairs) {
		mp[a[0]].push_back(a[1]);
		mp[a[1]].push_back(a[0]);
	} // 记录相邻

	//find begin
	int begin = 0;
	for (auto i = mp.begin(); i != mp.end(); i++) {
		if (i->second.size() == 1) {
			begin = i->first;
		}
	}
	vector<int> ans(n);
	ans[0] = begin;
	ans[1] = mp[begin][0];
	begin = ans[1];
	for (int i = 2; i < n; ++i) {
		begin = mp[begin][0] == ans[i - 2] ? mp[begin][1] : mp[begin][0];  
		// 1->2    2->1, 3   在2开始的时候，要选取3，不选取1
		ans[i] = begin;
	}
	return ans;
}



int getLucky(string s, int k) {
	string trans = "";
	for (auto& a : s) {
		trans+=to_string(a - 'a' + 1);
	}
	string temp = "";
	while (k) {
		int sum = 0;
		for (auto& a : trans) {
			sum += (a - '0');
		}
		temp = to_string(sum);
		trans = temp;
		k--;
	}
	int ans = 0;
	for (auto& c : temp) {
		ans = ans * 10 + (c - '0');
	}
	return ans;
}


int maxCompatibilitySum(vector<vector<int>>& students, vector<vector<int>>& mentors) {
	int m = students.size();
	int n = students[0].size();

	vector<vector<int>> dp(m, vector<int>(m));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < m; ++j) {
			int temp = 0;
			for (int k = 0; k < n; ++k) {
				if (students[i][k] == mentors[j][k]) {
					temp++;
				}
			}
			dp[i][j] = temp;
		}
	}
	vector<int> vec(m);
	for (int i = 0; i < m; ++i) {
		vec[i] = i;
	}
	int ans = 0;
	do {
		int cc = 0;
		for (int i = 0; i < m; ++i) {
			cc += dp[i][vec[i]];
		}
		ans = max(ans, cc);
	} while (std::next_permutation(vec.begin(), vec.end()));
	return ans;
}


struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) :val(x), left(nullptr), right(nullptr) {};
	TreeNode(int x, TreeNode* left, TreeNode* right) :val(x), left(left), right(right) {};
};

class Solutiontt {
public:
	unordered_map<int, vector<int>> gra;
	vector<int> ans;

	void trans(TreeNode* root) {
		if (!root) {
			return;
		}
		int val = root->val;
		if (root->left) {
			int lval = root->left->val;
			gra[val].push_back(lval);
			gra[lval].push_back(val);
			trans(root->left);
		}
		if (root->right) {
			int rval = root->right -> val;
			gra[val].push_back(rval);
			gra[rval].push_back(val);
			trans(root->right);
		}
	}

	void dfs(int x, int pre,int k) {
		if (k == 0) {
			ans.push_back(x);
		}
		for (auto& a : gra[x]) {
			if (a != pre) {
				dfs(a, x, k - 1);
			}
		}
	}

	vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {
		//树也是一种特殊的图
		//将其转换为无向图
		trans(root);
		dfs(target->val, target->val, k);		
	}
};





//1104 二叉树寻路
//Z字形
int getstart(int level) {
	return (1 << level)/2;
}
int getend(int level) {
	return (1 << (level + 1)) / 2 - 1;
}

vector<int> pathInZigZagTree(int label) {
	//对于taregt而言
	//转化为二进制，最高位11在第几个最高位，则在第几行
	int max_one = 0;
	int pre_one = 0;
	int index_bit=1;
	int cnt = 0;
	while (index_bit<=label) {
		index_bit= index_bit<<1;
		cnt++;
	}
	int start = getstart(cnt);
	int end = getend(cnt);
	//找到了当前行的头尾
	//若为奇数行，则考虑上一行来得到上一行的编号
	//若为偶数行，考虑当前行来得到上一行的父结点编号
	vector<int> ans;
	ans.push_back(label);
	while (cnt >1) {
		int f_start = getstart(cnt - 1);
		int f_end = getend(cnt - 1);
		int f_label = label / 2; //label的父结点
		int f_val = f_start + (f_end - f_label);
		ans.insert(ans.begin(),f_val);
		label = f_val;
		cnt--;
	}
	return ans;
}



//671. 二叉树中第二小的节点
//每个父结点为左右子节点中较小的那个、
//找到左右节点中较小的那个，依次往下，直到找到刚好大于target的
class Secondmin {
public:

	int findSecondMinimumValue(TreeNode* root) {
		if (!root)
			return -1;
		//顶点肯定为最小的值
		int ans = getval(root,root->val);
		return ans;
	}

	int getval(TreeNode* node,int target) {
		if (!node) {
			return -1;
		}
		if (node->val > target) {
			return node->val;
		}
		int left = getval(node->left, target);
		int right = getval(node->right,target);
		if (left == -1) {
			return right;
		}
		if (right == -1) {
			return left;
		}
		return min(left, right);
	}
};




//252周赛
//1953
long long numberOfWeeks(vector<int>& milestones) {
	long long max_ele = *max_element(milestones.begin(), milestones.end());
	long long sum = accumulate(milestones.begin(), milestones.end(), 0LL);
	long long sub = sum - max_ele;
	long long ans = 0;
	if (sub < max_ele) {
		ans = 2 * sub + 1;
	}
	else {
		ans = sum;
	}
	return ans;
}





//1954
//坐标(i, j)处有|i|+|j|个，  选择以(0,0)为中心的正方形,求包含neededApples个的最小正方形周长
long long minimumPerimeter(long long neededApples) {
	/*
		对于右上角为(n,n)的正方形，区域内的总和为
		sum = sigma|x|+|y|   x=(-n,n) y=(-n,n)
		//    x与y对称，因此|x|+|y|可变为2*|x| ,同时，y有2n+1和取值，所以消去y后变为  2*(2n+1)
			= 2*(2n+1)*sigma|x|   x = (-n,n) 
		    = 2*(2n+1) * sigma|x| x=(-n,n) 
			= 2*(2n+1)*(2) sigma|x| x=(1,n)
			= (2n+1)*(2n)*(n+1)
	*/
	long long edge = 0;
	for (; (2 * edge + 1) * (2 * edge) * (edge + 1) < neededApples;++edge) {}
	return edge;
}


//313 超级丑数
//与264类似。264为1，3，5   313为primes
//此处使用两种方法
//1 优先队列存储所有出现的，每次取最小
//2 类似于三指针，使用多指针
int nthSuperUglyNumber(int n, vector<int>& primes) {
	//每次取最小，使用最小堆或者priority_queue
	if (n == 1)
		return 1;
	priority_queue<int,vector<int>,greater<int>> que;
	unordered_set<int> uset;
	que.push(1);
	uset.insert(1);
	vector<int> dp(n);
	for (int i = 0; i < n; ++i) {
		int t = que.top();
		dp[i] = t;
		que.pop();
		for (auto& a : primes) {
			long long temp = a * t;
			if (!uset.count(temp)) {
				que.push(temp);
				uset.insert(temp);
			}	
		}
	}
	return dp[n - 1];
}

//case2     三指针的延伸
int nthSuperUglyNumber_multi_pointer(int n, vector<int>& primes) {
	if (n == 1) {
		return 1;
	}
	int len = primes.size();
	vector<int> pointers(len,1);  //pointers[i]表示peimes中的第i个数所对应的乘数
	vector<int> dp(n+1);
	dp[1] = 1;
	for (int i = 2; i <= n; ++i) {
		int min_val = INT_MAX;
		vector<int> nums(len);
		for (int j = 0; j < len; ++j) {
			nums[j] = dp[pointers[j]] * primes[j];
			min_val = min(min_val, nums[j]);
		}
		dp[i] = min_val;
		for (int k = 0; k < len; ++k) {
			if (min_val == nums[k]) {
				pointers[k]++;  
			}
		}
	}
	return dp[n];
}


//1137斐波那契数加上记忆化搜索
int cache[38];
int tribonacci(int n) {
	if (n == 0) {
		return 0;
	}
	else if (n == 1) {
		return 1;
	}
	else if (n == 2) {
		return 1;
	}
	if (cache[n] != 0) {  //<记忆化搜索
		return cache[n];
	}
	cache[n] = tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3);
	return cache[n];
}


//1337. 矩阵中战斗力最弱的 K 行
vector<int> kWeakestRows(vector<vector<int>>& mat, int k) {
	int m = mat.size();
	vector<int> index(m);
	iota(index.begin(), index.end(), 0);
	sort(index.begin(), index.end(), [&](const auto& a, const auto& b) {
		return mat[a] == mat[b] ? a < b : mat[a] < mat[b];
		//可以直接对vector进行比较，rec.cpp中写了例子
	});
	vector<int> ans(index.begin(), index.begin() + k);
	return ans;
}


//743. 网络延迟时间
//从结点k发送一个信号，多久才能传播到所有节点。不能使所有节点收到信号，则返回-1
//n个结点，起点为k
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
	//dijkstra
	//n个结点， 编号从1~n
	vector<vector<int>> graph(n+1, vector<int>(n+1,111)); //adjacency matrix
	vector<int> dist(n+1, 111);  //distance from source node,  最大为100，111当作无限
	vector<int> visited(n + 1,0);
	vector<int> pre(n + 1,0);

	for (auto a : times) {
		graph[a[0]][a[1]] = a[2];
		if (a[0] == k) {
			dist[a[1]] = a[2];
			pre[a[1]] = a[0];
		}
	}
	dist[k] = 0;
	//visited[k] = 1;
	for (int i = 1; i <= n; ++i) {  //找点
		int temp = -1;
		for (int j = 1; j <= n; ++j) {
			if (!visited[j] && (temp == -1 || dist[j] < dist[temp])) { //j没被访问过，并且j的距离短于temp的距离
				//-1是为了初始化
				//初始 或者 找到了l
				temp = j;  //存下当前的结点
			}
		}

		visited[temp] = 1;
		//找到了当前符合的结点，使用该节点去更新路径
		for (int j = 1; j <= n; ++j) {
			dist[j] = min(dist[j], dist[temp] + graph[temp][j]);
		}

	}
	//此时dist已经全部更新完毕
	int ans = *max_element(dist.begin()+1, dist.end());
	return ans == 111 ? -1 : ans;
}


/*
对于前K个最小数
无顺序返回的话
*/
class smallest_K {
public:
	vector<int> smallestK(vector<int>& arr, int k) {
		vector<int> ans(k);
		return ans;
	}
};


//502 IPO    initial public offering
//从给定项目中选择 最多 k 个不同项目的列表，以 最大化最终资本
int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
	int n = capital.size();
	//capital不有序
	vector<int> idx(n);
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(), [&capital](const int& a, const int& b) {
		return capital[a] < capital[b];//按照需要的启动资金从小到大进行排序
	});
	/*
	按照贪心问题进行求解，首先需要找出，当前的资本w能够进行哪些项目，选出其中利润最高的，花费p启动该项目
	找到利润最高的之后，完成该项目，若有利润q，那么当前的资本w为  w_new = w+q  (q为净利润
	对于之前钱不够而无法进行开展的项目，按照所需启动资金由小到大排序，每当资金满足时候，取出小顶堆顶部项目
	*/
	priority_queue<int> can_make;
	for (auto a : idx) {
		while(!can_make.empty() && capital[a]>w){  //当前的不能做了，卖了能做的赚钱
			w = w + can_make.top();  //选取能做里面最大利润的,卖一个
			can_make.pop();
			if (--k == 0) {
				return w;
			}
		}
		if(can_make.empty() && capital[a] > w) {
			//能做的项目为空并且也没有可以进行的项目了
			break;
		}
		//还能做
		can_make.push(profits[a]);
	}
	while (!can_make.empty() && k--) {
		w += can_make.top();
		can_make.pop();
	}
	return w;
}


class Solution68 {
public:
	string move_space(vector<string> s, int w_count, int w_len, int maxWidth) {
		int space_num = w_count - 1;
		int space = maxWidth - w_len - space_num;
		string rt = "";
		if (space % space_num) {
			int a = space / space_num;
			int b = space % space_num;
			for (int i = 0; i < s.size() - 1; ++i) {
				if (i == 0) {
					rt = rt + s[i];
					for (int j = 0; j < b; ++j) {
						rt += " ";
					}
				}
				else {
					rt = rt + s[i];
					for (int j = 0; j < a; ++j) {
						rt += " ";
					}
				}
			}
			rt += s[s.size() - 1];
		}
		return rt;
	}

	vector<string> fullJustify(vector<string>& words, int maxWidth) {
		vector<string> ans;
		int word_length = 0;
		int row_index = 0;
		int word_count = 0;
		int space_count = 0;
		vector<string> temp;
		int is_last = 0;

		for (int i = 0; i < words.size(); ++i) {
			int surplus = maxWidth - word_length - space_count; //还有多少空位         
			if (surplus < words[i].size()) {
				//放不下了，移到下一行
				//移动前先调整这一行的空格间距               
				//调整空格
				string str_rt = move_space(temp, word_count, word_length, maxWidth);
				ans.push_back(str_rt);
				temp.clear();
				//调整完毕，开始下一行
				row_index++;
				temp.push_back(words[i]);
			}
			else {
				//这一行还放得下
				word_count++;
				word_length += words[i].size();
				space_count = word_count - 1;
				temp.push_back(words[i]);
			}
		}
		//最后一行已经放进去了，最后一行进行左对齐
		string last = "";
		for (int i = 0; i < temp.size(); ++i) {
			if (i != temp.size() - 1) {
				last = last + temp[i] + " ";
			}
			else {
				last += temp[i];
				while (last.size() < maxWidth) {
					last += " ";
				}
				ans.push_back(last);
			}
		}
		return ans;
	}
};


// ======================================数位dp===============================================
class digitdp {
private:
	int n = 32;
	vector<vector<int>> dp;
public:
	int getLength(int n) {
		for (int i = 32; i > 0; --i) {
			if (((n >> i) & 1) == 1)
				return i;
		}
	}
			
	int findIntegers(int n) {
		//10^9 暴力法必死
		//二进制表示不包含连续1，双1就break
		//dp吧
		/*
		dp[i,j]表示二进制长度为i，最高位为j时，有多少种方法
		dp[i,j] = dp[i,j-1](可否补1)+(一定可以补0)
		*/
		//记忆化搜索应该也可以
		dp.resize(n, vector<int>(n, 0));
		dp[1][0] = 1, dp[1][1] = 1;
		for (int i = 1; i < n; ++i) {
			dp[i + 1][1] = dp[i][0];
			dp[i + 1][0] = dp[i][0] + dp[i][1];
		}
		int len = getLength(n);
		int ans = 0, pre = 0;
		for (int i = len; i >= 0; i--) {
			//判断当前位
			int now = ((n >> i) & 1);
			if (now == 1) {
				ans += dp[i + 1][0];
			}
			if (pre == 1 && now == 1)break;
			pre = now;
			if (i == 0)ans++;
		}
		return ans;
	}
};
//===========================================================================================




//678. 有效的括号字符串
//给定一个只包含三种字符的字符串：（ ，） 和 *
//*可以是(或者 ）
bool checkValidString(string s) {
	stack<char> left;
	stack<char> star;

	for (int i = 0; i < s.size(); ++i) {
		if (s[i] == '(') {
			left.push(i);
			//存取下标很巧妙，而不是直接存储括号和*
			//这样方便统计全局，考虑次序
		}
		else if (s[i] == '*') {
			star.push(i);
		}
		else { // == ')'
			if (!left.empty()) {
				left.pop();
			}
			else if (!star.empty()) {
				star.pop();
			}
			else {
				return false;
			}
		}
	}
	while (!left.empty() && !star.empty()) {
		if (left.top() > star.top()) {
			star.pop();
			continue;
		}
		else {   //只有出现在左括号右侧的*才能替代对应的右括号
			left.pop();
			star.pop();
		}
	}
	return left.empty();
}


//162. 寻找峰值
//找出数组中的任意一个峰值的索引，峰值即当前位置大于左右两侧
// nums[-1] = nums[n] = -∞
//时间复杂度控制在 O(log(n))    二分
int findPeakElement(vector<int>& nums) {
	//二分
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int mid = l + (r - l) / 2;
		if (nums[mid] < nums[mid + 1]) {
			//爬升
			l = mid + 1;
		}
		else {
			r = mid;
		}
	}
	return l;
}



//212. 单词搜索 II]
/*
	若在每个位置都进行 words中每个词进行深搜 ，会超时
	优化： 将word中的所有词存入前缀树，对board每个位置进行依次深搜，每次边深搜边和前缀树做对比，这样无需重复遍历word
*/
class trie1 {
private:
	
public:
	string word;
	unordered_map<char, trie1*> mp;
	trie1() :word("") {}

	void insert_words(trie1* root, string& words) {
		trie1* node = root;
		for (auto& a : words) {
			if (!node->mp.count(a)) {
				node->mp[a] = new trie1();
			}
			node = node->mp[a];
		}
		// word is end
		node->word = words;
	}
};


////212. 单词搜索 II]
class trie_example1 {
private:
	vector<vector<int>> steps{ {1,0},{-1,0},{0,1},{0,-1} };
	vector<vector<int>> visited;
public:
	bool dfs(vector<vector<char>>& board, vector<vector<int>>& visited, int i , int j, trie1* root, set<string>& temp) {
		int m = board.size();
		int n = board[0].size();
		if (!(root->mp.count(board[i][j]))) {
			return false;
		}

		root = root->mp[board[i][j]];
		if (root->word.size() > 0) {
			temp.insert(root->word);  // 不要直接return  ，比如 "oa"   "oaa", 都含有oa
			
		}

		visited[i][j] = 1;
		for (int s = 0; s < 4; ++s) {
			int new_x = i + steps[s][0];
			int new_y = j + steps[s][1];
			if (new_x >= 0 && new_x < m && new_y >= 0 && new_y < n) {
				if(!visited[new_x][new_y])
					dfs(board,visited, new_x, new_y, root, temp);
			}
		}
		visited[i][j] = 0;
		return true;
	}

	vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
		int m = board.size();
		int n = board[0].size();
		visited.resize(m, vector<int>(n, 0));
		set<string> temp;
		vector<string> ans;
		trie1 tr;
		trie1* root = new trie1();

		for (auto& a : words) {
			tr.insert_words(root, a);
		}
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				dfs(board,visited, i, j, root, temp);
			}
		}
		for (auto& b : temp) {
			ans.push_back(b);
		}
		return ans;
	}
};


//统计k进制下，1~m中1出现的个数   如 11有两个1
int count1(int m, int k) {
	//1~m,   k进制
	int count = 0;
	for (int i = 1; i <= m; i *= k) {
		int a = m / i, b = m % i;
		count += (a + k - 2) / k * i + (a % k == 1) * (b + 1);
	}
	return count;
}


// 36. 有效的数独
/*主要是判断一个数当前行、列有无重复。  每个小矩阵内是否有重复*/
// 大矩阵由九个小矩阵组成。
bool isValidSudoku(vector<vector<char>>& board) {
	vector<tuple<unordered_map<int, int>, unordered_map<int, int>, unordered_map<int, int>>> vec(9);
	//0~8分别表示1~9 row  col 
	//0~8分别表示每个矩阵
	//对于下标为(i,j)的数字， 其所属矩阵编号为 (i/3)*3+(j/3)
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			int idx = (i / 3) * 3 + (j / 3);
			if (board[i][j] == '.')
				continue;
			if (get<0>(vec[i]).count(board[i][j]) || get<1>(vec[j]).count(board[i][j]) || get<2>(vec[idx]).count(board[i][j])) {
				return false;
			}
			else {
				get<0>(vec[i])[board[i][j]]++;  //row
				get<1>(vec[j])[board[i][j]]++;   //col
				get<2>(vec[idx])[board[i][j]]++;  //矩阵
			}
		}
	}
	return true;
}


int cnt(int n) {
	int count1 = 0, count0 = 0;
	int num1 = 0, num0 = 0;
	int i = 0;
	while(n) {
		int first = n & 1;
		if (first == 1) {
			if (n >> 1) {
				num1 |= (1<<i);
				count1++;
			}
		}
		else {
			num0 |= (1<<i);
			count0++;
		}
		n >>= 1;
		i++;
	}
	
	int ans = 0;
	if (num1 <= num0) { //加
		ans = count1+1;
	}
	else { //减
		ans = count0+1;
	}
	cout << ans << endl;
	return ans;
}


//300. 最长递增子序列
//给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
//index不一定连续
int lengthOfLIS_greedy(vector<int>& nums) {
	//方法一 O(n^2) dp  之前写的有（见最长严格递增子序列）
	//此处使用方法二：贪心 + 二分
	return 0;
}


//数位DP
/*windy数
[L,R]范围内满足 不含前导零并且相邻两个数字之差至少为2的正整数
1、 边界转换  ans[L,R] = ans[1,R] - ans[1, L-1]
*/
//方法1， 可使用dfs
/*
需要记录的状态：
	当前枚举的位置  <--好维护
	前一位的数字    <--好维护
	当前位置可以填入的数  		
*/
class windy
{
private:
	vector<vector<LL>> dp;
	vector<LL> num;
public:
	LL dfs(LL pos, LL pre_num, int limit, int pre_zero) {
		//limit 表示是否有最高位的限制
		//pre_zero判断前导0，表示前面是否全为0
		//如果limit为1，那么枚举的数字必须小于最大数pos位置的数， limit为0，则表示可以取0-9任意数
		if (pos == 0) {
			return 1;  //结束标志
		}
		if (!limit && !pre_zero && dp[pos][pre_num] != -1) {
			// 没有限制，没有前导0 并且已经访问过
			return dp[pos][pre_num];
		}
		int max_num;
		if (limit) {
			//卡边界
			max_num = num[pos];
		}
		else {
			max_num = 9;
		}

		int count = 0;
		for (int i = 0; i <= max_num; ++i) {
			if (abs(i - pre_num) >= 2) {
				if (pre_zero && (i == 0)) {
					count += dfs(pos - 1, -2, limit && (i == max_num), 1);//如果有前导0并且现在加的还是0，下一位随意
					//limit && (i == max_num) 判断是否贴边走，若一直和最大值的数字相同，那么limit一直为1，否则有一个位置不同则limit=0
				}
				else {  //   !pre_zero || i!=0    没有前导零或者当前插入不是0
					count += dfs(pos - 1, i, limit && (i == max_num), 0);//如果没有前导0，正常搜
				}
			}
		}
		if (!limit && !pre_zero) { //没有最高位限制且没有前导0时记录结果
			dp[pos][pre_num] = count;
		}
		return count;
	}

	LL solve(LL x) {
		//边界转换
		int pos = 0;
		dp.resize(50, vector<LL>(10, -1)); //dp[i][j]表示 dp[pos][pre_num]
		num.resize(50, 0);  //考虑最长50位
		while (x) {
			num[++pos] = x % 10;
			x /= 10;
		}
		return dfs(pos, -2, 1, 1);
	}

	LL digdp(LL L, LL R) {
		LL a = solve(R);
		LL b = solve(L - 1);
		LL ans = a - b;
		cout << ans << endl;
		return ans;
	}

};



//112 路径总和1
//113 路径总和2
//437 路径总和3


//517. 超级洗衣机
int findMinMoves(vector<int>& machines) {
	int sum = accumulate(machines.begin(), machines.end(), 0);
	int n = machines.size();
	int each = (int)sum / n;
	if (sum % n != 0) {
		return -1;
	}
	for (auto& a : machines) {
		a = a - each;
	}
	int ans = 0;
	for (int i = 0; i < n - 1; ++i) {
		ans = max(ans, abs(machines[i]));
		ans = max(ans, machines[i + 1]);
		machines[i + 1] += machines[i];
	}
	return ans;
}

// 转换为十六进制
string toHex(int num) {
	if (num == 0) {
		return "0";
	}
	string replace = "0123456789abcdef", ans = "";
	while (num != 0 && ans.size() < 8) {  // 负数右移 左侧补1， 考虑32位，所以要对size做限制
		char temp = replace[num & 0xf];
		ans = temp + ans;
		num >>= 4;
	}
	return ans;
}

// 187. 重复的DNA序列
// 重复出现的长度为10的子序列
const int len = 10;
vector<string> findRepeatedDnaSequences(string s) {
	vector<string> ans;
	if (s.size() < 10) {
		return ans;
	}
	unordered_map<string, int> st;
	for (int i = 0; i < s.size() - len; ++i) {
		string window = s.substr(i, len);
		if (st.count(window) && st[window]<2) {
			ans.emplace_back(window);
		}
		st[window]++;		
	}
	return ans;
}


//352. 将数据流变为多个不相交区间
class SummaryRanges {
private:
	vector<vector<int>> ans;
	map<int, int> interval;  // 记录已有区间
	// map<int, int> 可以记录左右断点， 每次使用二分查找，  另外，map为有序字典
	//<left, right> 表示区间，  若一个数不在任何区间之中，那么记录为<val, val>
	// 底层是平衡树
public:
	SummaryRanges() {
		
	}

	void addNum(int val) {
		auto upper_val = interval.upper_bound(val); //大于val的区间
		auto val_pre = (upper_val == interval.begin()) ? interval.end() : prev(upper_val);  // 左侧小于val的区间
		if (val_pre != interval.end() && val <= val_pre->second && val >= val_pre->first) {  // val直接就在范围内
			return;
		}else{  // 不在范围内了
			//考虑左右是否有符合条件的区间
			int left_yes = (val_pre != interval.end() && val_pre->second + 1 == val);
			int right_yes = (upper_val != interval.end() && val + 1 == upper_val->first);
			if (left_yes && right_yes) {
				int left = val_pre->first;
				int right = upper_val->second;
				interval.erase(val_pre);
				interval.erase(upper_val);
				interval.emplace(left, right);
			}
			else if (left_yes && !right_yes) {
				interval[val_pre->first] = val;
			}
			else if (!left_yes && right_yes) {
				interval.emplace(val, upper_val->second);
				interval.erase(upper_val);
			}
			else {
				interval.emplace(val, val);
			}
		}
	}

	vector<vector<int>> getIntervals() {
		for (auto& a : interval) {
			ans.push_back({ a.first, a.second });
		}
		return ans;
	}
};


//273 整数转英文表示
class TransNumtoWord {
public:
	vector<string> singles = { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine" };
	vector<string> teens = { "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen" };
	vector<string> tens = { "", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };
	vector<string> thousands = { "", "Thousand", "Million", "Billion" };

	string numberToWords(int num) {
		if (num == 0) {
			return "";
		}

		std::function<void(string& temp, int three)> trans;
		trans = [&](string& temp, int three) {
			if (three == 0) {
				temp += singles[three];
			}
			else if (three < 10) {
				temp += singles[three] + " ";
			}
			else if (three < 20) {
				temp += teens[three%10] + " ";
			}
			else if (three < 100) {
				temp = temp + tens[three / 10] + " ";
				trans(temp, three % 10);
			}
			else {
				temp = temp + singles[three / 100] + " hundred ";
				trans(temp, three % 100);
			}
		};


		string ans = "";
		for (int i = 3, divisor = 1000000000; divisor >= 0 && i >= 0; i--, divisor /= 1000) {
			int digit = num / divisor;  // 高位到低位  三位获取
			if (digit) {
				num -= digit * divisor;   //获取余下位
				string temp;
				trans(temp, digit);
				temp = temp + thousands[i] + " ";
				ans += temp;
			}
		}
		return ans;
	}
};


//实现除法，  整数除法器( 额外添加一个保存余数)
int divide(int dividends, int divisors) {    // 被除数，  除数
	//被除数和除数均为 32 位有符号整数。
	//余数初始化为被除数， 保存在64位次寄存器，低四位   除数保存在64位寄存器，高四位
	long long dividend = dividends;
	long long divisor = divisors;
	int flag = ((dividend > 0) && (divisor > 0) || (dividend < 0) && (divisor < 0)) ? 0 : 1;
	long long new_divisor = (abs(divisor));  //高8位
	long long new_dividend = (abs(dividend));  //低8位,  余数初始化为被除数
	new_divisor <<= 32;
	long long quitient = 0;   //商寄存器  32位置0
	
	while(new_divisor>1) {
		new_dividend -= new_divisor;  // 新的余数
		if (new_dividend < 0) {
			new_dividend += new_divisor;
			quitient <<= 1;
		}
		else {
			quitient <<= 1;
			quitient ^= 1;
		}
		new_divisor >>= 1;
	}
	if (flag) {
		quitient = -1 * quitient;
	}
	return quitient;
}


//230. 二叉搜索树中第K小的元素
class BSTKthElement {
public:
	int kthSmallest(TreeNode* root, int k) {
		// 二叉搜索树的中序遍历是有序的，可以利用其特征进行第k小的值的查找
		/*
		//递归 
		std::function<int(TreeNode*, int&, int)> search = [&](TreeNode* root, int& index, int k)->int {
			if (!root) {
				return 0;
			}
			int l = search(root->left, index, k);
			index++;
			if (index == k) {
				return root->val;
			}
			int r = search(root->right, index, k);
			return l ? l : r;
		};
		int index = 0;
		return search(root, index, k);
		
		*/

		//迭代
		int index = 0;
		stack<TreeNode*> st;
		while (root || st.size() > 0) {
			while (root) {
				st.push(root);
				root = root->left;
			}
			//左边走到头
			TreeNode* temp = st.top();
			st.pop();
			index++;
			if (index == k) {
				return temp->val;
			}
			temp = temp->right;
			root = temp;
		}
		return -1;
	}
};


//2033. 获取单值网格的最小操作数
int minOperations(vector<vector<int>>& grid, int x) {
	vector<int> one_dim;
	for (auto& a : grid) {
		for (auto& b : a) {
			one_dim.push_back(b);
		}
	}
	sort(one_dim.begin(), one_dim.end());
	for (auto& a : one_dim) {
		if ((a - one_dim[0]) % x) {
			return -1;
		}
	}
	int mid1 = (int)(one_dim.size()) / 2;
	//int mid2 = (int)(one_dim.size()) / 2 - 1;  当只有一个元素时，不能用
	int ans = 0;
	for (auto& a : one_dim) {
		ans += abs(a - one_dim[mid1]);
	}
	return ans;

}


//211. 添加与搜索单词 - 数据结构设计
// 使用shared_ptr  创建前缀树
// 因为存在内存的开辟与释放问题，因此使用shared_ptr
class WordDictionary {
private:
	struct Trienode {
		bool isend;
		vector<std::shared_ptr<Trienode>> children;
		Trienode() :isend(false), children(26, nullptr) {};
	};
	std::shared_ptr<Trienode> root;

	bool find(std::shared_ptr<Trienode> ptr, string& temp, int index) {
		if (ptr != nullptr) {
			if (temp.size() == index) {
				return ptr->isend;
			}
			if (temp[index] == '.') {  //  . 可以表示任何一个字母。
				for (auto child : ptr->children) {
					if (find(child, temp, index + 1)) {
						return true;
					}
				}
			}
			else {
				return find(ptr->children[temp[index] - 'a'], temp, index + 1);
			}
		}
		return false;
	}

public:
	WordDictionary() :root(std::make_shared<Trienode>()) {}

	void addWord(string word) {
		auto temp = root;
		for (auto& w : word) {
			if (temp->children[w - 'a'] == nullptr) {
				temp->children[w - 'a'] = std::make_shared<Trienode>();
			}
			temp = temp->children[w - 'a'];
		}
		temp->isend = true;
	}

	bool search(string word) {
		return find(root, word, 0);
	}
};


// 453. 最小操作次数使数组元素相等
/*
给你一个长度为 n 的整数数组，每次操作将会使 n - 1 个元素增加 1 。返回让数组所有元素相等的最小操作次数。

* 纯数学题
假设目前数组总和为sum，我们需要移动次数为m，那么整体数组总和将会增加m * (n - 1)，这里的n为数组长度，最后数组所有元素都相等为x，于是有：
sum + m * (n - 1) = x * n     (1)
我们再设数组最小的元素为min_val，m = x - min_val​，即 ​x = m + min_val​带入(1)得：
m = sum - min_val * n​
*/

int minMoves(vector<int>& nums) {
	return accumulate(nums.begin(), nums.end(), 0) - nums.size() * (*min_element(nums.begin(), nums.end()));
}



// 240. 搜索二维矩阵 II
bool searchMatrix(vector<vector<int>>& matrix, int target) {
	int m = matrix.size();
	int n = matrix[0].size();
	for (int i = 0; i < m; ++i) {
		if (matrix[i][0]<target && matrix[i][n - 1]>target) {
			auto it = lower_bound(matrix[i].begin(), matrix[i].end(), target);
			if (it != matrix[i].end() && *it == target) {
				return true;
			}
		}
		else if (matrix[i][0] == target || matrix[i][n - 1] == target) {
			return true;
		}
		else {
			continue;
		}
	}
	return false;


	// 方法2 ，右上角开始遍历，类似于以右上角为根的二叉搜索树BST
	/*
		1  <- 3  <- 5  <-  7
		|     |     |      |
		10 <- 11 <- 16 <- 20
		|     |      |     |
		23 <- 30 <- 34 <- 60

		<- 表示左子树   | 表示右子树
	*/
	/*
	int m = matrix.size();
	int n = matrix[0].size();
	int r=0,c=n-1;
	while(r<m && c>0){
		if(matrix[r][c]<target){
			r++;
		}else if(matrix[r][c]>target){
			c--;
		}else{
			return true;
		}
	}
	return false;
	*/
}

// 240. 搜索二维矩阵 II
//  Z字搜索
/*
	取矩阵的左下角和右上角
	当点为(x,y)时，搜索以原矩阵左下角(m,n)为左下角，(x,y)为右上角的矩阵
	若matrix[x][y]>target, 那么y=y-1;   // 因为y的一整列递增，最小的值都大于target，那么整列都大于，直接删除该列，移动到更小列
	若matrix[x][y]<target, 那么x=x+1;   // 一整行递增，若当前行的 最大值小于target，那么该行均小于，直接删除该行，移动到更大行
*/
bool searchMatrixZ(vector<vector<int>>& matrix, int target) {
	int m = matrix.size(), n = matrix[0].size();
	int x = 0, y = n - 1;
	while (x < m && y >= 0) {
		if (matrix[x][y] == target) {
			return true;
		}
		if (matrix[x][y] > target) {
			--y;
		}
		else {
			++x;
		}
	}
	return false;
}


//22括号生成
class Solution22 {
public:
	int num;
	set<string> st;
	/*
	广度优先则自定义一个结构体，左右孩子分别为左右括号
	然后将stack换为que进行层次遍历
	*/

	void dfs(string temp, int left, int right) {
		if (left == num && right == num) {
			if (!temp.empty()) {
				st.emplace(temp);
			}
		}
		if (right > left) {
			return;
		}
		if (left < num) {
			dfs(temp + '(', left + 1, right);
		}
		if (right < num) {
			dfs(temp + ')', left, right + 1);
		}
	}
	vector<string> generateParenthesis(int n) {
		num = n;
		string temp = "";
		dfs(temp, 0, 0);
		vector<string> ans = vector<string>(st.begin(), st.end());
		return ans;
	}
};


// 301. 删除无效的括号
class Solution301 {
public:
	int l, r;
	set<string> st;
	void getParentheses(string& s, string temp, int index, int left, int right) {
		//结束
		if (left == l && right == r) {
			//cout<<left<<" "<<right<<endl<<endl;
			if (!temp.empty())
				st.insert(temp);
			return;
		}
		if (index == s.size()) {
			return;
		}
		if (s[index] != '(' && s[index] != ')') {
			temp += s[index];
			index++;
		}
		//约束
		if (left < l && s[index] == '(') {
			getParentheses(s, temp + '(', index + 1, left + 1, right);
		}
		if (right < left && s[index] == ')') {
			getParentheses(s, temp + ')', index + 1, left, right + 1);
		}

	}
	vector<string> removeInvalidParentheses(string s) {
		int left = 0, right = 0;
		int sumleft = 0, sumright = 0;
		for (auto& a : s) {
			if (a == '(') {
				left++;
				sumleft++;
			}
			else if (a == ')') {
				if (left > 0) {
					left--;  //符合要求的
				}
				else {
					right++;  //不符合要求的右括号
				}
				sumright++;
			}
		}
		// 最终若left不为0， 那么剩余的left为不满足要求的
		l = sumleft - left;  //满足的
		r = sumright - right; //满足的
		//获得了left和right的数量之后，进行括号生成， set去重
		string temp = "";
		getParentheses(s, temp, 0, 0, 0);
		vector<string> ans = vector<string>(st.begin(), st.end());
		return ans;
	}
};


// 869. 重新排序得到 2 的幂
class Solution869 {
public:
	vector<int> rec;   //记录每一位的使用情况
	bool isPower2(int n) {
		return (n & (n - 1)) == 0;
	}

	bool backtrack(string& num, int index, int temp) {
		if (index == num.size()) {
			return isPower2(temp);

		} //结束

		//不符合要求
		for (int i = 0; i < num.size(); ++i) {  //逐位
			if (rec[i] || (temp == 0 && num[i] == '0')) {   // 访问过  或者  此时为前导0
				// 可以参考全排列II， 再添加一个条件，避免重复计算
				// if (rec[i] || (temp == 0 && num[i] == '0') || (i>0 && !rec[i-1] && (num[i] == num[i-1])))
				continue;
			}
			rec[i] = 1;
			if (backtrack(num, index + 1, temp * 10 + num[i] - '0'))
				return true;
			rec[i] = 0;
		}
		return false;
	}

	bool reorderedPowerOf2(int n) {
		string num = to_string(n);
		sort(num.begin(), num.end());
		rec.resize(num.size());
		return backtrack(num, 0, 0);
	}
};


/*
全排列系列

46. 全排列I
	数组不含重复数字，返回其全排列

47. 全排列 II
	数组包含重复数字。 返回其全排列
	permuteUnique...
*/
class SolutionPermute {
public:
	// 1
	vector<vector<int>> ans1;

	//2
	vector<int> visited;
	vector<vector<int>> ans2;


	void backtrack1(vector<int>& num, int idx) {
		/*
		* 不保证字典序
			若输入1 2 3
			那么输出依次为  1 2 3   1 3 2   2 1 3  2 3 1   3  1  2   3  2  1 
			 1 2 3到1 3 2时， idx为1， i为2  实现swap   后还原至 1 2 3
		*/
		int len = num.size();
		if (idx == len) {
			ans1.emplace_back(num);
			return;
		}
		for (int i = idx; i < len; ++i) {
			// idx 表示开始位置，通过交换来实现 不同组合
			// 此处的swap方法也可以采用一个额外的数组，对未选取的数字进行位置标记，避免重复选取
			swap(num[i], num[idx]);
			backtrack1(num, idx + 1);
			swap(num[i], num[idx]);
		}
	}

	vector<vector<int>> permuteI(vector<int>& nums) {
		backtrack1(nums, 0);
	}

	//2222222===============================================================================
	void backtrace2(vector<int>& nums, vector<int>& temp, int idx) {
		if (idx == nums.size()) {
			ans2.emplace_back(temp);
			return;
		}

		for (int i = 0; i < (int)nums.size(); ++i) {
			if (visited[i] || (i > 0 && !visited[i - 1] && nums[i - 1] == nums[i])) {  // 最重要的地方
				continue;
			}
			// 若有相同数字，前一个数字的情况考虑过后，才会考虑第二个重复的数字
			// 那么i>0 && !visited[i-1] && nums[i-1] == nums[i] 就避免了相同数字重复情况的出现
			// 因为 !visited[i-1] 表示前一个数已经遍历过结束了
			temp.push_back(nums[i]);
			visited[i] = 1;
			backtrace2(nums, temp, idx + 1);
			visited[i] = 0;
			temp.pop_back();
		}
	}
	vector<vector<int>> permuteUnique(vector<int>& nums) {
		int n = nums.size();
		visited.resize(n, 0);
		sort(nums.begin(), nums.end());
		vector<int> temp;
		backtrace2(nums, temp, 0);
		return ans2;
	}

};


// 500. 键盘行
//使用正则表达式
vector<string> findWords(vector<string>& words) {
	regex pattern("(^((A|a)|(S|s)|(D|d)|(F|f)|(G|g)|(H|h)|(J|j)|(K|k)|(L|l))+$)|\
(^((Z|z)|(X|x)|(C|c)|(V|v)|(B|b)|(N|n)|(M|m))+$)|\
(^((Q|q)|(W|w)|(E|e)|(R|r)|(T|t)|(Y|y)|(U|u)|(I|i)|(O|o)|(P|p))+$)");
/*
srring换行时， \要紧贴行尾，下一行开头前不能有空格或tab
*/
	int tag = 0;
	smatch mt;
	vector<string> ans;
	for (auto& word : words) {
		bool ret = regex_match(word, mt, pattern);
		if (ret) {
			ans.emplace_back(word);
		}
	}
	return ans;
}


// 166. 分数到小数
string fractionToDecimal(int numerator, int denominator) {
	string ans = "";
	//计算整数部分
	LL a = numerator, b = denominator;
	if (a * b < 0) {
		ans += "-";
	}
	a = abs(a);
	b = abs(b);
	LL before = a / b;
	string bf = to_string(before);
	ans = ans + bf + ".";
	a %= b;
	unordered_map<int, int> mp;
	int idx = ans.size();
	//小数部分
	while (a) {
		mp.insert({ a, idx });  //出现位置
		a *= 10;
		int temp = a / b;
		ans += to_string(temp);
		a %= b;
		if (mp.count(a)) {
			int pre = mp[a];
			ans.insert(ans.begin() + pre, '(');
			ans += ")";
			break;
		}
		idx++;
	}
	
	return ans;
}

class Solution42 {
public:
	// 42. 接雨水
	int trap(vector<int>& height) {
		// 左右两侧分别找出高于当前位置的最大值
		// 算容量时，按照  min(左最大，右最大)
		// 边界按0处理
		int n = height.size();
		vector<pair<int, int>> vec(n, { 0,0 });
		int lmax = 0, rmax = 0;
		for (int i = 0; i < n; ++i) {
			vec[i].first = lmax;
			lmax = max(lmax, height[i]);
		}
		for (int i = n - 1; i >= 0; --i) {
			vec[i].second = rmax;
			rmax = max(rmax, height[i]);
		}
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			int temp = min(vec[i].first, vec[i].second);
			if (temp < height[i]) {
				continue;
			}
			else {
				int aa = (temp - height[i]);
				ans += aa;
			}
		}
		return ans;
	}

	// 单调栈
	// 时间复杂度O(n)
	int trap_case2(vector<int>& height) {
		/*
			栈中保存的为下标，按照从栈底到栈顶，下标对应的height值递减来存储
			若遇到比当前栈顶大的值，则出栈直到满足递减 或 栈为空
		*/
		int n = height.size();
		int ans = 0;
		stack<int> st;
		for (int i = 0; i < n; ++i) {
			while (!st.empty() && height[i] > height[st.top()]) {
				int mid = st.top();
				st.pop();
				if (st.empty()) {
					break; // 至少存两个数才能满足蓄水条件
				}
				int left = st.top();
				int len = i - left - 1;
				ans += (len * (min(height[left], height[i]) - height[mid]));
			}
			st.push(i);
		}
		return ans;
	}
};

// 1218. 最长定差子序列
int longestSubsequence(vector<int>& arr, int difference) {
	unordered_map<int, int> ump;
	// ump 存储当前数字的等差子序列长度
	int ans = 0;
	for (auto& a : arr) {
		if (ump.count(a - difference)) {
			ump[a] = 1 + ump[a - difference];

		}
		else {
			ump[a] = 1;
		}
		ans = max(ans, ump[a]);
	}
	return ans;
}



// 629. K个逆序对数组
class Solution629 {
public:
	int MOD = 1e9 + 7;
	int kInversePairs(int n, int k) {
		/*
			若当前有1-4（i-1个数据）插入的为5(插入后有i个)，那么有以下几种插入方式
			5xxxx
				此时多产生了i-1个逆序对，假设此时共有j个逆序对，那么  f[i][j] = f[i-1][j-(i-1)] 
			x5xxx
				f[i][j] = f[i-1][j-(i-2)]
			xx5xx
				f[i][j] = f[i-1][j-(i-3)]
			xxx5x
				f[i][j] = f[i-1][j-(i-4)]
			xxxx5
				f[i][j] = f[i-1][j-(i-5)]
		*/
		vector<vector<int>> dp(n + 1, vector<int>(k+1));
		for (int i = 1; i <= n; ++i) {  //  数字个数
			for (int j = 0; j <= k; ++j) {  // 此时有多少个逆序对
				for (int h = 0; h <= min(i - 1, j); ++h) {  //  前有i-1个数字， j代表当前逆序对数量  
					dp[i][j] = (dp[i][j] + dp[i - 1][j - h]) % MOD;
				}
			}
		}
 	}

	//优化
	int kInversePairs_optim(int n, int k) {
		return 0;
	}
};


// trie  改用shared_ptr+unordered_map
// 给trie每个结点一个值， 创建sum函数， sum(prefix)输出所有以该prefix为前缀的字符串的和
class Trie_shared_unordered {
private:
	struct re_trie {
		unordered_map<char, std::shared_ptr<re_trie>> children;
		bool isend;
		int sum;
	};
	void insert(std::shared_ptr<re_trie> root, string word, int index = 0) {  //传入参数时候，可选当前字符串的sum
		if (index == word.size()) {
			root->isend = true;
			return;
		}
		if (!root->children.count(word[index])) {
			root->children[word[index]] = std::make_shared<re_trie>();
		}
		//此处有字符，把以该字符为前缀的所有字符串sum加起来
		//root->sum += sum;
		insert(root->children[word[index]], word, index + 1);
	}
	bool find(std::shared_ptr< re_trie> root, string word, int index = 0) {
		if (index == word.size()) {
			if (root->isend)
				return true;
			else
				return false;
		}
		if (!root->children.count(word[index])) {
			return false;
		}
		find(root->children[word[index]], word, index + 1);
	}

};


//318 最大单词长度乘积
int maxProduct(vector<string>& words) {
	int n = words.size();
	//最大26位，int存储
	//arr[i]每一位存储的是一个单词中出现的字母状况
	vector<int> arr(n, 0);
	for (int i = 0; i < n; ++i) {
		for (auto aa : words[i]) {
			arr[i] |= (1 << (aa - 'a'));
		}
	}
	for (int i = 0; i < n;++i) {
		cout << arr[i] << endl;
	}
	int ans = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			if (arr[i] & arr[j]) {  //真则有相同
				continue;
			}
			else {
				ans = max((int)words[i].size() * (int)words[j].size(), ans);
			}
		}
	}
	return ans;
}


// 563 二叉树的坡度
// 不喜欢做树的递归
struct TreeNode563 {
	int val;
	TreeNode563* left;
	TreeNode563* right;
	TreeNode563() : val(0), left(nullptr), right(nullptr) {}
	TreeNode563(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode563(int x, TreeNode563* lc, TreeNode563* rc) : val(x), left(lc), right(rc) {}
};

class Solution563 {
public:
	int ans = 0;
	int dfs(TreeNode563* root) {
		if (root == nullptr) {
			return 0;   // 从底向上 
		}
		int lroot = dfs(root->left);
		int rroot = dfs(root->right);
		ans += abs(lroot - rroot);
		return root->val + lroot + rroot;
	}

	int findTilt(TreeNode563* root) {
		dfs(root);
		return ans;
	}
};


// 397. 整数替换
class Solution397 {
public:
	int integerReplacement(int n) {
		if (n == 1) {
			return 0;
		}
		else if ((n & 1)) {  //奇
		   //return 2+min(integerReplacement((n-1)/2), integerReplacement((n+1)/2))
			return 2 + min(integerReplacement(n / 2), integerReplacement(n / 2 + 1));  //2代表一次+1或-2，和一次/2
			//不要直接写+1， 2147483647会越界
		}
		else {
			return 1 + integerReplacement(n >> 1);   //此处的1代表一次 /2
		}
	}

	int integerReplacement_case2(int n) {
		long nn = n;
		int ans = 0;
		while (nn != 1) {
			//因为可以用 n+1 或者n-1来替换，考虑该用哪种替换
			// n-1去掉了最后一位1。 
			// 若此时后两位为01，n+1转变为10，比n-1复杂。 
			// 若为11， n+1变为00,.。。
			if ((nn & 3) == 3 && nn != 3) {  // ......11 
				nn++;
			}
			else if ((nn & 1) == 1) {  // 000...011 或 01
				nn--;
			}
			else {
				nn >>= 1;
			}
			ans++;
		}
		return ans;
	}
};



// 594. 最长和谐子序列
/*
	最长子数组，最大值和最小值之间的差别 正好是 1
*/
class Solution594 {
public:
	int findLHS_case1(vector<int>& nums) {
		// case 1   排序数组，找出相邻元素，差值为1的最长序列
		sort(nums.begin(), nums.end());
		int n = nums.size();
		int begin = 0;
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			while (nums[i] - nums[begin] > 1) {
				begin++;
			}
			if (nums[i] - nums[begin] == 1) {
				ans = max(ans, i - begin + 1);
			}
		}
		return ans;
	}

	int findLHS_case2(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		unordered_map<int, int> mp;   //hash map存储数字出现次数
		for (auto& a : nums) {
			mp[a]++;
		}
		int ans = 0;
		for (int i = 0; i < (int)nums.size(); ++i) {
			if (mp.count(nums[i]+1)) {
				ans = max(ans, mp[nums[i]] + mp[nums[i + 1]]);
			}
		}
		return ans;
	}
};



//559. N 叉树的最大深度
// N叉树以层次遍历 数组形式给出
class Node559 {
public:
	int val;
	vector<Node559*> children;

	Node559() {}

	Node559(int _val) {
		val = _val;
	}

	Node559(int _val, vector<Node559*> _children) {
		val = _val;
		children = _children;
	}
};
int maxDepth(Node559* root) {
	if (root == nullptr) {
		return 0;
	}
	int max_dep = 0;
	for (int i = 0; i < root->children.size(); ++i) {
		max_dep = max(max_dep, maxDepth(root->children[i]));
	}
	return max_dep + 1;
}

// 方法2  使用层序遍历
int maxDepth_case2(Node559* root) {
	if (root == nullptr) {
		return 0;
	}
	queue<Node559*> que;
	que.push(root);
	int depth = 0;
	while (!que.empty()) {
		depth++;
		int len = que.size();
		while (len) {
			Node559* temp = que.front();
			que.pop();
			for (auto& a : temp->children) {
				que.push(a);
			}
			len--;
		}
	}
	return depth;
}


// 384. 打乱数组
// only write shuffle
vector<int> shuffle(vector<int> temp) {
	int n = temp.size();
	for (int i = n - 1; i >= 0; --i) {
		int j = rand() % (i + 1);
		swap(temp[j], temp[i]);
	}
	return temp;
}


// 5930. 两栋颜色不同且距离最远的房子
//O(n)的解法，有点巧妙
int maxDistance(vector<int>& colors) {
	int n = colors.size();
	if (colors[0] != colors[n - 1]) {
		return n - 1;
	}
	int ans = 0;
	for (int i = 1; i < n; ++i) {
		if (colors[i] != colors[0]) {
			ans = max(ans, i);
		}
	}
	for (int j = n - 1; j >= 0; --j) {
		if (colors[j] != colors[n - 1]) {
			ans = max(ans, n - j);
		}
	}
	return ans;

	/*
	O(n^2)解法
	int n = colors.size();
		int res = 0;   // 两栋颜色不同房子的最远距离
		// 遍历两栋房子下标并维护最远距离
		for (int i = 0; i < n; ++i){
			for (int j = i + 1; j < n; ++j){
				if (colors[i] != colors[j]){
					res = max(res, j - i);
				}
			}
		}
		return res;
	*/
}


// 859. 亲密字符串
// 可以通过交换 s 中的两个字母得到与 goal 相等的结果，就返回 true ；否则返回 false 
//  "aa"和"aa"是满足的  ， "ab" "ba"是不满足的
/* 若满足条件， 首先
*	s和goal长度相同
*	i!=j  s[i] == goal[j]  && s[j] == goal[i], 此时又有两种情况
*		s == goal, 此时s[i]==goal[j]==s[j]==goal[i]
*		s != goal, 此时s[i]==goal[j]  !=   s[j]==goal[i]
*/
bool buddyStrings(string s, string goal) {
	if (s.size() != goal.size()) {
		return false;
	}
	if (s == goal) {
		// 判断s[i]==goal[j]==s[j]==goal[i] ?
		vector<int> vec(26, 0);
		for (auto& a : s) {
			vec[a - 'a']++;
			if (vec[a - 'a'] > 1) {
				return true;
			}
		}
		return false;
	}
	else {  // s!=goal 至少一个位置不同，找   s[i]==goal[j]  且   s[j]==goal[i]
		int first = -1, second = -1;
		int n = s.size();
		for (int i = 0; i < n; ++i) {
			if (s[i] != goal[i]) {
				if (first == -1) {
					first = i;
				}
				else if (second == -1) {
					second = i;
				}
				else {
					return false;
				}
			}
		}
		return second != -1 && (s[first] == goal[second]) && (s[second] == goal[first]);
		//second!=-1说明只有一个位置不同
	}
}

//5186. 区间内查询数字的频率
class RangeFreqQuery5186 {
private:
	unordered_map<int, vector<int>> mp;
public:
	RangeFreqQuery5186(vector<int>& arr) {
		for (int i = 0; i < (int)arr.size(); ++i) {
			mp[arr[i]].emplace_back(i);
		}
	}

	//自己实现一下 upper_bound 和 lower_bound
	int query(int left, int right, int value) {
		vector<int>::iterator l = lower_bound(mp[value].begin(), mp[value].end(), left);
		vector<int>::iterator r = upper_bound(mp[value].begin(), mp[value].end(), right);
		return r - l;
	}
};

/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery* obj = new RangeFreqQuery(arr);
 * int param_1 = obj->query(left,right,value);
 */




// 423. 从英文中重建数字
// 有趣的题目， 需要在哈希表进行查找时进行优化
string originalDigits(string s) {
	unordered_map<char, int> mp;
	for (auto& a : s) {
		mp[a]++;
	}
	// 先使用贪心
	// 最难的是如何确定每个单词的个数，例如找"one"的个数，若依次查找 'o'、'n'、'e'， 那么可能会找到two中的o， seven中的n， eight中的e
	// 这样没法确定是否包含"one"
	// 因此要给每个  被查找的对象一个特殊标记
	/*
	可以看到，对于  zero->z为其所特有  two->w， four -> r   six->x   eight->g 
	对剩下的  one   three   five  seven   nine   可以取其余任意特殊标记  
			one -> o  three -> e   five -> i  seven -> v  nine -> n

	最终可以把字母转换为    zero  one  wto  ethre   rfou  ifve  xsi  vseen  geiht  nine
	*/
	// vector<string> vec{ "zero","one","two","three","four","five","six","seven","eight","nine" };
	vector<string> vec{ "zero","geiht","wto","xsi","seven","hetre","vfie","four","one","inne" };
	vector<int> idx{ 0,8,2,6,7,3,5,4,1,9 };
	string ans = "";
	for (int i = 0; i < 10; ++i) {
		while (mp[vec[i][0]] != 0) {
			ans += (idx[i] + '0');
			for (auto& a : vec[i]) {
				mp[a]--;
			}
		}
	}
	sort(ans.begin(), ans.end());
	return ans;
}



//5933. k 镜像数字的和  hard
//一个 k 镜像数字 指的是一个在十进制和 k 进制下从前往后读和从后往前读都一样的 没有前导 0 的 正 整数
// 给你进制 k 和一个数字 n ，请你返回 k 镜像数字中 最小 的 n 个数 之和
class KMirrorNum {
public:
	/* 本题重点在于
	* 1、已知一个十进制对称数，求下一个十进制对称数
	* 2、判断一个字符串是否对称
	* 3、计算十进制转换到k进制
	*/
	long long kMirror(int k, int n) {
		
	}
};


//438. 找到字符串中所有字母异位词
vector<int> findAnagrams(string s, string p) {
	if (s.size() < p.size()) {
		return {};
	}
	int m = s.size(), n = p.size();
	vector<int> srec(26, 0);
	vector<int> prec(26, 0);
	vector<int> ans;
	for (int i = 0; i < m; ++i) {
		if (i < n) {
			prec[p[i] - 'a']++;
			srec[s[i] - 'a']++;
		}
		else {
			if (srec == prec) {
				ans.emplace_back(i - n);
			}
			srec[s[i - n] - 'a']--;
			srec[s[i] - 'a']++;
		}
	}
	if (srec == prec) {
		ans.emplace_back(m - n);
	}
	return ans;
}


// 400. 第 N 位数字
int findNthDigit(int n) {
	if (n < 10) {
		return n;
	}
	LL count = 10;
	int k = 0;
	LL last = 0;
	while (count <= n) {
		last = count;
		count = (LL)((k + 1) * pow(10, k + 1) - (pow(10, k + 1) - 1) / 9);
		++k;
	}
	// k位满足了
	LL new_n = 0;
	LL k_num = pow(10, k - 1) + (n - last - 1) / k;  // 从k位的最小值开始，找到n所在的数字
	int index = (n - last - 1) % k;
	return to_string(k_num)[index] - '0';
}


// //**********   bind    和    make_shared
class  A {
public:
	int add(int a, int b) {
		return a - b;
	}

	void doit() {
		std::cout << "func" << std::endl;
	}
};

 
// 519. 随机翻转矩阵
/*
	m*n的矩阵，对于随机产生其下标，肯定会想到拉直矩阵
	注意要产生不为0位置的下标，若使用rand()函数，其产生的是一个连续区间内的随机数字
	因此，若矩阵某一段为  0 0 0 1 1 0 1
	那么使用rand()无法避开值为1的地方
	故考虑变形矩阵， 变为 0 0 0 0 1 1 1 ， 可以想到，这是一种类似于swap的操作
  * 但是 ！！！ 因为每次调用flip()函数， 会返回当前随机产生的下标
  * 若 0 0 0 0 1 1 1  这次flip()后的结果，index = 2，使得数组变为 0 0 1 0 1 1 1 ， 返回{2/n, 2%n}
  * swap后， 变为 0 0 0 1 1 1 1， 此时，若再次flap（）， 也会产生index = 2， 违背了只产生非0位置的要求
  * 此次的index = 2， 实际上返回的是swap之前的 index = 3的位置。 
  ！！！ 因此，需要保留一次映射来记录每次虚拟swap的值

  0 1 2 3 4 5   若第一次生成4  那么有4->5的映射， 若下次生成2，那么有2->4的映射，但是4->5有，那么应该变为2->5
*/
class Solution519 {
private:
	int m;
	int n;
	unordered_map<int, int> mp;  // 存储拟swap的信息
	int total;
public:
	// Solution519(int m, int n) 使用二元矩阵的大小 m 和 n 初始化该对象
	Solution519(int m, int n) {
		this->m = m;
		this->n = n;
		this->total = m * n;
		srand(time(nullptr));
	}

	// int[] flip() 返回一个满足 matrix[i][j] == 0 的随机下标 [i, j] ，并将其对应格子中的值变为 1
	vector<int> flip() {
		int index = rand() % total;
		int row = index / n;
		int col = index % n;
		vector<int> ans;
		if (mp.count(index)) {  //  此位置是其他位置swap来的
			ans = { mp[index] / n, mp[index] % n };
		}
		else {
			ans = { row, col };
		}

		// 将该位置与最后一个位置交换
		if (mp.count(total - 1)) {  //值为0最后一个位置是其他位置交换而来
			mp[index] = mp[total - 1];
		}
		else {
			mp[index] = total - 1;
		}
		total--;
		return ans;
	}

	//void reset() 将矩阵中所有的值重置为 0
	void reset() {
		total = m * n;
		mp.clear();
	}
};


// 计数排序
//计数排序，不是基于元素比较，而是利用数组下标确定元素的正确位置
// 待排序列：9 3 5 4 9 1 2 7 8 1 3 6 5 3 4 0 10 9 7 9
//数组值：| 0 1 2 3 4 5 6 7 8 9 10 |
//次数值：| 1 2 1 3 2 2 1 2 1 4 1  |
//直接便利数组，输出数组元素的下标值，元素的值是几就输出多少次。
// 输出  0 1 1 2 3 3 3 4 4 5 5 6 7 7 8 9 9 9 10
void countSort(vector<int>& vec) {
	int min_ele = *min_element(vec.begin(), vec.end());
	int max_ele = *max_element(vec.begin(), vec.end());
	int interval = max_ele - min_ele + 1;
	vector<int> temp(interval, 0);
	for (auto& a : vec) {
		temp[a]++;
	}
	for (int i = 0; i < temp.size(); ++i) {
		int t = temp[i];
		while (t) {
			cout << i << endl;
			--t;
		}
	}
}

// 282. 给表达式添加运算符
//给定一个仅包含数字 0-9 的字符串 num 和一个目标值整数 target ，
//在 num 的数字之间添加 二元 运算符（不是一元）+、- 或 *  和  ** 拼接  1，2-》12**
//返回所有能够得到目标值的表达式
class Solution282 {
private:
	int len;
	LL target;
	vector<string> ans;
	string num;
public:
	// 回溯
	/*
		乘号*会改变运算顺序，若有 1+2+3， 若此时加个*并且后续为4
		那么式子变为 1+2+3*4  对于这个式子， 定义 sum = 1+2+3  则prenum = 3  (代表最后一位带符号的操作数)
		对于*后  可以得到结果为： (sum-prenum)+prenum*4 = (1+2+3-3)+3*4 = 15
	*/
	void dfs(string& path, LL sum, LL presum, int index) {
		if (index == len) {
			if (sum == target) {
				ans.emplace_back(path);
			}
			return;
		}
		string temp = "";
		for (int i = index; i < len; ++i) {
			temp += num[i];  // 拼接
			LL opt_num = stoll(temp);   //当前位置的数
			if (index == 0) {
				path += temp;
				dfs(path, opt_num, opt_num, i + 1);
				path.erase(path.size() - temp.size());
			}
			else {
				path = path + "+" + temp;
				dfs(path, sum + opt_num, opt_num, i +1);

				path.replace(path.size() - temp.size() - 1, 1, "-");
				dfs(path, sum - opt_num,  -opt_num, i +1);

				path.replace(path.size() - temp.size() - 1, 1, "*");
				dfs(path, sum - presum + presum * opt_num, presum * opt_num, i +1);
				path.erase(path.size() - temp.size() - 1);
			}
			if (temp == "0") {
				//前导0
				return;
			}
		}
	}

	//addOperators("2147483648", -2147483648);
	vector<string> addOperators(string num, int target) {
		this->len = num.size();
		this->target = target;
		this->num = num;
		string temp = "";
		dfs(temp, 0, 0, 0);
		return ans;
	}
};


// 快速幂法
// 50 pow(x,y)
/*
	若有一个求 x^77, 从左到右可以看到  x^1 -> x^2 -> x^4 -> x^9 -> x^19 -> x^38 ->x^77
	显然，若从左到右求，我们没法知道哪一步需要诚意 x
	但是从右到左求，显而易见，当为奇数(如77)时，低一位为 下取整 77/2 = 38
	当为偶数时，低一位为 38/2
*/
class Solution50 {
public:
	double QuickMul(double x, LL N) {
		if (N == 0) {
			return 1.0;
		}
		double y = QuickMul(x, N / 2);
		return N % 2 == 0 ? y * y : x * y * y;
	}

	double myPow(double x, int n) {
		LL N = n;
		return n >= 0 ? QuickMul(x, N) : 1 / QuickMul(x, -N);
	}
};


// 372. 超级次方
//你的任务是计算 ab 对 1337 取模，a 是一个正整数，b 是一个非常大的正整数且会以数组形式给出。
/*
	本题基于50 幂方法
	// 取模运算的法则
	(a+b)%c = (a%c+b%c)%c
	(a-b)%c = (a%c-b%c)%c
	(a*b)%c = (a%c*b%c)%c
*/
class Solution372 {
public:
	using LL = long long;
	static constexpr int MOD = 1337;
	// 快速幂的递归写法
	LL QuickPow(LL x, LL n) {
		LL ans = 1;
		while (n) {
			if (n % 2 == 1) {
				ans = (ans * x) % MOD;// 可以在这里MOD，防止乘法越界
			}
			x = (x % MOD) * (x % MOD) % MOD;
			n /= 2;
		}
		return ans;
	}

	// 此处c=1337
	int superPow(int a, vector<int>& b) {
		LL ans = 1;
		for (int i = b.size() - 1; i >= 0; --i) {
			ans = ans * QuickPow(a, b[i]) % MOD;
			a = QuickPow(a, 10);
		}
		return ans;
	}
};

// 1034. 边界着色
// 图论搜索题目   BFS/DFS
class Solution1034 {
private:
	vector<vector<int>> visited;
	vector<vector<int>> move{ {0,-1},{0,1},{-1,0},{1,0} };
	int color;
	int m;
	int n;
	int pre_color;
	vector<vector<int>> rec;
public:
	void dfs(vector<vector<int>>& grid, int row, int col) {
		visited[row][col] = 1;
		bool isborder = false;
		for (auto& a : move) {  // 当前位置往四个方向走
			int new_row = row + a[0];
			int new_col = col + a[1];
			// 判断边界(更新后的点为非正常情况，说明 [row, col] 为边界）     （正常情况：范围内，颜色为初始颜色)
			if (!(new_row >= 0 && new_row < m && new_col >= 0 && new_col < n && grid[new_row][new_col] == pre_color)) {
				isborder = true;
			}
			else if (!visited[new_row][new_col]) {  //正常情况且未访问过
				visited[new_row][new_col] = 1;
				dfs(grid, new_row, new_col);
			}
		}
		if (isborder) {  // 注意边界坐标的保留
			rec.push_back({ row,col });
		}
	}

	vector<vector<int>> colorBorder(vector<vector<int>>& grid, int row, int col, int color) {
		this->m = grid.size();
		this->n = grid[0].size();
		this->visited.resize(m, vector<int>(n, 0));
		this->color = color;
		this->pre_color = grid[row][col];
		dfs(grid, row, col);
		for (auto& a : rec) {
			grid[a[0]][a[1]] = color;
		}
		return grid;
	}
};


//689. 三个无重叠子数组的最大和
// 数组中求和最大的3个不重叠且长度为k（连续）的子数组
//dp不会
vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
	int n = nums.size();
	vector<int> ans;
	// 三个滑动窗口
	int sum1 = 0, maxsum1 = 0, maxsum1index = 0;
	int sum2 = 0, maxsum2 = 0, maxsum2index = 0, maxsum1indexrec = 0;
	int sum3 = 0, maxsum3 = 0;
	for (int i = 2 * k; i < n; ++i) {
		sum1 += nums[i - 2 * k];
		sum2 += nums[i - k];
		sum3 += nums[i];
		if (i >= 3 * k - 1) {
			if (sum1 > maxsum1) {
				maxsum1 = sum1;
				maxsum1index = i - 3 * k + 1;
			}
			if (sum2 + maxsum1 > maxsum2) {
				maxsum2 = sum2 + maxsum1;
				maxsum1indexrec = maxsum1index;
				maxsum2index = i - 2 * k + 1;
			}
			if (sum3 + maxsum2 > maxsum3) {
				maxsum3 = sum3 + maxsum2;
				ans = { maxsum1indexrec,maxsum2index,i - k + 1 };
			}
			sum1 -= nums[i - 3 * k + 1];
			sum2 -= nums[i - 2 * k + 1];
			sum3 -= nums[i - k + 1];
		}
	}
	return ans;

}


//851. 喧闹和富有
//返回一个整数数组 answer 作为答案，其中 answer[x] = y 的前提是，在所有拥有的钱肯定不少于 person x 的人中，person y 是最安静的人
//case1 dfs  从穷到富，记录图
//case2 拓扑排序
class Solution851 {
public:
	vector<int> loudAndRich_case1(vector<vector<int>>& richer, vector<int>& quiet) {
		int n = (int)quiet.size();
		vector<vector<int>> graph(n);

		for (auto& a : richer) {
			graph[a[1]].emplace_back(a[0]); // 形成一条边 a[1]->a[0], 从穷的走向富的，a[0]比较富,逆着走
		}

		vector<int> ans(n, -1);  // ans记录比自己富有的人中，安静度最小的那一个

		function<void(int)> dfs = [&](const int& x)->void {
			if (ans[x] != -1) {
				return;
			}
			// 先给自身一个值，可能自己入度为0
			ans[x] = x;
			for (auto& j : graph[x]) {  // 按顺序来，j比x富有
				dfs(j);  //往富的走，直到走至最后一个比自己富的
				// 更新的条件是，安静度比当前的安静度小
				if (quiet[ans[j]] <= quiet[ans[x]]) {  // 可以更新
					ans[x] = ans[j];
				}
			}
		};

		for (int i = 0; i < n; ++i) {
			dfs(i);
		}
		return ans;
	}

	vector<int> loudAndRich_case2(vector<vector<int>>& richer, vector<int>& quiet) {
		int n = (int)quiet.size();
		vector<vector<int>> graph(n);
		vector<int> indeg(n, 0);
		for (auto& a : richer) {
			graph[a[0]].emplace_back(a[1]);
			++indeg[a[1]];
		}

		queue<int> que;
		vector<int> ans(n);
		for (int deg = 0; deg < n; ++deg) {
			if (!indeg[deg]) {  //入度为0
				que.push(deg);
			}
			ans[deg] = deg;  //每个结点的ans先置为自身
		}
		while (!que.empty()) {
			int temp = que.front();  // 当前的富人结点
			que.pop();
			for (int k = 0; k < (int)graph[temp].size(); ++k) {
				//找比富人穷的结点
				int next_param = graph[temp][k];  // 比他穷
				if (--indeg[next_param]==0) {
					que.push(next_param);  //加入
				}	
				
				if (quiet[ans[next_param]] > quiet[ans[temp]]) {  // 用更小的答案来更新穷的结点的ans
					ans[next_param] = ans[temp];  // ans[temp]的 原因是小心多个结点的传递
				}
			}
		}
		return ans;
	}
};


//1610. 可见点的最大数目
// 返回视野中能看到的点的最大数量，人是可以在一定角度内旋转的
// 思路：计算出所有点的极角， 排序后，二分法或者滑动窗口
// 几何   坐标系   数学

//             !!!!!!!!!!  注意    atan2()的返回值范围为 [−π,π]，它的覆盖范围为 2π。
//  对于 点p, 以其为边界的可视区域为   p+angle,  若此时大于180度，  那么之前计算的atan值为负的点，应该是在可视范围，但是并没有计算在内，因此要将其转为正数， 加个2*pi(画图理解)
const auto M_PI = 3.14159265;
int visiblePoints(vector<vector<int>>& points, int angle, vector<int>& location) {   //angle传入的是角度， 按数值传入，如90度传入90
	vector<double> polar;
	int count = 0;  // 重合
	for (auto& point : points) {
		if (point[0] == location[0] && point[1] == location[1]) {
			count++;
			continue;
		}
		double degree = atan2(point[1] - location[1], point[0] - location[0]);
		polar.emplace_back(degree);
	}
	sort(polar.begin(), polar.end());
	int m = (int)polar.size();
	for (int i = 0; i < m; ++i) {
		polar.emplace_back(polar[i] + 2 * M_PI);
	}

	// 人的可视范围为 angle
	// case1 可以从边缘的点开始， 找到角度为angle时，有多少个点在
	// case2 滑动窗口简单一些
	double range = angle * M_PI / 180.0;

	int ans1 = 0;
	for (int i = 0; i < (int)polar.size(); ++i) {  //方法1
	   int j = upper_bound(polar.begin(), polar.end(), polar[i] + range) - polar.begin();
	   ans1 = max(ans1, j-i);
	}

	int ans2 = 0;
	int k = 0;
	for (int i = 0; i < (int)polar.size(); ++i) {   //方法2
		while (k < (int)polar.size() && polar[k] <= (polar[i] + range)) {
			++k;
		}
		ans2 = max(ans2, k - i);
	}
	//return ans1+count;

	return ans2 + count;
}


//针对419和997的并查集

class unionFind_419and997 {
private:
	// 记录一下连通集的数量
	int set_size;
	vector<int> root; //记录父节点
	vector<int> range; //记录每个集合的秩，即元素个数
public:
	unionFind_419and997(int n){  //结点数
		root.resize(n);
		range.resize(n, 1);
		this->set_size = n;
		for (int i = 0; i < n; ++i) {
			root[i] = i;  //父节点设置为自己
		}
	}

	void union_num(int x, int y) {  // 按照秩合并， 小->大
		int i = find_num(x);
		int j = find_num(y);
		if (i != j) {
			if (range[i] <= range[j]) {
				root[i] = j;
				range[j] += range[i];
			}
			else {
				root[j] = i;
				range[i] += range[j];
			}
		}
		--set_size;
	}

	int find_num(int x) { // find给个路径压缩
		//路径上的每一个元素都置为x的父节点
		return root[x] == x ? x : (root[x] = find_num(root[x]));
	}
};


//419. 甲板上的战舰
/*
	这题的题目描述很垃圾，大体意思就是找有几个连通集， 理解为战舰队列比较方便
	对于甲板上的战舰， 只能整行排列或者整列排列, 同时两个战舰队列之间要留空至少一格 ，举个例子（.表示空，X表示有）
	X  .  .  X
	.  .  .  X
	.  .  .  X
	.  .  .  X这就算有两个战舰队列 ，返回2
	  
	.  .  .  X
	X  X  X  X
	.  .  .  X   
	.  .  .  X 这就是不合规范的，因为未留空

	.  .  .  X
	X  X  .  X
	.  .  .  X
	.  .  .  X   返回2
*/
class Solution419 {
public:
	// 方法1，从左上角开始，  若当前位置有战舰，  那么只考虑他的左侧和上侧， 如果有一个地方有的话，说明是一行或者一列
	//  那么整体的战舰队列数目不增加
	int countBattleships_case1(vector<vector<char>>& board) {
		int ans = 0;
		int m = (int)board.size(), n = (int)board[0].size();
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (board[i][j] == 'X') {
					if (i > 0 && board[i - 1][j] == 'X') {
						continue;
					}
					if (j > 0 && board[i][j - 1] == 'X') {
						continue;
					}
					ans++;
				}
			}
		}
		return ans;
	}

	//方法2  统计连通集的数量，即 unionFind_419and997 中的  set_size
};


// 997. 找到小镇的法官
// 被997的测试用例骗了，不能拿并查集
// 就是看有没有所有的点直接指向同一个终点，此终点的出度为0
// 统计每个结点的出入度，出度为0且入读为n-1的，就为法官，代码不写



//475. 供暖器
// houses和heaters并不是按顺序写的
int findRadius(vector<int>& houses, vector<int>& heaters) {
	sort(houses.begin(), houses.end());
	sort(heaters.begin(), heaters.end());
	int max_radius = 0;
	// 如何去找离房屋最近的两个供暖器 , 比如房屋2  找取暖器1和4
	//二分？还是新开数组给O（n）
	int n = houses.size();
	int pre = 0, last = 0;
	int ans = 0;
	for (auto& h : houses) {
		while (last<heaters.size() && h>heaters[last]) {
			pre = last;
			++last;  //找到房屋左侧和右侧的热水器
		}
		if (last == 0) {  //第一个热水器就在房子同一位置或右侧
			ans = max(ans, heaters[last] - h);  // 简单来说就是房屋左侧没有热水器
		}
		else if (pre == heaters.size() - 1) {  // 房屋右侧没有热水器 
			ans = max(ans, houses[houses.size() - 1] - heaters[pre]);
		}
		else {
			ans = max(ans, min(h - heaters[pre], heaters[last] - h));
		}
		//cout<<h<<" " <<heaters[pre]<<" "<<heaters[last]<<endl;
	}
	return ans;
}


// 字符串匹配问题
class Solution_substring {
public:
	// 686. 重复叠加字符串匹配
	int repeatedStringMatch(string a, string b) {
		int m = a.size();
		int n = b.size();
		int ans = 1;
		string temp = a;
		while (a.size() <= 2 * m + n) {
			if (a.find(b) == string::npos) {
				++ans;
				a += temp;
			}
			else {
				return ans;
			}
		}
		return -1;
	}
};


//207. 课程表
//210. 课程表II
//630. 课程表III
//1462.课程表IV
class SolutionKCB {
public:
	// 课程表I  显然拓扑排序/或者dfs/bfs    判断是否有环    可以自身到自身
	bool canFinish207(int numCourses, vector<vector<int>>& prerequisites) {
		vector<vector<int>> graph(numCourses);
		vector<int> indeg(numCourses);
		for (auto& a : prerequisites) {
			if (a[0] == a[1]) {
				return false;
			}
			++indeg[a[1]];
			graph[a[0]].emplace_back(a[1]);
		}
		queue<int> que;
		for (int i = 0; i < (int)indeg.size();++i) {
			if (!indeg[i]) {
				que.push(i);
			}
		}
		while (!que.empty()) {
			int temp = que.front();
			que.pop();
			for (int i = 0; i < graph[temp].size(); ++i) {
				if (--indeg[graph[temp][i]] == 0) {
					que.push(graph[temp][i]);
				}
			}
		}
		return all_of(indeg.begin(), indeg.end(), [](const int& a) {
			return a == 0;
		});
	}


	// 课程表II和I的意思一样，只不过是多了个打印结果

	// 630. 课程表 III
	int scheduleCourse630(vector<vector<int>>& courses) {
		return 0;
	}
};

// 1705. 吃苹果的最大数目
// apples[i]代表第i天产生多少个苹果
// days[i]代表第i天的苹果多久腐烂
// 贪心法，每次吃最早要过期的苹果
// 以天为条件，比较好做
int eatenApples(vector<int>& apples, vector<int>& days) {
	std::function cmp = [&](const pair<int, int>& a, const pair<int, int>& b) ->bool {
		return a.first == b.first ? a.second >= b.second : a.first >= b.first;
	};
	priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> que(cmp);   //天 果子数
	int n = (int)apples.size();
	int ans = 0;
	int days_count = 0;
	while (days_count < n) {
		if (!que.empty() && que.top().first <= days_count) {
			que.pop();
		}
		int day_end = days[days_count] + days_count;
		int apple_num = apples[days_count];
		if (apple_num > 0) {
			que.push({ day_end,apple_num });
		}
		if (!que.empty()) {
			PII temp = que.top();
			que.pop();
			temp.second--;
			if (temp.second > 0) {
				que.push({ temp.first,temp.second });
			}
			++ans;
		}
		++days_count;
	}

	while (!que.empty()) {
		if (!que.empty() && que.top().first <= days_count) {
			que.pop();
		}
		if (que.empty()) {
			return ans;
		}
		PII temp = que.top();
		que.pop();
		int num = min(temp.first - days_count, temp.second);
		ans += num;
		days_count += num;
	}
	return ans;
}


// 使用 stringstream 使用指定字符分割字符串
vector<string> ss(string text) {
	stringstream ss;
	ss << text;
	string temp;
	vector<string> ans;
	while (getline(ss, temp, ' ')) {
		ans.push_back(temp);
	}
	return ans;
}



// rabin-karp
class RabinKarp {
private:
	int p = 26;
public:
	RabinKarp() {};
	~RabinKarp() {};
	int rabinkarp(string s, string t) {  //s父串，  t子串
		// 子串hash
		int m = s.size();
		int n = t.size();
		unsigned long long subhash = 0;
		for (int i = 0; i < n; ++i) {
			subhash = subhash * p + (t[i] - 'a');
		}

		//父串
		unsigned long long hash = 0;
		unsigned long long power = 1;
		unordered_map<unsigned long long, int> mp;
		for (int i = 0; i < n; ++i) {
			hash = hash * p + s[i] - 'a';
			power *= p;
		}
		mp[hash]++;
		if (mp.count(subhash)) {
			return 0;   // return index
		}
		
		for (int i = n; i < m; ++i) {
			hash = hash * p - (s[i - n] - 'a') * power + (s[i] - 'a');
			mp[hash]++;
			if (mp.count(subhash)) {
				return i - n + 1;
			}
		}
		return -1;
	}
};


// 1044. 最长重复子串   hard
// 重复子串代表该连续子串在 字符串中出现的次数为2次或更多，出现之间可以考虑重叠，返回 任意一个 具有  **最长长度** 的重复子串
// // rabin-karp算法  + 二分   (字符串哈希)
// 二分处理长度
// 对于 rabin-karp 考虑将每一个字符转换为 0-26  转换为26进制  考虑所有字符串中所有的乘积
// 那如何进行hash呢？ 我们可以用一个质数 p ，比如 31 当作底数； 
// 将字符串转化为 sub[0]*p^(m-1)+sub[1]*p^(m-2)...+sub[m-1]

class Solution1014 {
public:
	unordered_set<unsigned long long, int> set;
	string longestDupSubstring(string s) {
		int n = s.size();
		int l = 1;
		int r = n - 1;
		int len = 0;
		unsigned long long prime = 31;

		auto find = [&](int len)->int {
			unsigned long long power = 1;
			unsigned long long hash = 0;

			for (int i = 0; i < len; ++i) {
				hash = hash * prime + (s[i] - 'a');
				power = power * prime;
			}
			unordered_map<unsigned long long, int> mp;
			mp[hash]++;
			for (int i = len; i < n; ++i) {
				hash = hash * prime - power * (s[i - len] - 'a') + (s[i] - 'a');
				if (mp.count(hash)) {
					// 返回起始坐标
					return (i - len + 1);
				}
				mp[hash]++;
			}
			return -1;
		};

		int ans = -1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			int index = find(mid);
			if (index != -1) {  //找到重复的，那么加大长度继续查找
				len = mid;   //记录当前长度
				l = mid + 1;
				ans = index;
			}
			else {
				r = mid - 1;
			}
		}
		if (ans == -1) {
			return "";
		}
		else {
			return s.substr(ans, len);
		}
	}
};


//825. 适龄的朋友
class Solution825 {
public:
	//case1使用 二分查找+双指针
	int numFriendRequests_case1(vector<int>& ages) {
		int n = (int)ages.size();
		int ans = 0;
		sort(ages.begin(), ages.end());
		// x->y  条件2和条件3合并  大的给小的发送
		// 全满足   0.5 * age[x] + 7 < age[y] <= age[x] 才发送
		// 当前用户为x !!!!!!
		for (int i = 0; i < n; ++i) {
			int left = upper_bound(ages.begin(), ages.end(), 0.5 * ages[i] + 7) - ages.begin();
			int right = upper_bound(ages.begin(), ages.end(), ages[i]) - ages.begin();
			//cout<<i<<left<<right<<endl;
			ans += (right > left ? right - left - 1 : 0);
		}
		return ans;
	}

	// case2 使用计数排序加前缀和
	int numFriendRequests_case2(vector<int>& ages) {
		// ages为[1,120]    计数排序
		vector<int> count(121);
		for (auto& age : ages) {
			count[age]++;
		}

		//年龄人数的前缀和
		vector<int> presum(121);
		for (int i = 1; i < 121; ++i) {
			presum[i] = presum[i - 1] + count[i];
		}

		int ans = 0;
		for (int i = 1; i < 121; ++i) {
			if (i < 15) {
				continue;
			}
			if (count[i]) {
				// 当前用户为x, 只有当y>0.5*age[x]+7 且 0.5x+7<x，  可以看到x大于15 才发送  找边界  
				int left = 0.5 * i + 7 + 1;  //向上取整
				ans += count[i] * (presum[i] - presum[left - 1] - 1);
				//乘以count[i]是不排除同年龄有多个人的情况
			}
		}
		return ans;
	}
};

// 1995. 统计特殊四元组
// 是个简单题，但是注意hash的存储细节！！！！！！！！！！！！
int countQuadruplets(vector<int>& nums) {
	//a < b < c < d   不连续！
	int n = nums.size();
	unordered_map<int, int> mp;
	int ans = 0;
	for (int b = n - 3; b >= 1; --b) {
		/*
		// 这个块是错误答案，因为我们先计算a+b,会造成重复计算
		// 因为从大长度逐步扩展到小长度
		// 对于提前存储结果，从小长度到大长度比较好
		for(int a = 0;a<b;++a){  //  先找出所有的a + b
			mp[nums[a]+nums[b]]++;
		}
		int d=n-1;
		for(int c = d-1;c>b;--c){
			ans+=(mp[nums[d]-nums[c]]);
			cout<<c<<" "<<d<<endl;
		}*/
		//b给了最长的长度，那么我们先存储c 和 d的值
		int c = b + 1;
		for (int d = c + 1; d < n; ++d) {
			mp[nums[d] - nums[c]]++;
		}
		for (int a = 0; a < b; ++a) {
			ans += mp[nums[a] + nums[b]];
		}
	}
	return ans;
}


// 472. 连接词
// 使用trie-tree 前缀树   给个shared_ptr的实现
//  每个单词的 结尾给个end标记

class Solution472 {
private:
	struct TrieTree {
		bool isend;
		vector<std::shared_ptr<TrieTree>> children;
		TrieTree() :isend(false), children(26, nullptr) {};
	};
	std::shared_ptr<TrieTree> root;

	void insert(shared_ptr<TrieTree> ptr, string s, int index) {
		if (index == s.size()) {
			ptr->isend = true;
			return;
		}

		int temp = s[index] - 'a';
		if (ptr->children[temp] == nullptr) {
			ptr->children[temp] = std::make_shared<TrieTree>();
		}
		ptr = ptr->children[temp];
		insert(ptr, s, index + 1);
	}

	int find(shared_ptr<TrieTree> ptr, string s, int index) {
		if (ptr != nullptr) {
			if (ptr->isend) {
				if (index == s.size()) {
					return 1;
				}
				else {
					return find(root, s.substr(index), 0);
				}
			}
			else {
				if (index == s.size() || !ptr->children[s[index] - 'a']) {
					return -1;
				}
				else {
					return find(ptr->children[s[index] - 'a'], s, index + 1);
				}
			}
		}
		return -1;
	}

public:
	Solution472() :root(std::make_shared<TrieTree>()) {}

	vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
		auto cmp = [&](const string& a, const string& b)->bool{
			return a.size() < b.size();
		};
		sort(words.begin(), words.end(), cmp);
		vector<string> ans;
		for (auto& a : words) {
			int index = 0;
			shared_ptr<TrieTree> temp = root;
			while (index < a.size() && index>=0) {
				index = find(temp, a, index);
			}
			if (index == a.size()) {
				ans.emplace_back(a);
			}
			insert(temp, a, 0);
		}
		return ans;
	}
};


// 846. 一手顺子
// mid 比较简单   排序+ hash   贪心
// 排序也可以换为  priority_queue  每次取首
bool isNStraightHand(vector<int>& hand, int groupSize) {
	// hash记录数量
	// 贪心法
	int n = hand.size();
	if (n % groupSize) {
		return false;
	}
	sort(hand.begin(), hand.end());
	unordered_map<int, int> mp;
	for (auto& a : hand) {
		mp[a]++;
	}
	int i = 0;
	while (i < n) {
		int temp = hand[i];
		if (mp[temp] > 0) {
			mp[temp]--;
			int next = temp + 1;
			for (int j = 1; j < groupSize; ++j) {
				//cout<<temp<<"  "<<next<<endl;
				if (mp[next] <= 0) {
					return false;
				}
				else {
					mp[next]--;
				}
				next++;
			}
		}
		i++;
	}
	return true;
}


// 390. 消除游戏
// 从左到右，删除第一个数字，然后每隔一个数字删除一个，直到到达列表末尾
// 从右到左类似
// 返回最后剩下的一个数字
int lastRemaining(int n) {
	int count = 1;
	int gap = 1;
	int head = 1;
	while (n >= 2) {
		if (count % 2) {  //从左到右
			head += gap;
			//cout<<head<<endl;
		}
		else {  // 从右到左
			if (n % 2) {  //奇数
				head += gap;
			}
			//cout<<head<<endl;
		}
		gap *= 2;
		n /= 2;
		count++;
	}
	return head;
}


// zeller公式  蔡勒公式
string dayOfTheWeek(int day, int month, int year) {
	// 1971.1.1 星期五, 记录距离该日期过了多少天，倒推周几
	// case1 zeller 公式
	vector<string> week{ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
	// 月份范围为3-14,  这一年的1 2月要看做上一年的13  14月，因此年份减1
	if (month < 3 && month>0) {
		month += 12;
		year--;
	}
	int c = year / 100;
	int y = year % 100;
	int D = (c / 4 - 2 * c + y + (y / 4) + (13 * (month + 1) / 5) + day - 1);
	while (D < 0) {
		D += 7;
	}
	//cout<<c<<"  "<<y<<"  "<< D <<endl;
	return week[D % 7];
}


// 913. 猫和老鼠
// 无向图行走博弈问题
// 猫赢 return 2   老鼠赢return 1    平局return 0
class Solution913 {
private:
	int n;
	vector<vector<vector<int>>> mem;    //  mem[c][m]表示  猫在c  老鼠在m  的状态
	// // 单纯的bfs   超时了  TLE
	// 记忆化搜索需要保存什么信息
public:
	int bfs(vector<vector<int>>& graph, int mouse, int cat, int step) {
		int count = step % 2;

		if (cat == mouse) {
			return 2;
		}
		if (mouse == 0) {
			return 1;
		}
		if (step > 2 * n) {
			return 0;
		}

		if (mem[cat][mouse][step] != -1) {
			return mem[cat][mouse][step];
		}

		// 老鼠先走
		if (!count) {
			int peace = 0;  //  平局
			for (auto& m : graph[mouse]) {
				int res = bfs(graph, m, cat, step + 1);
				if (res == 1) {
					mem[cat][mouse][step] = 1;
					return 1;
				}
				if (res == 0) {
					peace = 1;
				}
			}

			// 当前所有位置都没赢且有平局，   平局好过输
			if (peace) {
				mem[cat][mouse][step] = 0;
				return 0;
			}

			// 老鼠没赢    且  无法达到平局，  那么猫赢了
			mem[cat][mouse][step] = 2;
			return 2;
		}

		// 猫走
		int peace = 0; //猫平局
		for (auto& c : graph[cat]) {
			if (c == 0) {
				continue;   // 猫进不去洞
			}
			int res = bfs(graph, mouse, c, step + 1);
			if (res == 2) {
				mem[cat][mouse][step] = 2;
				return 2;
			}
			if (res == 0) {
				peace = 1;
			}

		}
		// 当前所有位置都没赢且有平局，   平局好过输
		if (peace) {
			mem[cat][mouse][step] = 0;
			return 0;
		}
		// 老鼠赢
		mem[cat][mouse][step] = 1;
		return 1;
	}

	int catMouseGame(vector<vector<int>>& graph) {
		this->n = graph.size();
		this->mem.resize(n, vector<vector<int>>(n,vector<int>(2*n+1, -1)));
		return bfs(graph, 1, 2, 0);
	}
};


// 407. 接雨水 II
// 3D接雨水
// 边界收缩
class Solution407 {
public:
	using TIII = tuple<int, int, int>;
	int trapRainWater(vector<vector<int>>& heightMap) {
		// 从最外层开始，逐步向内收缩
		priority_queue<TIII, vector<TIII>, greater<TIII>> que;
		int m = (int)heightMap.size();
		int n = (int)heightMap[0].size();
		vector<vector<int>> visited(m, vector<int>(n, 0));
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
					// 加入初始的边缘
					que.push(make_tuple(heightMap[i][j], i, j));
					visited[i][j] = 1;
				}
			}
		}

		vector<vector<int>> step{ {0,1},{0,-1},{1,0},{-1,0} };
		int ans = 0;
		while (!que.empty()) {
			TIII temp = que.top();
			que.pop();
			int nxt = INT_MAX;
			for (int i = 0; i < 4; ++i) {
				int new_x = get<1>(temp) + step[i][0];
				int new_y = get<2>(temp) + step[i][1];
				if (new_x >= 0 && new_x < m && new_y >= 0 && new_y < n && !visited[new_x][new_y]) {
					// 未越界
					if (heightMap[new_x][new_y] < get<0>(temp)) {  // 当前位置可蓄水
						ans += get<0>(temp) - heightMap[new_x][new_y];
					}
					visited[new_x][new_y] = 1;
					que.push(make_tuple(max(get<0>(temp), heightMap[new_x][new_y]), new_x, new_y));    
					//用max考虑是否注入了水
				}
			}
		}
		return ans;
	}
};


// 71. 简化路径
// 先按 ‘/’划分字符串
class Solution71 {
public:
	string simplifyPath_case1(string path) {
		// sstream分割字符串
		stack<string> st;
		stringstream ss;
		// istringstream 分割字符串，输入格式
		// istringstream iss(path);
		ss << path;
		string temp;
		while (getline(ss, temp, '/')) {
			if (!temp.empty() && temp != "." && temp != "..") {
				st.emplace(temp);
			}
			if (!st.empty() && temp == "..") {
				st.pop();
			}
		}

		if (st.empty()) {
			return "/";
		}

		string ans = "";
		while (!st.empty()) {
			string temp = st.top();
			st.pop();
			ans = "/" + temp + ans;
		}
		return ans;
	}


	// 方法2自己手写的split    其他意思类似
	string simplifyPath_case2(string path) {
		string ans = "";
		stack<string> st;
		
		auto split = [](const string& s, char tag)->vector<string> {
			vector<string> ans;
			string temp;
			for (auto& a : s) {
				if (a == tag) {
					if (temp != "") {
						ans.push_back(temp);
					}
					temp.clear();
					continue;
				}
				temp += a;
			}
			ans.push_back(move(temp));
			return ans;
		};

		vector<string> vec = split(path, '/');
		for (auto& a : vec) {
			if (a == "") {
				continue;
			}
			if (a != "." && a != "..") {
				st.push(a);
			}
			if (!st.empty()&&a == "..") {
				st.pop();
			}
		}

		if (st.empty()) {
			return "/";
		}

		while (!st.empty()) {
			auto a = st.top();
			st.pop();
			if (a.empty()) {
				continue;
			}
			ans = "/" + a + ans;
		}
		return ans;
	}
};


//================数值重新排序以满足条件  的问题    还原数组
class SolutionArrayResort {
public:
	//954. 二倍数对数组
	bool canReorderDoubled(vector<int>& arr) {
		map<double, int> mp;
		for (auto& a : arr) {
			mp[a]++;
		}

		for (auto& [key, value] : mp) {
			if (value == 0) {
				continue;
			}
			if (key < 0) {
				if (mp[key / 2.0] >= value) {
					mp[key / 2.0] -= value;
				}
				else {
					return false;
				}
			}
			else {
				if (mp[key * 2.0] >= value) {
					mp[key * 2.0] -= value;
				}
				else {
					return false;
				}
			}
		}
		return true;
	}

//2007. 从双倍数组中还原原数组
	vector<int> findOriginalArray(vector<int>& changed) {
		int n = changed.size();
		vector<int> ans;
		if (n % 2) {
			return ans;
		}

		sort(changed.begin(), changed.end());    
		// 从小到大逐渐查找，未匹配上的放入队列
		queue<int> que;
		for(int i = 0;i<n;++i){
			if (!que.empty() && changed[i] == 2 * que.front()) {
				ans.emplace_back(que.front());
				que.pop();
			}
			else {
				que.push(changed[i]);
			}
		}

		if (!que.empty()) {
			ans.clear();
		}
		return ans;
	}


// 2122. 还原原数组
// 和2007比，  多了个枚举间隔K的步骤
// case1 双指针   或     case2 队列

	vector<int> recoverArray_case1(vector<int>& nums) {
		int n = nums.size();
		sort(nums.begin(), nums.end());

		for (int i = 1; i < n; ++i) {
			if ((nums[i] - nums[0]) % 2 || nums[i] == nums[0]) {
				continue;
			}

			vector<int> used(n + 1);
			used[0] = used[i] = 1;
			int k = (nums[i] - nums[0]) / 2;  // 枚举个k
			// i 的作用是枚举k
			vector<int> ans;
			ans.emplace_back(nums[0] + k);

			int left = 0, right = i;  // left为low的起点   right为high的起点
			for (int j = 1; j + j < n; ++j) {  //要找的原数组的数字个数
				while (used[left]) {
					++left;   // 找没用过的low数组
				}
				// 找到了一个left， 开始找对应的right
				while (right < n && (used[right] || nums[left] + 2 * k != nums[right])) {
					++right;
				}

				if (right == n) {
					break;
				}

				//找到了
				//cout<<k<<" "<<left<<" "<<right<<endl;
				used[right] = 1;
				ans.emplace_back(nums[left] + k);
				++left;
				++right;
			}
			if (ans.size() == n / 2) {
				return ans;
			}
		}
		return vector<int>{};
	}
};





// 306. 累加数
// 我对于字符串类的爆搜  做的题太少     字符串爆搜
class Solution306 {
public:
	int n;
	bool isAdditiveNumber(string num) {
		this-> n = (int)num.size();
		for (int i = 0; i < n-2; ++i) {  // 第一个数字的终点
			if (num[0] == '0' && i > 0) {
				return false;
			}
			for (int j = i + 1; j < n-1; ++j) {  // 第二个数字的终点
				// 对于dfs, 需要哪些信息
				// 考虑枚举前两个数字， 确定了  firstbegin = 0  firstend = i  [0,i+1)  
				// secondbegin = i+1 secondend = j+1  [i+1, j+1)

				// 若第二个数字存在前导0
				if (num[i + 1] == '0' && j > i + 1) {
					continue;
				}
				LL firstNum = stoll(num.substr(0, i + 1));
				LL secondNum = stoll(num.substr(i + 1, j - i));
				if (dfs(num, firstNum, secondNum, j + 1)) {
					return true;
				}
			}
		}
		return false;
	}

	bool dfs(string& num, LL& firstnum, LL& secondnum, int lastbegin) {
		if (lastbegin == n) {
			return true;
		}
		for (int i = lastbegin; i < n; ++i) {
			if (num[lastbegin] == '0' && i > lastbegin) {
				return false;
			}
			LL now = stoll(num.substr(lastbegin, i - lastbegin + 1));
			if (now > firstnum + secondnum) {
				return false;
			}
			if (now == firstnum + secondnum) {
				if (dfs(num, secondnum, now, i + 1)) {
					return true;
				}
				else {
					break;
				}
			}
		}
		return false;
	}

	//string stringAdd(string& s1, string& s2) {
		// 大数加法

	//}
};



// 1036. 逃离大迷宫
class Solution1036 {
private:
	vector<vector<int>> step{ {1,0},{-1,0},{0,1},{0,-1} };
	int max_ele;
	unordered_map<string, int> blk;
public:
	using PII = pair<int, int>;
	bool bfs(vector<int>& start, vector<int>& end) {
		unordered_map<string, int> visited;
		queue<PII> que;
		que.push(make_pair(start[0], start[1]));
		visited[to_string(start[0]) + "_" + to_string(start[1])]++;
		int count = 1;
		while (!que.empty() && count <= max_ele) {
			PII temp = que.front();
			que.pop();
			int x = temp.first, y = temp.second;
			for (int i = 0; i < 4; ++i) {
				int new_x = x + step[i][0];
				int new_y = y + step[i][1];
				//cout<<new_x<<" "<<new_y<<endl;
				if (new_x >= 0 && new_x < 1e6 && new_y >= 0 && new_y < 1e6 && visited[to_string(new_x) + "_" + to_string(new_y)] == 0 && blk[to_string(new_x) + "_" + to_string(new_y)] == 0) {
					if (new_x == end[0] && new_y == end[1]) {
						return true;
					}
					que.push({ new_x,new_y });
					visited[to_string(new_x) + "_" + to_string(new_y)]++;
					count++;
				}
			}
		}
		//cout<<count<<" "<<que.size()<<" "<<max_ele<<endl;
		return count > max_ele;
	}

	bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target) {
		// 10^6不能爆搜
		// blocked不能走动的点位数量为  0~200个
		// 目的是判断能否走到终点，那么只要判断封锁点位是否把初始点(source)给围住就行了
		// source 和 target 进行   bfs，只要栈内的点的数量大于 blocked.size()， 那就说明blocked围不住了
		int n = blocked.size();
		this->max_ele = n * (n - 1) / 2;     ///     这个范围最重要
		for (auto& a : blocked) {   //  添加禁止位置
			string temp = to_string(a[0]) + "_" + to_string(a[1]);
			blk[temp]++;
		}
		int a = bfs(source, target);

		if (!a) {
			return false;
		}
		int b = bfs(target, source);
		if (!b) {
			return false;
		}
		return true;
	}
};


// 334. 递增的三元子序列
// 这题实际上就是找   有没有长度为3的递增子序列
// 使用 300 严格递增子序列的解法 ，只是到长度为三就停止，否则return false
// 贪心+二分查找
bool increasingTriplet(vector<int>& nums) {
	int n = nums.size();
	if (n == 0) {
		return 0;
	}
	int len = 1;
	vector<int> b(n + 1);
	b[len] = nums[0];
	for (int i = 1; i < n; ++i) {
		if (nums[i] > b[len]) {
			b[++len] = nums[i];
			if (len >= 3) {
				return true;
			}
		}
		else {
			int l = 1, r = len;
			while (l <= r) {
				int mid = l + (r - l) / 2;
				if (b[mid] < nums[i]) {
					l = mid + 1;
				}
				else {
					r = mid - 1;
				}
			}
			b[l] = nums[i];
		}
	}
	return false;
}


// 多路归并
class multimerge {
public:

	//丑数II   264
	int nthUglyNumber(int n) {
		if (n == 1) {
			return 1;
		}
		int count = 1;
		//如何确保顺序?
		//priority_queue每次取出最小值
		priority_queue<int, vector<int>, greater<int>> pr_que;
		pr_que.emplace(1);
		while (n > 0) {
			int min_ele = pr_que.top();
			pr_que.pop();
			pr_que.emplace((min_ele * 2));
			pr_que.emplace((min_ele * 3));
			pr_que.emplace((min_ele * 5));
			n--;
		}
		return pr_que.top();
	}

	//丑数 方法2
	//dp  三指针
	int nthUglyNumber_case2(int n) {
		if (n == 1) {
			return 1;
		}
		vector<int> dp(n);
		dp[1] = 1;
		int p2 = 1, p3 = 1, p5 = 1;
		for (int i = 2; i < n; ++i) {
			int nums1 = p2 * 2, nums2 = p3 * 3, nums3 = p5 * 5;
			dp[i] = min(min(nums1, nums2), nums3); //找出乘了以后的最小值
			if (dp[i] == nums1) {
				p2++;
			}
			if (dp[i] == nums2) {
				p3++;
			}
			if (dp[i] == nums3) {
				p5++;
			}//选取了，则后移

		}
		cout << dp[n - 1] << endl;
		return dp[n - 1];
	}

// 373. 查找和最小的K对数字
	//给定两个以升序排列的整数数组 nums1 和 nums2 , 以及一个整数 k 。
	//定义一对值 (u, v)，其中第一个元素来自 nums1，第二个元素来自 nums2 。
	// 找到和最小的k个数对
	vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
		int m = nums1.size();
		int n = nums2.size();

		auto cmp = [&](const auto& a, const auto& b)->bool {
			return nums1[a.first] + nums2[a.second] > nums1[b.first] + nums2[b.second];
		};

		priority_queue<PII, vector<PII>, decltype(cmp)> que(cmp);

		for (int i = 0; i < min(m, k); ++i) {
			que.emplace(make_pair(nums1[i], nums2[0]));  // 先取nums1中的最小值和nums2中的所有数相加
			// 后续每次取和最小的一组，更新下一位
		}

		vector<vector<int>> ans;
		while (!que.empty() && k > 0) {
			auto temp = que.top();
			ans.push_back({ nums1[temp.first], nums2[temp.second] });
			que.pop();
			if (temp.second + 1 < n) {
				que.push(make_pair(temp.first, temp.second + 1));
			}
			--k;
		}
		return ans;
	}
};


// 5982. 解决智力问题
// dp的查（填）表法  和  刷表法
// 查（填）表就是由谁而来（在DP转移过程中，每个状态是由它之前的状态推导而来
// 刷表就是向下推进（每个状态产生其他的新状态
/*
	也是个可选可不选的问题，但是选了之后，一段时间内不能再做选择
	我认为可以理解为打家劫舍问题的一个拓展版本， 可归纳
*/
class Solution5982 {
private:
	int len;
public:
	// case1 给个查（填）表法， 即
	long long mostPoints_case1(vector<vector<int>>& questions) {
		// 考虑当前的状态由谁而来
		// 从后往前考虑，注意越界问题, dp[i]代表的是到位置i的最大得分
		int len = questions.size();
		vector<long long> dp(len + 1, 0);
		for (int i = len - 1; i >= 0; --i) {
			int last = i + questions[i][1] + 1;
			dp[i] = max(dp[i + 1], last >= len ? questions[i][0] : dp[last] + questions[i][0]);
			// max(不选当前位置，那么等于上一个位置， 选当前位置)
			// 当前位置，是由前一个位置更新而来,可以理解为，前一个位置也是后一个位置加上他的得分而来
		}
		return dp[0];
	}

	// case2 给个刷表法， 即
	long long mostPoints_case2(vector<vector<int>>& questions) {
		// 考虑当前状态如何影响下一状态
		int len = questions.size();
		vector<long long> dp(len, 0);
		for (int i = 0; i < len; ++i) {
			int nxt = i + questions[i][1];  //下一状态
			if (nxt < len) {  //能转移到下一位置
				dp[nxt] = max(dp[nxt], dp[i] + questions[i][0]);  //选了当前位置，下一位置被影响了
			}
			else {  //不能转移到下一位置,那么当前位置为最后的位置，记录和
				dp[len] = max(dp[len], dp[i] + questions[i][0]);
			}
			// 若此时不选当前位置,那么下一位置为i+1
			dp[i + 1] = max(dp[i], dp[i + 1]);
		}
		return dp[len];
	}

	// case3 给dfs的方法
	// 最朴素的dfs会超时
	long long mostPoints_case3(vector<vector<int>>& questions) {
		this->len = (int)questions.size();
		long long ans = dfs(questions, 0, 0, 0);
		return ans;
	}

	long long dfs(vector<vector<int>>& questions, int sum, int index, int tag) {
		/*
			question 
			sum：当前的和
			index：当前问题的下标
			tag: 标记现在禁止选取的限制是否已经解除
		*/

		if (index == questions.size()) {
			return sum;
		}

		int a = 0, b = 0, c = 0;  
		// a代表之已经选过了，现在无法再选
		// b代表现在可以选，并且选了的操作
		// c代表现在可以选但是没有选
		if (tag > 0) {
			a += dfs(questions, sum, index + 1, --tag);
		}
		else {
			b+= dfs(questions, sum+questions[index][0], index + 1, questions[index][1]);  //选
			c+= dfs(questions, sum, index + 1, 0);  //不选
		}
		return max(a, max(b, c));
	}
};



// 382. 链表随机节点
// 类型为不知道容器的大小，但是要从中随机选择出k个数字。 
// 可用于数据库的数据选择 或 文本读取等。 
// 1、只能顺序读取  2、不可一次项读取到内存中
// 蓄水池抽样算法
/*
	蓄水池算法
*/
vector<int> reservoir(vector<int>& vec, int k) {
	// 假设不知道vec的size是多少
	// 从中取出k个数字, k确定小于vec
	vector<int> ans;
	int count = 0;
	while (count<k) {
		ans.push_back(vec[count]);
		++count;
	}
	// 开始随机替换, 概率为 k/i 当前为第i行
	for (vector<int>::iterator begin = vec.begin() + k + 1; begin < vec.end();++begin) {
		uniform_int_distribution<unsigned> u(0, count);
		default_random_engine e;
		int index = u(e);
		if (index < k) {
			static uniform_int_distribution<unsigned> p(0, k-1);
			static default_random_engine q;
			int line = p(q);
			ans[line] = *begin;
		}
		++count;
	}
	return ans;
}
// 若抽取一个数字的话
int pool_1(vector<int>& vec) {
	int ans;
	int count = 0;
	// 开始随机替换, 概率为 1/i 当前为第i行
	for (auto& a : vec) {
		count++;
		static uniform_int_distribution<unsigned> u(0, 10);
		static default_random_engine e;
		float index = (float)u(e) / 10.0;
		if (index < 1.0 / float(count)) {   // 第一个元素肯定选取，第二个元素有1/i的概率替换掉第一个元素，后续类似
			ans = a;
		}
	}
	return ans;
}


// 2013.检测正方形
// 一个 双层哈希表   的设计题
class DetectSquares {
public:
	//正方形不是长方形
	unordered_map<int, unordered_map<int, int>> mp;
	// {横坐标，{纵坐标，该点的数量}}
	DetectSquares() {

	}

	void add(vector<int> point) {
		int x = point[0];
		int y = point[1];
		mp[x][y]++;  //[x,y]点++
	}

	int count(vector<int> point) {
		int x = point[0];
		int y = point[1];
		if (!mp.count(x)) {
			return 0;
		}
		// 找横坐标相同的点
		int ans = 0;
		for (auto &[y1, count]:mp[x]) {
			if (y1 != y) {  //横坐标相同的点的纵坐标
				int edge = y1 - y;  //负也不影响，那就+变-  -变+
				// 下一个点是  (x+d,y) (x+d,y1)  或 (x-d,y)  (x-d,y1)
				// 因为存在重复点，因此要做乘法
				ans += (mp[x + edge].count(y) ? mp[x + edge][y] : 0) * (mp[x + edge].count(y1) ? mp[x + edge][y1] : 0) * (mp[x].count(y1) ? mp[x][y1] : 0);
				ans += (mp[x - edge].count(y) ? mp[x - edge][y] : 0) * (mp[x - edge].count(y1) ? mp[x - edge][y1] : 0) * (mp[x].count(y1) ? mp[x][y1] : 0);
			}
		}
		return ans;
	}
};


//386. 字典序排数
vector<int> lexicalOrder_case1(int n) { //递归
	// 1-> 10->100  101...->11->110 111....  2->20... 21 ...
	// 一共 n 个数字
	vector<int> ans;

	function<void(vector<int>& vec, int num, int n)> dgt = [&](vector<int>& vec, int num, int n)->void {
		if (num > n) {
			return;
		}
		vec.push_back(num);
		for (int i = 0; i < 10; ++i) {
			dgt(vec, num * 10 + i, n);
		}
	};

	for (int i = 1; i < 10; ++i) {
		dgt(ans, i, n);
	}
	return ans;
}

// 这题处理进位的方法好巧妙
vector<int> lexicalOrder_case2(int n) {
	// 1-> 10->100  101...->11->110 111....  2->20... 21 ...
	// 一共 n 个数字
	// eg  若到达了109， 第二位要发生进位了， 处理进位使用/10  +1
	vector<int> ans;
	int j = 1;
	for (int i = 0; i < n; ++i) {  // n个数
		ans.push_back(j);
		if (j * 10 <= n) {
			j *= 10;
		}
		else {
			// 越界则倒退回来
			while (j + 1 > n || j % 10 == 9) {
				j /= 10;
			}
			++j;  // 进位  若n=110, 当num=109， 按照字典序，下一位为11
			// 109/10=10， 10+1->11,   
		}
	}

	return ans;
}

// 378. 有序矩阵中第 K 小的元素





// 375. 猜数字大小 II
int getMoneyAmount(int n) {
	return 0;
}


class tests {
private:
	typedef shared_ptr<tests> ptr;
	int a = 0;
	int b = 1;
public:
};


// 给定坐标系第一象限的点
// 返回所给的点集中满足右上方没有集合中其他点的  所有点
/*
1 2
5 3
4 6
7 5
9 0

满足条件的是
4 6
7 5
9 0
*/
class Dot {
public:
	int max_y;
	vector<PII> vec;
	vector<PII> ans;
	vector<PII> getDot(vector<PII>& v) {
		auto cmp = [&](const PII& a, const PII& b)->bool {
			return a.second >= b.second;
		};

		sort(v.begin(), v.end(), cmp);
		int last_x = -1;
		int last_y = 1e9;
		for (auto& a : vec) {

			if (a.second<last_y && a.first>last_x) {
				last_x = a.first;
				last_y = a.second;
				ans.emplace_back(a);
			}
		}
		return ans;
	}
};

/*
int main(){
  int N;
  cin N;
  Dot dt;
  while(N){
	int x, y;
	cin >> x >> y;
	dt.vec.emplace_back(make_pair(x,y));
  }
  dt.getDot(dt.vec);
  return 0;
}
*/


/*
给定一个数组序列, 需要求选出一个区间, 使得该区间是所有区间中经过如下计算的值最大的一个：
区间中的 最小数 * 区间所有数的和 最后程序输出经过 计算后的最大值 即可，不需要输出具体的区间。
如给定序列  [6 2 1]则根据上述公式, 可得到所有可以选定各个区间的计算值:
[6] = 6 * 6 = 36;
[2] = 2 * 2 = 4;
[1] = 1 * 1 = 1;
[6,2] = 2 * 8 = 16;
[2,1] = 1 * 3 = 3;
[6, 2, 1] = 1 * 9 = 9;
从上述计算可见选定区间 [6] ，计算值为 36， 则程序输出为 36。
区间内的所有数字都在[0, 100]的范围内;
*/

int count_max(vector<int> vec) {
	int n = vec.size();
	sort(vec.begin(), vec.end());
	vector<int> presum(n+1);
	presum[0] = vec[0];
	for (int i = 0; i < n;++i) {
		presum[i+1] = presum[i] + vec[i];
	}
	// [i,j] = presum[j+1]-presum[i]
	long long max_num = 0;
	for (int i = 0; i < n; ++i) {
		// 最小值起始位置
		for (int j = i; j < n; ++j) {
			// 最小值的结束位置
			long long temp = (presum[j + 1] - presum[i]) * vec[i];
			max_num = max(temp, max_num);
		}
	}
	return max_num;
}

/*
int main() {
	int N;
	cin >> N;
	int x;
	vector<int> vec;
	while (N) {
		cin >> x;
		vec.push_back(x);
		--N;
	}
	return count_max( vec);
}*/


void finishJob(int coder_num, vector<vector<int>>& jobs) {
	int n = jobs.size();
	sort(jobs.begin(), jobs.end(), [&](const vector<int>& a, const vector<int>& b)->bool {return a[1] < b[1]; });

	// 需要不断更新当前时间点的输入的idea
	std::function<bool(const vector<int>& a, const vector<int>& b)> cmp =
		[&](const vector<int>& a, const vector<int>& b)->bool {
		return a[3] == b[3] ? a[0] > b[0] : a[3] > b[3];
	};
	priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> que(cmp);

	int time = 0;
	int i = 0;
	// 初始工作集合

	std::function<bool(const PII& a, const PII& b)> cmp1 = [&](const PII& a, const PII& b)->bool {return a.first > b.first; };
	priority_queue <PII, vector<PII>, decltype(cmp1) > coder_end(cmp1);
	int temp = coder_num;
	vector<int> ans(n);
	for (; time <= jobs[n-1][1] || !coder_end.empty(); ++time) {
		// 查看当前是否有满足的工作
		while (i < n && jobs[i][1] <= time) {
			que.push(jobs[i]);
			i+=1;// 记录一下当前工作的位置
		}

		// 看是否有coder完成工作
		while (!coder_end.empty() && coder_end.top().first <= time) {
			ans[coder_end.top().second-1] = time;
			coder_end.pop();
			++temp;
		}


		// 执行目前能做的工作
		while (temp > 0 && !que.empty()) {
			temp--;
			coder_end.push(make_pair(que.top()[3] + time, que.top()[4]));   // push进去  结束时间，任务index
			que.pop();
		}
	}
	for (std::vector<int>::iterator a = ans.end()-1; a >= ans.begin();--a) {
		cout << *a << endl;
		if (a == ans.begin()) {
			break;
		}
	}
	return;
}

/*
* Test case  
2 2 5
1 1 1 2
1 2 1 1
1 3 2 2
2 1 1 2
2 3 5 5
* 
int main() {
	int PM, Coder, Idea;
	cin >> PM >> Coder >> Idea;
	vector<vector<int>> vec(Idea, vector<int>(5));
	int PmIdx, BeginTime, Pr, Need;
	while (Idea) {
		cin >>vec[Idea-1][0] >> vec[Idea-1][1] >> vec[Idea-1][2] >> vec[Idea-1][3];
		vec[Idea - 1][4] = Idea;
		Idea--;
	}
	finishJob(Coder, vec);
	return 0;
}
*/

// 零钱兑换（力扣322）
// 一维dp
int coinChange(vector<int>& coins, int amount) {
	if (amount == 0) {
		return 0;
	}
	int n = coins.size();
	vector<int> dp(amount + 1, amount + 1);
	dp[0] = 0;
	for (int i = 1; i <= amount; ++i) {
		for (int j = 0; j < n; ++j) { //coins
			if (i >= coins[j]) {
				dp[i] = min(dp[i - coins[j]] + 1, dp[i]);
			}
		}
	}
	return dp[amount] > amount ? -1 : dp[amount];
}


// 518 零钱兑换2
//请你计算并返回可以凑成总金额的硬币组合数。
/*
int change(int amount, vector<int>& coins) {
	if (amount == 0) {
		return 1;
	}
	int n = coins.size();*/
	// 状态：金额，可选择的物品
	// 选择：选取和不选取
	/*
		for  状态1
			for 状态2
				dp[状态1][状态2]
		若只使用前 i 个物品（可以重复使用），当背包容量为 j 时，有 dp[i][j] 种方法可以装满背包。
	*/
	/*
	vector<vector<int>> dp(n + 1, vector<int>(amount + 1, 0));
	for (int i = 0; i <= n; ++i) {
		dp[i][0] = 1;
	}

	for (int i = 1; i <= n; ++i) {  // 硬币的index  ，i=1表示使用第一个
		for (int j = 1; j <= amount; ++j) {   //金额
			if (j >= coins[i - 1]) {  // 选了上一个才到当前步骤
				dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]]; // 不选 + 选
			}
			else {
				dp[i][j] = dp[i - 1][j];  //上一个没选
			}
		}
	}
	return dp[n][amount];
}

*/



//821. 字符的最短距离
vector<int> shortestToChar(string s, char c) {
	// 每个位置左右两侧可能都会出现字符c
	int n = s.size();
	int idx = -n;
	vector<int> ans(n);
	for (int i = 0; i < n; ++i) {  //左侧
		if (s[i] == c) {
			idx = i;
		}
		ans[i] = i - idx;
	}

	idx = 2 * n;
	for (int i = n - 1; i >= 0; --i) { // 右侧
		if (s[i] == c) {
			idx = i;
		}
		ans[i] = min(ans[i], idx - i);
	}
	return ans;
}

// 388. 文件的最长绝对路径
// 模拟题
int lengthLongestPath(string input) {
	// 根据 \t\t..的个数来确定层数，一旦减少就相当于回退到上一目录了
	// 根据 \n 来划分input
	vector<string> ipt;
	stringstream ss;
	ss << input;
	string substr;
	while (getline(ss, substr, '\n')) {    // stringstream 实现字符串分割
		ipt.push_back(substr);
	}
	unordered_map<int, vector<string>> mp;
	int ans = 0;
	string pre;  // 上一级string
	int pre_count = 0;   // 上一级层数编号 
	for (auto& a : ipt) {
		int count = 0;
		int idx = 0;
		int tag = 0;
		for (int i = 0; i < a.size(); ++i) {
			if (a[i] == '\t') {
				count += 1;
				idx = i + 1;
			}
			if (a[i] == '.') {  //文件
				tag = 1;
			}
		}
		if (count > pre_count) {  // 产生了层的变化
			mp[count].push_back(mp[count-1].back() + "\\" + a.substr(idx));
		}
		else {
			mp[count].push_back(a.substr());
		}

		if (tag) {
			ans = max(ans, static_cast<int>(mp[count].back().size()));
		}
	}
	return ans;
}


// 2245. 转角路径的乘积中最多能有几个尾随零
// 尾随0的数量，取决于因数中5和2的个数。 min(num2, num5)。
// 图中有一个拐点，枚举路径不好枚举，直接枚举拐点
int maxTrailingZeross(vector<vector<int>>& grid) {
	int m = grid.size();  // 行
	int n = grid[0].size(); // 列
	// 保存各行各列的前缀和
	vector<vector<int>> col2(m + 1, vector<int>(n + 1)); // 列前缀
	vector<vector<int>> row2(m + 1, vector<int>(n + 1)); // 行前缀
	vector<vector<int>> col5(m + 1, vector<int>(n + 1)); // 列前缀
	vector<vector<int>> row5(m + 1, vector<int>(n + 1)); // 行前缀
	for (int i = 1; i <= m; ++i) {  // 行
		for (int j = 1; j <= n; ++j) { // 列
			int x = grid[i-1][j-1];
			int num2 = 0, num5 = 0;
			while (x % 2 == 0) {
				++num2;
				x /= 2;
			}
			while (x % 5 == 0) {
				++num5;
				x /= 5;
			}
			
			col2[i][j] = col2[i - 1][j] + num2;
			row2[i][j] = row2[i][j - 1] + num2;
			col5[i][j] = col5[i - 1][j] + num5;
			row5[i][j] = row5[i][j - 1] + num5;
		}
	}

	int ans = 0;
	// 枚举拐点
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			// 从左走向拐点，在向上走
			ans = max(ans, min(row2[i][j] + col2[i - 1][j], row5[i][j] + col5[i - 1][j]));
			// 从左走向拐点，在向下走
			ans = max(ans, min(row2[i][j] + col2[m][j] - col2[i][j], row5[i][j] + col5[m][j] - col5[i][j]));
			// 从右走向拐点，在向上走
			ans = max(ans, min(row2[i][n] - row2[i][j] + col2[i][j], row5[i][n] - row5[i][j] + col5[i][j]));
				// 从右走向拐点，在向下走
			ans = max(ans, min(row2[i][n] - row2[i][j] + col2[m][j] - col2[i-1][j], row5[i][n] - row5[i][j] + col5[m][j] - col5[i-1][j]));
		}
	}
	return ans;
}




//396. 旋转函数
/**
 * 把数组逆转跟把乘数逆转是一样的，可以看出有如下规律
 *   4     3     2     6
 *
 *  0*4   1*3   2*2   3*6   F(0)
 *
 *  3*4   0*3   1*2   2*6   F(1) = F(0) - SUM(data) + N * data[0];
 *
 *  2*4   3*3   0*2   1*6   F(2) = F(1) - SUM(data) + N * data[1];
 *
 *  1*4   2*3   3*2   0*6   F(3) = F(2) - SUM(data) + N * data[2];
 *
 */
class Solution396 {
public:
	int maxRotateFunction_case1(vector<int>& nums) {  // 暴力法会超时
		int n = nums.size();
		long long ans = INT_MIN;
		vector<int>vec(n);
		iota(vec.begin(), vec.end(), 0);
		for (int i = 0; i < n; ++i) {
			long long temp = 0;
			for (int j = 0; j < n; ++j) {
				temp += (vec[j] % n) * nums[j];
				vec[j] += 1;
			}
			ans = max(ans, temp);
		}
		return static_cast<int>(ans);
	}

	int maxRotateFunction_case2(vector<int>& nums) {
		int n = nums.size();
		int summ = accumulate(nums.begin(), nums.end(), 0);
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			ans += i * nums[i];
		}
		int a = ans;
		for (int i = 1; i < n; ++i) {
			int temp = a + summ - n * nums[n - i];
			ans = max(ans, temp);
			a = temp;
		}
		return ans;
	}
};



// 凸包算法
// 凸包算法-Graham scan
class Point {
public:
	Point() :x(0), y(0) {}
	Point(float xx, float yy) :x(xx), y(yy) {}
	bool operator==(const Point& p) {
		if (abs(p.x - this->x) < 0.00001 && abs(p.y - this->y < 0.00001)) {
			return true;
		}
		return false;
	}
	bool operator !=(const Point& p) {
		return !(*this == p);
	}
	friend std::ostream& operator<<(std::ostream out, Point& p) {
		out << "x: " << p.x << "   y:" << p.y << std::endl;
		return out;
	}

	float x;
	float y;	
};

// 1866. 恰有 K 根木棍可以看到的排列数目
// 有 n 根长度互不相同的木棍，长度为从 1 到 n 的整数。请你将这些木棍排成一排，并满足从左侧 可以看到 恰好 k 根木棍。
// 从左侧 可以看到 木棍的前提是这个木棍的 左侧 不存在比它 更长的 木棍。
class Solution1866 {
public:
	int M = 1e9 + 7;
	int rearrangeSticks(int n, int k) {
		// dp[i][j] 表示i根木棍，能看到j根的方案数目
		vector<vector<long long>> dp(n + 1, vector<long long>(k + 1));
		dp[1][1] = 1;
		for (int i = 2; i <= n; ++i) {
			for (int j = 1; j <= min(i, k); ++j) {
				dp[i][j] = dp[i - 1][j - 1];
				dp[i][j] += (dp[i - 1][j] * (i - 1)) % M;
				dp[i][j] %= M;
				// i-1是因为，当目前排列是12345，此时已经能够看到j根
				// 再插入木板6，就可以看到j+1根
				// 此时就可以通过调整最短的模板1，来达到再次看到j根的目的，
				// 可以看到木板1可以插到其余每块模板的后边， 共i-1种情况
			}
		}
		return static_cast<int>(dp[n][k]);
	}
};



// 计算其余点相对于初始点的极角
// 这个写法没处理好最后在一行线上的点，会有遗漏
class PolarAngle {
public:
	PolarAngle() :basePoint(Point()) {}
	PolarAngle(Point& p) :basePoint(p) {}
	
	// 计算cos
	float ccos(const Point& base, const Point& q) {
		float xdis = q.x - base.x;
		float ydis = q.y - base.y;
		float dis = std::sqrtf(xdis * xdis + ydis * ydis);
		return xdis / dis;
	}

	bool operator()(const Point& a, const Point& b) {
		return ccos(basePoint, a) > ccos(basePoint, b);
	}

private:
	Point basePoint;
};

std::vector<Point> convexHull(std::vector<pair<float, float>>& points) {
	int n = points.size();
	vector<Point> vec(n);
	for (int i = 0; i < n; ++i) {
		vec[i] = Point(points[i].first, points[i].second);
	}
	if (n <= 3) {
		return vec;
	}

	auto lessY = [&](const auto& a, const auto& b)->bool {
		return a.y < b.y;
	};

	auto antiCW = [&](const Point & a, const Point & b, const Point & c)->bool {
		Point ab(b.x - a.x, b.y - a.y);
		Point bc(c.x - b.x, c.y - b.y);
		return (ab.x * bc.y - ab.y * bc.x)>=0;
	};

	// 把 y 最小的起始点找出来
	std::vector<Point>::iterator init_point = min_element(vec.begin(), vec.end(), lessY);
	std::iter_swap(init_point, vec.begin());
	std::sort(vec.begin() + 1, vec.end(), PolarAngle(vec.front()));

	stack<Point> st;
	for (auto i = vec.begin(); i != vec.end();++i) {
		if (i == vec.begin() || i == vec.begin() + 1) {
			st.push(*i);
			continue;
		}
		Point p1 = st.top(); st.pop();
		while (!st.empty() && !antiCW(p1, st.top(), *i)) {
			p1 = st.top();
			st.pop();
		}
		st.push(p1);
		st.push(*i);
	}
	return vec;
}


// 587. 安装栅栏
// graham 凸包算法，   这种写法处理了最后在一行上的点
class Solution587 {
public:
	int anti_Clockwise(const vector<int>& a, const vector<int>& b, const vector<int>& c) {
		return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
		// 大于0说明是逆时针方向
	}
	int distance(const vector<int>& a, const vector<int>& b) {
		return (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
	}


	vector<vector<int>> outerTrees(vector<vector<int>>& trees) {
		// 凸包算法 
		int n = trees.size();
		if (n <= 3) {
			return trees;
		}

		int base = 0;
		// 找到y最小的点
		for (int i = 0; i < n; ++i) {
			if (trees[i][1] < trees[base][1]) {
				base = i;
			}
		}
		swap(trees[base], trees[0]);
		// 按照极角的大小进行排序
		sort(trees.begin() + 1, trees.end(), [&](const vector<int>& a, const vector<int>& b)->bool {
			int diff = anti_Clockwise(trees[0], a, b) - anti_Clockwise(trees[0], b, a);
			if (diff == 0) { //一条线
				return distance(trees[0], a) < distance(trees[0], b);
			}
			else {
				return diff > 0;
			}
			});

		/* 对于凸包最后且在同一条直线的元素按照距离从小到大进行排序 */
		int r = n - 1;
		while (r >= 0 && anti_Clockwise(trees[0], trees[n - 1], trees[r]) == 0) {
			r--;
		}
		for (int l = r + 1, h = n - 1; l < h; l++, h--) {
			swap(trees[l], trees[h]);
		}

		stack<int> st;
		st.emplace(0);
		st.emplace(1);
		for (int i = 2; i < n; ++i) {
			int top = st.top();
			st.pop();
			// 顺时针，则弹出
			while (!st.empty() && (anti_Clockwise(trees[st.top()], trees[top], trees[i]) < 0)) {
				top = st.top();
				st.pop();
			}
			st.emplace(top);
			st.emplace(i);
		}

		vector<vector<int>> ans;
		while (!st.empty()) {
			ans.emplace_back(trees[st.top()]);
			st.pop();
		}
		return ans;
	}
};



/*
// define 和 typedef在定义指针时的不同
#define INT_D int*
typedef int* int_p;

int main() {
	INT_D a, b;
	int_p c, d;
	cout << "a type: " << typeid(a).name() << endl;
	cout << "b type: " << typeid(b).name() << endl;
	cout << "c type: " << typeid(c).name() << endl;
	cout << "d type: " << typeid(d).name() << endl;
}
// 输出时，b的类型为int，而非指针
*/



class BB;
class AA {
public:
	shared_ptr<BB> ptr;
	~AA() {
		cout << "A delete" << endl;
	}

};
class BB {
public:
	shared_ptr<AA> ptr;
	~BB() {
		cout << "B delete" << endl;
	}
};
void fun() {
	shared_ptr<AA> pa(new AA());
	shared_ptr<BB> pb(new BB());
	pa->ptr = pb;
	pb->ptr = pa;
	cout << pa.use_count() << endl;
	cout << pb.use_count() << endl;
}
/*
int main() {
	fun();
	return 0;
}
*/


// traits
/*
template <typename T>
struct iterator_traits{
	typedef typename T::value_type value_type;
};

template <typename T>
typename iterator_traits<T>::value_type func(T ite) {
	return *ite;
}

// partial specialization
template <typename T>
struct iterator_traits<T*> {  // 迭代器是个原生指针
	typedef T value_type;
};

// 当迭代器是个 pointer-to-const(指向常数对象的指针，如 const int 而非 int)
// 令其萃取出来的类型为 T 而 非T、
template <typename T>
struct iterator_traits<const T*> {
	typedef T value_type;    // 萃取出来的型别是T而非const T
};
// 现在，不论面对的是迭代器，或是原生指针int*， 或是const int*，都能萃取出正确的value_type。

*/

// traits
// C++  容器中自定义类型 value_type
template <typename T>
class Bag {
public:
	typedef T value_type;
	T val;
	Bag(T value) :val(value) {}
};

template <typename T>
class Box {
public:
	typedef T value_type;
	T val;
	Box(T value) :val(value) {}
};

template <typename T>
T::value_type unpack(T container) {
	cout << container.val <<"  " << typeid(container.val).name() << endl;
	return container.val;
}
/*
int main() {
	Bag bg(10);
	Box bx(20.0);
	unpack(bg);
	unpack(bx);
	return 0;
}
*/


// 自定义unordered_map 的hash函数
struct pair_hash {
	template <typename T1, typename T2>
	std::size_t operator()(const std::pair<T1, T2>& p) const{
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);
		return h1 ^ h2;
	}
};
// pair 作为 map 的 key
unordered_map<pair<int, int>, int, pair_hash> mpp;


// 417. 太平洋大西洋水流问题
// bfs / dfs 边界反向搜索
class Solution417 {
public:
	vector<vector<int>> steps{ {-1,0},{0,1},{1,0},{0,-1} }; // 上 右 下 左
	vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
		// 四次bfs， 最后找公共位置（能到达两个海洋的顶点）
		
		std::function<void(int, int, vector<vector<int>>&, vector<vector<int>>&)> bfs =
			[&](int x, int y, vector<vector<int>>& visited, vector<vector<int>>& heights)->void {
			if (visited[x][y] == 1) {
				return;
			}
			visited[x][y] = 1;
			int m = heights.size();
			int n = heights[0].size();
			for (int i = 0; i < 4; ++i) {
				int new_x = x + steps[i][0];
				int new_y = y + steps[i][1];
				if (new_x >= 0 && new_x < m && new_y >= 0 && new_y<n && heights[new_x][new_y]>=heights[x][y]) {
					bfs(new_x, new_y, visited, heights);
				}
			}
		};

		int m = heights.size();
		int n = heights[0].size();
		vector<vector<int>> vec1(m, vector<int>(n, 0));
		vector<vector<int>> vec2(m, vector<int>(n, 0));
		for (int i = 0; i < m; ++i) { // 行遍历
			bfs(i, 0, vec1, heights); //左
			bfs(i, n - 1, vec2, heights); //右
		}
		for (int i = 0; i < n; ++i) { //列遍历
			bfs(0, i, vec1, heights); // 上
			bfs(m-1, i, vec2, heights); // 下
		}
		
		vector<vector<int>> ans;
		// 找相同的顶点
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (vec1[i][j] & vec2[i][j]) {
					ans.push_back({ i,j });
				}
			}
		}
		return ans;
	}
};


template <class T>
int compare_test(const T& t1, const T& t2) {
	return t1 >= t2 ? 1 : -1;
}

// specialization
template<>
int compare_test(const char* const& t1, const char* const& t2)
{
	return strcmp(t1, t2);
}


// 类模板
template <typename T> class Foo_test {
public:
	void Bar() {}
	void Barst(T a) {
		cout << "barst" << endl;
	}
};
// 特例化
//  类模板中的成员函数放到类模板定义外写时，语法为
/*
template<类型参数表>
返回值类型 类模板名<类型参数名列表>::成员函数名(参数表)
{
...
}
*/
template <>
void Foo_test<int>::Bar() {
	cout << "int 特例化" << endl;
}

/*
int main() {
	Foo_test<string> fss;
	Foo_test<int> fii;
	fss.Bar();  //使用普通模板，即 Foo<string>::Bar()
	fii.Bar();  // 使用特例化模板，即Foo<intn>::Bar()
	return 0;
}
*/


// 可变参数函数模板
template<typename T>
ostream& print(ostream& os, const T& t) {  //流对象不可复制，切记不要忘记引用
	return os << t;
}
template<typename T, typename... args>
ostream& print(ostream& os, const T& t, const args&... rest) {
	os << t << " , ";
	return print(os, rest...);
}

/*
int main() {
	print(cout, "i", "s", 42, 100);
	// output: i , s , 42 , 100
	return 0;
}
*/


// 6049. 含最多 K 个可整除元素的子数组
int countDistinct(vector<int>& nums, int k, int p) {
	unordered_map<string, int> mp;
	for (int i = 0; i < nums.size(); i++) {
		int count = k;
		string temp = "";
		for (int j = i; j < nums.size(); j++) {
			if (nums[j] >= p && nums[j] % p == 0) {
				count--;
			}
			if (count < 0) {
				break;
			}
			temp += (to_string(nums[j]));
			temp += ",";
			// 当前位置可以选
			mp[temp]++;
		}
	}
	return mp.size();
}

//=================================================== 双指针解决 子数组/子字符串 的问题   乘积，加法等

//6050. 字符串的总引力
/*
分类讨论：

如果 s[i] 之前没有遇到过，那么每个子串的引力值都会增加 1，引力值之和会增加 i，再加上 1，即 s[i] 单独组成的子串的引力值；
如果 s[i] 之前遇到过，设其上次出现的下标为 j，
那么向子串 s[0..i-1], s[1..i-1], s[2..i-1],...,s[j..i-1]的末尾添加 s[i] 后，引力值是不会变化的，因为 s[i]s[i] 已经在 s[j]s[j] 处出现过了，
因此只有i−j−1 个子串的引力值会增加 1，引力值之和会增加 i−j−1，再加上 1，即 s[i]s[i] 单独组成的子串的引力值。

*/
class Solution6050 {
public:
	using LL = long long;
	long long appealSum(string s) {
		LL ans = 0l;
		LL pre_sum = 0l;
		vector<int> vec(26, -1); //记录每个字符上次出现的位置
		for (int i = 0; i < s.size(); ++i) {
			pre_sum += (vec[s[i] - 'a'] == -1 ? i - vec[s[i] - 'a'] : i - vec[s[i] - 'a']);  // 没出现过   出现过
			// 因为默认下标为-1， 所以少+1
			vec[s[i] - 'a'] = i;
			ans += pre_sum;
		}
		return ans;
	}
};


// 713. 乘积小于 K 的子数组
class Solution713 {
public:
	int numSubarrayProductLessThanK(vector<int>& nums, int k) {
		int n = nums.size();
		int left = 0, right = 0;
		int ans = 0;
		long long temp = 1;
		while (left <= right && right < n) {
			temp *= nums[right];
			while (temp >= k && left <= right) {
				temp /= nums[left];
				++left;
			}
			ans += right - left + 1;
			++right;
		}
		return ans;
	}
};



// 591. 标签验证器
// hard
// 类似于HTML的解析
class Solution591 {
public:
	bool isValid(string code) {
		/*
		<tagname>  xxxxxx   </tagname>   ctag   tagname
		<![CDATA[ xxxxx ]]>   cdata_content(可包含任意字符，均视为常规字符)
		*/
		int n = code.size();
		int i = 0;
		stack<string> st;
		while (i < n) {
			if (code[i] == '<') {  // 边界左侧
				if (code[i + 1] == '/') {  // 右边界开始，  找 >
					int right_index = code.find_first_of('>', i + 1);
					if (right_index == string::npos) {  // 没找到 >
						return false;
					}
					string tag_name = code.substr(i + 2, right_index - i - 2);
					// 看栈里的tagname是否匹配
					if (st.empty() || st.top() != tag_name) { // 不匹配
						return false;
					}
					// 匹配
					st.pop();
					i = right_index + 1;
					if (st.empty() && i != n) {  // 栈内已空，如果i未结束，说明在标签外还有内容
						return false;
					}
				}
				else if(code[i+1] == '!') {  // CDATA左边界开始, 忽略内容，只判断括号 <![CDATA[    ]]>
					if (st.empty()) {
						return false;
					}
					string left_brackets = code.substr(i, 9);
					if (left_brackets != "<![CDATA[") {
						return false;
					}
					int right_cdata = code.find("]]>", i);
					if(right_cdata == string::npos){
						return false;
					}
					// cdata满足括号匹配
					i = right_cdata + 1;
				}
				else {  // 普通的左括号< >, 找到第一个tagname
					int right_index = code.find(">", i);
					if (right_index == string::npos) {
						return false;
					}
					string tagname = code.substr(i + 1, right_index - i - 1);
					// 判断tagname是否合法
					if (tagname.size() < 1 || tagname.size() > 9) {
						return false;
					}
					if (!all_of(tagname.begin(), tagname.end(), [&](const char& a) {return isupper(a); })) {
						return false;
					}
					st.push(tagname);
					i = right_index + 1;
				}
			
			}
			else {
				if (st.empty()) {
					return false;
				}
				++i;
			}
		}
		return st.empty();
	}
};


/*
* 433. 最小基因变化
将bfs应用到字符串上， 是个很好的例题， 也可以看为最短路径
keyword： bfs， 字符串bfs，  字符串转变一位
*/
class Solution433 {
public:
	int minMutation(string start, string end, vector<string>& bank) {
		// 变换后的序列一定要在bank之中
		// queue 实现bfs
		// hashmap 查重
		// 每次把bank中能直接转换过去（一位不同）的序列加到图中
		queue<pair<string, int>> que; // int存储当前步数
		unordered_map<string, int> mp;
		que.push({ start,0 });
		mp[start]++;

		auto judge = [&](const string& a, const string& b)->bool {
			int count = 0;
			int n = a.size();
			for (int i = 0; i < n; ++i) {
				if (a[i] != b[i]) {
					++count;
				}
			}
			return count == 1;
		};

		while (!que.empty()) {
			int len = que.size();
			for (int i = 0; i < len; ++i) {
				auto temp = que.front();
				que.pop();
				for (auto& a : bank) {
					if (judge(temp.first, a) && !mp.count(a)) {
						que.push({ a,temp.second + 1 });
						mp[a]++;
						if (a == end) {
							return temp.second + 1;
						}
					}
				}
			}
		}
		return -1;
	}
};

// 好家伙， 和433最小基因变化一模一样
// 127.单词接龙
class Solution127 {
public:
	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
		queue<pair<string, int>> que; // int存储当前步数
		unordered_map<string, int> mp; // 去重
		unordered_map<string, string> mp1; // 保存前驱节点
		que.push({ beginWord,0 });
		mp[beginWord]++;

		auto judge = [&](const string& a, const string& b)->bool {
			int count = 0;
			int n = a.size();
			for (int i = 0; i < n; ++i) {
				if (a[i] != b[i]) {
					++count;
				}
			}
			return count == 1;
		};

		while (!que.empty()) {
			int len = que.size();
			for (int i = 0; i < len; ++i) {
				auto temp = que.front();
				que.pop();
				for (auto& a : wordList) {
					if (judge(temp.first, a) && !mp.count(a)) {
						que.push({ a,temp.second + 1 });
						mp[a]++;
						mp1[a] = temp.first;
						if (a == endWord) {
							vector<string> ans{ a };
							while (a != beginWord) {
								a = mp1[a];
								ans.push_back(a);    // 懒得改了，直接把一条路径存进来了
							}
							return ans.size();
						}
					}
				}
			}
		}
		return 0;
	}
};

// 126. 单词接龙 II
/* 和127 单词接龙的区别就是， 单词接龙I 只需要输出路径的长度，即使一条路径也满足
* II 需要输出所有的最短路径，map的去重，只要保证序列一直向前就行，不要回退。
*/
class Solution126 {
public:
	int step;
	string beginw;
	queue<pair<string, int>> que;
	unordered_map<string, int> mp1; //去重
	unordered_map<string, vector<string>> mp2; // 保存一个点所有的前缀
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
		int n = beginWord.size();
		que.push({ beginWord,0 });
		beginw = beginWord;
		auto judge = [&](const string& a, const string& b)->bool {
			int count = 0;
			int n = a.size();
			for (int i = 0; i < n; ++i) {
				if (a[i] != b[i]) {
					++count;
				}
			}
			return count == 1;
		};
		
		step = INT_MAX;
		unordered_map<string, int> que_count;
		while (!que.empty()) {
			int n = que.size();
			for (int i = 0; i < n; ++i) {
				auto temp = que.front();
				que.pop();
				mp1[temp.first]++;
				vector<string> rec;
				for (auto& a : wordList) {
					if (judge(temp.first, a) && !mp1.count(a)) {
						rec.push_back(a);
						mp2[a].push_back(temp.first); // 前继结点
						if (!que_count.count(a)) {
							que.push({ a, temp.second + 1 });
							que_count[a]++;
						}
						if (a == endWord) {
							step = min(step, temp.second + 1);
						}
					}
				}
			}
			if (que_count.count(endWord)) {
				break;
			}
		}
		vector<vector<string>> ans;
		std::function<void(vector<vector<string>>&, string, vector<string>, int)> getAns = [&](vector<vector<string>>& ans, string node, vector<string> temp, int steps)->void {
			temp.push_back(node);
			if (steps > step) {
				return;
			}
			
			if (steps == step && node == beginw) {
				reverse(temp.begin(), temp.end());
				ans.push_back(temp);
				return;
			}

			if (mp2.count(node)) {
				for (auto& a : mp2[node]) {
					getAns(ans, a, temp, steps+1);
				}
			}
			return;
		};

		vector<string> temp;
		if (que_count.count(endWord)) {
			getAns(ans, endWord, temp, 0);
		}
		return ans;
	}
};


// stable_sort 中， return false时，数字相对位置不变
// find是查找子串，而find_first_of类似于模式匹配，只要与其中的一个字符匹配就行。


//23. 合并K个升序链表
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution123213 {
public:
	struct ListNode {
		int val;
		ListNode* next;
		ListNode() : val(0), next(nullptr) {}
		ListNode(int x) : val(x), next(nullptr) {}
		ListNode(int x, ListNode* next) : val(x), next(next) {}
	};

	/**
	* 方法1 考虑普通两个链表的合并，以第一个链表为基础，逐个合并到第一个链表
	*/
	ListNode* mergeTwo(ListNode* a, ListNode* b) {
		if (a == nullptr || b == nullptr) {
			return a == nullptr ? b : a;
		}
		ListNode head, *tail = &head, *aptr = a, *bptr = b;
		while (aptr && bptr) {
			if (aptr->val < bptr->val) {
				tail->next = aptr;
				aptr = aptr->next;
			}
			else {
				tail->next = bptr;
				bptr = bptr->next;
			}
		}
		// 未完的，直接接到后边
		tail->next = aptr == nullptr ? bptr : aptr;
		return head.next;
	}

	ListNode* mergeKLists_case1(vector<ListNode*>& lists) {
		ListNode* ans = nullptr;
		for (int i = 0; i < lists.size(); ++i) {
			ans = mergeTwo(ans, lists[i]);
		}
		return ans;
	}

	/**
	* 分治法合并，每次合并两个。从lists中从小到大
	
	ListNode* merge(vector<ListNode*>& lists, int l, int r) {
		if (l == r) {
			return lists[l];
		}

		int mid = l + (r - l) / 2;
		mergeTwo()
	}

	ListNode* mergeKLists_case2(vector<ListNode*>& lists) {
		

	}*/



};



/**
* 6058. 统计打字方案数
* dp
* 思路是按数字分组，判断每组的可能方案，将它们乘起来。例如 222 有四种可能，33 有两种可能，那么 22233 就有 2*4=8 种可能。
多少种可能是怎么来的？以 ...222(k 个 2) 为例，在它后面继续增加一个 2 可以有：
直接增加一个 2
...222(k 个 2) + 2 = ...222(k 个 2) 种可能

新加入的 2 也可以和原来最后一个 2 组合
...22(k-1 个 2) + 22 = ...22(k-1 个 2) 种可能

也可以和原来最后一个 22 组合
...2(k-2 个 2) + 222 = ...2(k-2 个 2) 种可能
把他们都加起来就是 k+1 个 2 的可能方案数，对于数字 7 和 9，他们可以是连续四个字符组合。
*/
class Solution6058 {
public:
	int MOD = 1e9 + 7;
	vector<long> vec3;
	vector<long> vec4;
	int countTexts(string pressedKeys) {
		// n最大为100000
		vec3.resize(100001);
		vec4.resize(100001);
		vec3[0] = 1;
		vec3[1] = 1;
		vec3[2] = 2;
		vec3[3] = 4;
		vec4[0] = 1;
		vec4[1] = 1;
		vec4[2] = 2;
		vec4[3] = 4;
		for (int i = 4; i < 100001; ++i) {
			vec3[i] = (vec3[i - 1] + vec3[i - 2] + vec3[i - 3]) % MOD;
			vec4[i] = (vec4[i - 1] + vec4[i - 2] + vec4[i - 3] + vec4[i - 4]) % MOD;
		}
		// 已经计算出了所有长度下  连续重复字符  的组合数
		// 开始处理原字符， 若为重复序列，判断按键字母长度是3还是4
		int i = 0;
		long long ans = 1ll;
		int n = pressedKeys.size();
		while (i < n - 1) {
			int j = i;
			int len = 1;
			while (j < n - 1 && pressedKeys[j] == pressedKeys[j + 1]) {
				++j;
			}
			if (pressedKeys[i] == '7' || pressedKeys[i] == '9') {
				ans = (ans * vec4[j - i + 1]) % MOD;
			}
			else {
				ans = (ans * vec3[j - i + 1]) % MOD;
			}
			i = j + 1;
		}
		return static_cast<int>(ans);
	}
};



/**
* 6059. 检查是否有合法括号字符串路径    
* 一开始我使用二维visited来记录
* 但是会超时
* 于是改用  记忆化搜索剪枝
* visited改为三维，   第三维记录的是此路径的括号值
*/
class Solution4444444 {
public:
	vector<vector<int>> step{ {1,0},{0,1} };
	vector<vector<int>> visited;
	bool hasValidPath(vector<vector<char>>& grid) {
		int m = grid.size();
		int n = grid[0].size();
		int ans = 0;
		vector<vector<vector<int>>> visited(m, vector<vector<int>>(n, vector<int>(m + n, 0)));
		int st=0;   // ’（‘加一   ’）‘减一

		std::function<void(vector<vector<char>>&, int, vector<vector<vector<int>>>&,int, int)> dfs = [&](vector<vector<char>>& grid, int st, vector<vector<vector<int>>>& visited,int x, int y)->void {
			if (st > (m - x + n - y)) {
				return;
			}
			
			if (grid[x][y] == ')') {
				if (st>0) {
					--st;
				}
				else {
					return;
				}
			}
			else {
				++st;
			}
			if (st==0 && (x == m - 1 && y == n - 1)) {
				ans = 1;
				return;
			}
			if (visited[x][y][st]) {
				return;
			}
			visited[x][y][st] = 1;
			for (auto& a : step) {
				int new_x = x + a[0];
				int new_y = y + a[1];
				if (new_x >= 0 && new_x < m && new_y >= 0 && new_y < n && !visited[new_x][new_y][st]) {
					dfs(grid,st, visited, new_x, new_y);
				}
			}
			return;
		};
		dfs(grid, st, visited,0, 0);
		return ans == 1;
	}
};



// 448. 找到所有数组中消失的数字
// [1,n]范围内的数在长度为n的数组中，各数字出现了一次，但是现在丢失了几个数字，找出丢失的数字。
// 构造原数组本身充当哈希表。
vector<int> findDisappearedNumbers(vector<int>& nums) {
	int n = nums.size();
	for (auto& a : nums) {
		int index = (a - 1) % n; //即使加了n也能恢复到原数值
		nums[index] += n;
	}
	vector<int> ans;
	for (int i = 0; i < n; ++i) {
		if (nums[i] <= n) {
			ans.push_back(i + 1);
		}
	}
	return ans;
}


// 442. 数组中重复的数据
//给你一个长度为 n 的整数数组 nums ，其中 nums 的所有整数都在范围 [1, n] 内，且每个整数出现 一次 或 两次 。
//请你找出所有出现 两次 的整数，并以数组形式返回。
// 把原数组当作哈希表
/**
* note: 如果按照448那样子进行+n操作，因为数组中一个元素能出现多次，当输入为[2,2]
* 即长度为2，且 最终的答案也是2，    若进行+n操作
* 可以看到，第一次操作，index=（2-1）%n=1   数组变为[2,4]
* 第二次操作，到位置4，发现大于n，那么存入（4%2）=0，错误
* 
* 对于输入[4,3,2,7,8,2,3,1]，n=8 前五次操作后，可得到 [12,11,10,15,8,2,11,1] 
* 当 i=5时，nums[i]==2， 那么（2-1）%8=1, nums[1]==11>n   ，存入nums[i]%n=2    
* i=6, nums[(3-1)%8] == 10>n 存入nums[i]%n=3  ... 这种是可以满足的
* 当等于数组长度的数 第二次出现正好早最后一位，那么不行
*/
vector<int> findDuplicates(vector<int>& nums) {
	int n = nums.size();
	vector<int> ans;
	for (int i = 0; i < n; ++i) {
		int index = (abs(nums[i]) - 1);
		if (nums[index] < 0) {  // 出现过，恢复原数值
			ans.push_back(abs(nums[i]));
		}
		else {
			nums[index] *= -1;
		}
	}
	return ans;
}



// 449. 序列化和反序列化二叉搜索树
// 可以利用二叉搜索树中序遍历是有序的性质
class Code449 {
public:
	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};

	// Encodes a tree to a single string.
	// 前序遍历  和 中序遍历  能确定一颗二叉树   中序遍历和后序遍历能确定一颗二叉树
	// 但是对于二叉搜索树，其前序遍历和后序遍历都能确定一颗唯一的二叉树
	string serialize(TreeNode* root) {
		vector<int> vec;
		string res = "";
		postOrder(root, vec);
		if (vec.empty()) {
			return res;
		}
		for (auto& a : vec) {
			res.append(to_string(a) + ",");
		}
		res.pop_back();  // 去掉最后一个 ","
		return res;
	}

	// 后序遍历  递归法
	void postOrder(TreeNode* const root, vector<int>& vec) {
		if (root == nullptr) {
			return;
		}
		postOrder(root->left, vec);
		postOrder(root->right, vec);
		vec.emplace_back(root->val);
	}

	// 后序遍历  非递归
	// 非递归使用栈的方式实现
	void postOrder_iter(TreeNode* const root, vector<int>& vec) {
		stack<TreeNode*> st;
		TreeNode* temp = root;
		int visited_node = 0;  //按value来记录最近访问过的结点,主要是判断是否要去右子树
		while (temp != nullptr || !st.empty()) {
			// 先走到最左边
			if (temp != nullptr) {
				st.push(temp);
				temp = temp->left;
			}
			else {  // 已经走到最左边了
				// 看是否最左边的结点还有右子树
				temp = st.top();
				if (temp->right != nullptr && temp->right->val != visited_node) {  //没访问过
					// 到其右子树，再找右子树的最左结点
					temp = temp->right;
					st.push(temp);
					temp = temp->left;
				}
				else { //访问过了
					st.pop();  // 弹出当前节点
					vec.push_back(temp->val);
					visited_node = temp->val;
					temp = nullptr;
				}
			}			
		}
	}


	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) { // 根据后序遍历还原
		int lower = INT_MIN;
		int upper = INT_MAX;
		vector<int> vec;
		// 解析一下 data
		stringstream ss; 
		ss << data;
		string temp = "";
		while (std::getline(ss, temp, ',')) {
			vec.emplace_back(stoi(temp));
		}
		stack<int> st;
		for (auto& a : vec) {
			st.push(a);
		}
		return construct(lower, upper, st);
	}

	TreeNode* construct(int lower, int upper, stack<int>& st) {
		if (st.empty() || st.top() < lower || st.top() > upper) {
			return nullptr;
		}
		int val = st.top();
		st.pop();
		TreeNode* root = new TreeNode(val);
		root->right = construct(val, upper, st);  // 右边的都大于val ，即val为lower bound
		root->left = construct(lower, val, st);  //左边的都小于val ，即val为upper bound
		return root;
	}
};



// 297. 二叉树的序列化与反序列化
class Codec {
public:
	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};

	// Encodes a tree to a single string.
	string serialize(TreeNode* root) {
		string res = "";
		TreeNode* p = root;
		preOrder(p, res);
		// 使用递归，把最后一个逗号删除
		res.pop_back();

		// 使用迭代，最后还少个"X"
		// preOrder_case2(p, res);
		// res += "X";

		return res;
	}

	// 使用递归方式
	void preOrder(TreeNode* root, string& str) {
		if (root == nullptr) {
			str += "X";
			str += ",";
			return;
		}
		str += to_string(root->val);
		str += ",";
		preOrder(root->left, str);
		preOrder(root->right, str);
	}

	// 使用 非递归方式 进行前序遍历
	void preOrder_case2(TreeNode* root, string& str) {
		stack<TreeNode*> st;
		st.push(root);
		str = str + to_string(root->left->val) + ",";
		TreeNode* temp = root;
		int visited_node = 0;
		while (temp!= nullptr || !st.empty()) {
			if (temp!=nullptr) {  // 一直向左走
				st.push(temp);
				str += to_string(temp->val);   // 不断保存经过的结点
				str += ",";
				temp = temp->left;  // 走到最左
			}
			else { // 找右子树
				str += "X";   // 不断保存经过的结点
				str += ",";
				temp = st.top();  // 回溯到父结点
				st.pop();
				temp = temp->right;
				
			}
		}
	}


	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) {  // 只给了一个string ，说明不能通过 前/中 + 后序  来确定唯一的二叉树
		deque<string> que;
		stringstream ss;
		string temp;
		ss << data;
		while (getline(ss, temp, ',')) {
			que.push_back(temp);
		}
		// 按照 ’,‘ 分割好了
		return anls(que);

	}

	// 从左到右解析字符串
	TreeNode* anls(deque<string>& que) {
		// que 从前往后
		if (!que.empty() && que.front() == "X") {  // 为空结点
			que.pop_front();
			return nullptr;
		}
		auto temp = new TreeNode(stoi(que.front()));
		que.pop_front();
		temp->left = anls(que);
		temp->right = anls(que);
		return temp;
	}
};

// 中序遍历  非递归实现



// 712. 两个字符串的最小ASCII删除和
// 给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和 。
// 转身去求  最大ASCII码的公共字符串，这样的话删除量就最小
// 1143. 最长公共子序列
int minimumDeleteSum(string s1, string s2) {
	int m = s1.size();
	int n = s2.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 1; i <= m; ++i) {    // 求最长公共子序列
		for (int j = 1; j <= n; ++j) {
			if (s1[i - 1] == s2[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1] + s2[j - 1];
			}
			else {
				dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]); 
			}
		}
	}
	int asc1 = 0, asc2 = 0;
	for (auto& a : s1) {
		asc1 += a;
	}
	for (auto& a : s2) {
		asc2 += a;
	}
	return asc1 + asc2 - 2 * dp[m][n];
}



// 236. 二叉树的最近公共祖先
class Solution236 {
public:
	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) { //  递归方式求解
		if (root == nullptr) {
			return nullptr;
		}
		if (root == p || root == q) {
			return root;  // 找到目标值后，返回当前父结点
		}
		TreeNode* left = lowestCommonAncestor(root->left, p, q);
		TreeNode* right = lowestCommonAncestor(root->right, p, q); // 从left得到的父结点继续往下找
		
		// 如果分别在以root为根节点的左右子树上
		if (left != nullptr && right != nullptr) {
			return root;
		}
		else if (left != nullptr) { // 左不空，右空，说明p q均在以root为根的左子树
			return left;
		}
		else { // 左空，右不空，说明p q均在以root为根的右子树
			return right;
		}
	}

	// 保存到两个节点的路径，找两条路径上的最后一个公共结点
	// 测试用例过大会超时
	TreeNode* lowestCommonAncestor_find_con(TreeNode* root, TreeNode* p, TreeNode* q) {
		vector<TreeNode*> ans1;
		vector<TreeNode*> ans2;
		vector<TreeNode*> temp1, temp2;
		dfs(root, temp1, ans1, p->val);
		dfs(root, temp2, ans2, q->val);
		int n = (ans1.size() <= ans2.size()) ? ans1.size() : ans2.size();
		TreeNode* ans = nullptr;
		for (int i = 0; i < n; ++i) {
			if (ans1[i] == ans2[i]) {
				ans = ans1[i];
			}
		}
		return ans;
	}
	void dfs(TreeNode* root, vector<TreeNode*> vec, vector<TreeNode*>& ans, int target) {
		// 边界
		if (root == nullptr) {
			return;
		}
		vec.push_back(root);
		if (root->val == target) {
			ans = vec;
			return;
		}
		dfs(root->left, vec, ans, target);
		dfs(root->right, vec, ans, target);
	}
};


// ==================================================================  下一个系列  ================================================
// 
//面试题 05.04. 下一个数
// 给定一个正整数，找出与其二进制表达式中1的个数相同且大小最接近的那两个数（一个略大，一个略小）。
class Next0504 {
public:
	vector<int> findClosedNumbers(int num) {
		if (num == 1) {
			return { 2,-1 };
		}
		if (num == 2147483647) {
			return { -1,-1 };
		}
		// 预先将num转为0-1数组
		// 找大的数， 遇到01  就转换为10， 并把右侧所有的1移到低位   ... 01 ...  =>  ... 10 ...
		// 找小的数， 遇到10， 就转换为01， 并把右侧所有的1移到高位
		vector<int> str;
		for (int i = 0; i < 32; ++i) {
			str.push_back(num & 1);
			num >>= 1;
		}

		vector<int> gnum = str;
		int maxtag = -1;
		for (int i = 0; i < str.size() - 1; ++i) {  //找 01  从高位向低位
			if (str[i] == 1  && str[i + 1] == 0) {  // 存储顺序是反的
				maxtag = 1;
				swap(gnum[i], gnum[i + 1]);
				int left = i - 1, right = 0;
				while (left > right) {
					while (left > right && gnum[left] == 1 && gnum[right] == 0) {
						swap(gnum[left], gnum[right]);
						left--;
						right++;
					}
					while (left > right && gnum[left] != 1) {
						--left;
					}
					while (left > right && gnum[right]!=0) {
						++right;
					}
				}
				break;
			}
		}
		
		// 找小 
		int mintag = -1;
		for (int i = 0; i < str.size() - 1; ++i) {
			if (str[i] == 0 && str[i + 1] == 1) {
				mintag = 1;
				swap(str[i], str[i + 1]);
				int left = i - 1, right = 0;
				while (left > right) {
					while (left > right && str[left] == 0 && str[right] == 1) {
						swap(str[left], str[right]);
						left--;
						right++;
					}
					while (left > right && str[left] != 0) {
						left--;
					}
					while (left > right && str[right] != 1) {
						right++;
					}
				}
				break;
			}
		}
		int ans1 = 0, ans2 = 0;
		int tag1 = -1;
		for (int i = gnum.size() -1; i >= 0; --i) {
			if (gnum[i] == 1) {
				tag1 = i;
			}
			if (tag1 != -1) {
				ans1 <<= 1;
				ans1 |= gnum[i];
				
			}
		}
		tag1 = -1;
		for (int i = str.size() - 1; i >= 0; --i) {
			if (str[i] == 1) {
				tag1 = i;
			}
			if (tag1 != -1) {
				ans2 <<= 1;
				ans2 |= str[i];
				
			}
		}
		if (mintag == -1) {
			ans2 = -1;
		}
		if (maxtag == -1) {
			ans1 = -1;
		}
		return { ans1,ans2 };
	}
};

// 496. 下一个更大元素 I
// nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素
class Next496 {
public:
	vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
		// 建个hash吧，不然查询很花时间
		// 常见的下一个更大的数，常用单调栈
		unordered_map<int, int> mp;
		int n = nums2.size();  // 记录一下最大值
		stack<int> st;  // 单调栈保存右侧所有的数字
		st.push(nums2[n - 1]);
		mp[nums2[n - 1]] = -1;

		for (int i = nums2.size() - 2; i >= 0; --i) {
			// 找右侧比当前值大的
			while (!st.empty() && nums2[i] > st.top()) {  // 更新单调栈
				st.pop();
			}
			// 找到了， 或者没找到栈空了
			if (!st.empty()) {  // 找到了
				mp[nums2[i]] = st.top();
			}
			else {
				mp[nums2[i]] = -1;
			}
			st.push(nums2[i]);
		}
		vector<int> ans;
		for (int i = 0; i < nums1.size(); ++i) {
			ans.push_back(mp[nums1[i]]);
		}
		return ans;
	}
};

// 503. 下一个更大元素 II
class Next503 {
public:
	// 或者吧nums拼接做，最后结果取一半，也可以
	vector<int> nextGreaterElements(vector<int>& nums) {
		int n = nums.size();
		vector<int> vec(n, -1);
		stack<int> st;
		for (int i = 0; i <= 2 * nums.size() - 1; ++i) {
			while (!st.empty() && nums[i % n] > nums[st.top()]) {
				vec[st.top()] = nums[i % n];
				st.pop();
			}
			st.push(i % n);
		}
		return vec;
	}
};

// 556. 下一个更大元素 III
// 给你一个正整数 n ，请你找出符合条件的最小整数，
// 其由重新排列 n 中存在的每位数字组成，并且其值大于 n 。如果不存在这样的正整数，则返回 -1 。
/*
	这题，也就是从同一数组中取完所有数字，来获得下一个更大的数字d
	把 n 拆分成一个个数字，然后就变成 leetcode31. 下一个排列了！
*/
// 从后往前，找到第一个升序排列的   如  12385764， 第一个升序排列是 5 7， 下标记为i 和 j， 
// 然后从[j, end], 从end开始，找到第一个大于 i位置的数 

/*
class Next556 {
public:
	vector<int> nextGreaterElements(vector<int>& nums) {
		
	}
};

// 31. 下一个排列
class Next31 {
public:

};*/



// 691. 贴纸拼词
/*
 dfs 搜索stickers是否适用并返回最少数量，注意剪枝和记忆化搜索
*/
class Solution691 {
public:
	unordered_map<string, int> mp;
	int minStickers(vector<string>& stickers, string target) {
		// 统计每个sticker中字母出现的数量
		int n = stickers.size();
		vector<vector<int>> vec(n, vector<int>(26, 0));
		for (int i = 0; i < n; ++i) {
			for (auto& a : stickers[i]) {
				vec[i][a - 'a']++;
			}
		}
		mp[""] = 0;
		return dfs(vec, target);
	}

	// 剪枝，考虑target剩下字符
	int dfs(vector<vector<int>>& vec, string target) {
		if (mp.count(target)) {
			return mp[target]; // 返回这个target的子集（包括本身）所需要的最少ticks
		}
		int ans = INT_MAX;
		// 记录一下当前target还剩哪些字符
		vector<int> rec(26, 0);
		for (auto& a : target) {
			rec[a - 'a']++;
		}

		// 来找stickers， 看哪些可以满足, stickers是可以重复使用的
		for (int i = 0; i < vec.size(); ++i) {
			// 验证
			if (vec[i][target[0] - 'a'] == 0) {   // 为什么剪枝要这么做
				continue;
			}
			/*		
			*/

			string left_target = "";  // 记录使用了这个sticker后，还剩多少字母
			for (int k = 0; k < 26; ++k) {
				if (rec[k] - vec[i][k] > 0) {
					// 当前sticker的字母不会超过target目前所拥有的
					left_target += string(rec[k] - vec[i][k], 'a' + k);  // constructor:   string (size_t n, char c);
				}
			}

			// 遍历完所有stickers一遍之后，剩下的target
			int use_count = dfs(vec, left_target); 
			if (use_count != -1) {
				ans = min(ans, use_count + 1); //加上当前这张
			}
		}
		mp[target] = (ans == INT_MAX ? -1 : ans);
		return mp[target];
	}
};


// 6065. 按位与结果大于零的最长组合
/*
 	1 <= candidates.length <= 10^5
	1 <= candidates[i] <= 10^7
	我一开始做了dfs， 超时。
*/
/*
* 位操作，  还是太菜
	要求 多个数按位与 大于0
	那么 这几个数按位与操作后，二进制一定不全为0，也就是说肯定至少有一位他们全是1
	我只要找到，所有的数字中，二进制中哪一位的1最多，也就意味着这些数 进行 与操作不会得到0
*/
class Solution123121233 {
public:
	int largestCombination(vector<int>& candidates) {
		int ans = 0;
		for (int i = 0; i <= 30; ++i) {  // 枚举每一位
			int count = 0;
			for (auto& a : candidates) {
				if ((a >> i) & 1) {
					count++;
				}
			}
			ans = max(ans, count);
		}
		return ans;
	}
};

// 剑指 Offer II 074. 合并区间
// 56. 合并区间
// 区间并集问题  区间合并
class Solution56 {
public:
	vector<vector<int>> merge(vector<vector<int>>& intervals) {
		sort(intervals.begin(), intervals.end());
		vector<vector<int>> ans;
		for (int i = 0; i < intervals.size(); ++i) {
			int l = intervals[i][0], r = intervals[i][1];
			if (!ans.empty() && l <= ans.back()[1]) {  // 当前区间的左边小于上个点的右边，满足合并 
				ans.back()[1] = max(ans.back()[1], r);
			}
			else {  // 不满足合并
				ans.push_back(intervals[i]);
			}
		}
		return ans;
	}
};



// 6066. 统计区间中的整数数目
/*
	区间并集问题
	给个set， 每次add之后，将相交的区间合并
		// 和普通区间合并问题的区别是，这个是边加边合并
		// 普通合并问题是 提前给完集合，进行合并
*/
class CountIntervals {
public:
	using PII = pair<int, int>;
	vector<PII> vec;
	set<PII> st;
	int ans = 0;
	CountIntervals() {
		
	}

	void add(int left, int right) {
		// 准备合并区间
		// 先找left落到其他区间右侧   [       ]   [       ]
		//                              L[       R]   这种形式, 要小心右侧还能合并，三合一
		//    l[     r]              l[        r]
		//        L[     R]     L[       R] 
		// 查找时，找到 第一个区间的右节点r大于L的区间  上两行，两种情况都能这么找
		/*
			存储pair的时候，反着存， 以{r,l}存入到set中
			{c, d} = lower_bound({a, b})是要(c > a)||(c == a && d >= b)
			那么  auto it = st.lower_bound({L, -1e9})  就是 找到 (r>L) || (r == L && l >= -1e9)
		*/
		// 先找插入点位
		int L = left;
		int R = right;
		auto it = st.lower_bound({ left - 1, -1e9 });  // 找到大于等于left的区间右侧, 即 r > L, 二者相邻也满足合并要求， 即r == L - 1 
		// l,r记为set中的区间边界， LR为插入区间的边界
		while (it != st.end()) {
			//插入点在当前区间内部，及 it->first <= left < right <= it->right
			// it->first  本身 就已经小于left
			if (it->second > right + 1) {  // l要大于R, 考虑二者相邻也能合并，即 R+1==l
				break;   //不相交， r > L  且 l > R
			}
			// 可以合并
			L = min(it->second, L);  // 可能连续更新，记录合并区间的边界
			R = max(it->first, R);

			ans -= (it->first - it->second + 1);  //把将要删除掉的区间，减掉里面的整数点
			st.erase(it++);  //it++ 是先记下 it 的值，然后 ++，所以等调用 erase 的时候 it 值已经变了，但是 erase 收到的还是原来的值。
			// it++返回右值，++it返回左值
		}
		ans += (R - L + 1);// 记录新的点数
		st.insert({ R, L });
	}

	int count() {
		return ans;
	}
};


/*
	对于第K小/大的数， 简单、中等题就使用优先队列，难题就是二分法
*/
// 668. 乘法表中第k小的数
class Solution668 {
public:
	/*
		每一列递增，统计每一列大于num的个数
	*/
	int count_col(int m, int n, int num) {
		int i = m, j = 1;
		int sum = 0;
		while (i >= 1 && j <= n) {
			if (i * j <= num) {
				sum += i;
				++j;
			}
			else {
				--i;
			}
		}
		return sum;
	}

	int findKthNumber(int m, int n, int k) {
		// 二分查找
		int left = 1, right = m * n;
		while (left < right) {  // 统计在num中的大于mid的个数
			int mid = left + (right - left) / 2;  // 如何保证mid在表中
			int count = count_col(m, n, mid);
			if (count < k) {
				left = mid + 1;
			}
			else {
				right = mid;
			}
		}
		return left;
	}
};


// 462. 最少移动次数使数组元素相等 II
class Solution462 {
public:
	int minMoves2_case1(vector<int>& nums) {  // 排序后取中位数，求和
		int n = nums.size();
		if (n == 1) {
			return 0;
		}
		sort(nums.begin(), nums.end());
		int mid = n / 2;
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			ans += abs(nums[i] - nums[mid]);
		}
		return ans;
	}

	// 快速选择算法， 找mid
	int minMoves2_case2(vector<int>& nums) {
		int n = nums.size();
		int mid = quickSort(nums, 0, n - 1, n / 2);
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			ans += abs(nums[i] - mid);
		}
	}

	int quickSort(vector<int>& nums, int left, int right, int mid) {
		int index = random_index(left, right);
		swap(nums[index], nums[right]);
		int rtn = partion(nums, left, right);
		if (rtn == mid) {
			return nums[rtn];
		}
		else if (rtn < mid) {  // 当前数在中位数左边， 开始从当前数的右边找，因为左边都是小于该数的
			return quickSort(nums, rtn + 1, right, mid);
		}
		else {
			return quickSort(nums, left, rtn - 1, mid);
		}
	}

	int random_index(int left, int right) {
		return rand() % (right - left + 1) + left;
	}

	int partion(vector<int>& nums, int left, int right) {
		int i = left - 1;
		for (int j = left; j <= right; ++j) {
			if (nums[j] < nums[right]) {
				swap(nums[++i], nums[j]);  // 小于pivot的都挪到左边去
			}
		}
		swap(nums[i+1], nums[right]);// pivot 归位
		return i+1;
	}
};

// 436. 寻找右区间
class Solution436 {
public:
	vector<int> findRightInterval(vector<vector<int>>& intervals) {
		// 要查找的是左侧大于另一个区间右侧的点
		// 二分查找左点，给个vector<pair<int, int>> 存储左端点和原先下标
		int n = intervals.size();
		vector<pair<int, int>> vec;
		for (int i = 0; i < n; ++i) {
			vec.emplace_back(make_pair(intervals[i][0], i));
		}
		sort(vec.begin(), vec.end());  // 左端点从小到大
		// 针对 每一个点的右端点  查找最小左端点
		vector<int> ans;
		for (int i = 0; i < n; ++i) {
			auto fd = lower_bound(vec.begin(), vec.end(), make_pair(intervals[i][1], 0));
			if (fd != vec.end()) {
				ans.emplace_back(fd->second);
			}
			else {
				ans.push_back(-1);
			}
		}
		return ans;
	}
};

// 6077. 巫师的总力量和
// 保存滑动窗口的最小值是比较困难的
// 这题思想和  1856. 子数组最小乘积的最大值  很像
// 单调栈 + 前缀和
// 巧用前缀和的前缀和
class Solution6077 {
public:
	int MOD = 1e9 + 7;
	int totalStrength(vector<int>& strength) {
		// 先找出每个元素x   左侧<=x的最近点(第一个点)，没有则默认-1  右侧 <=x 的最近点，没有则默认n
		// 单调栈实现找最小值的下标
		int n = strength.size();
		vector<int> left(n, -1), right(n, n);
		stack<int> st;
		for (int i = 0; i < n; ++i) {
			while (!st.empty() && strength[st.top()] >= strength[i]) st.pop();
			if (!st.empty()) left[i] = st.top();
			st.push(i);
		}

		while (!st.empty()) st.pop();
		for (int i = n - 1; i >= 0; --i) {
			while (!st.empty() && strength[st.top()] > strength[i]) st.pop();
			if (!st.empty()) right[i] = st.top();
			st.push(i);
		}

		// 求前缀和
		vector<int> psum(n + 1);
		for (int i = 1; i <= n; ++i) {
			psum[i] = (psum[i - 1] + strength[i - 1]) % MOD;
		}
		// 求前缀和的前缀和
		vector<long long> ppsum(n + 2);
		for (int i = 2; i <= n + 1; ++i) {
			ppsum[i] = (ppsum[i - 1] + psum[i - 1]) % MOD;
		}

		// 针对每个min值，计算其作用区间
		// 每个min值管辖范围为 [left[min] + 1, right[min] - 1]
		long long ans = 0;
		for (int i = 0; i < n; ++i) {
			long l = left[i] + 1, r = right[i] - 1;
			long long lsum = ((ppsum[i + 1] - ppsum[l]) * (r - i + 1));
			long long rsum = ((ppsum[r + 2] - ppsum[i + 1]) * (i - l + 1));
			long long ttt = (rsum - lsum) % MOD;
			ans = (ans + strength[i] * ttt) % MOD;
		}
		return (int)((ans + MOD) % MOD);
	}
};



// 464. 我能赢吗
// dfs + 记忆化搜索
class Solution464 {
public:
	unordered_map<int, bool> mem;
	bool canIWin(int maxChoosableInteger, int desiredTotal) {
		// 和石子游戏I有点像，但是这题没有按照顺序
		// 石子游戏I 只能取两端的数字
		if ((maxChoosableInteger * (maxChoosableInteger + 1)) / 2 < desiredTotal) {
			return false;
			// n(n+1)/2
		}
		return dfs(maxChoosableInteger, 0, desiredTotal, 0);
	}
	bool dfs(int maxChoosableInteger, int visited, int desiredTotal, int nowTotal) {
		// visited以二进制的方式存储被使用过的数字
		if (!mem.count(visited)) {
			// 开始选数字
			bool ret = false;
			for (int i = 0; i < maxChoosableInteger; ++i) {
				if (((visited >> i) & 1) == 0) {   // 这位没选过
				// 我选了这位能直接获胜
					if (i + nowTotal + 1 >= desiredTotal) {
						ret = true;
						break;
					}
					// 如果我选了这位不能直接获胜
					// 那么如果之后另一个人输了，也表示我能赢
					if (!dfs(maxChoosableInteger, visited | (1 << i), desiredTotal, nowTotal + i + 1)) {
						ret = true;
						break;
					}
				}
			}
			mem[visited] = ret;
		}
		return mem[visited];
	}
};

// 675. 为高尔夫比赛砍树
class Solution675 {
public:
	using PII = pair<int, int>;
	vector<vector<int>> steps{ {0,-1},{0,1},{-1,0},{1,0} };
	int cutOffTree(vector<vector<int>>& forest) {
		// 题目要求按照所有树从高到低的顺序砍完
		// 对所有树进行排序，并且保存下标
		vector<PII> vec;
		for (int i = 0; i < forest.size(); ++i) {
			for (int j = 0; j < forest[i].size(); ++j) {
				if (forest[i][j] > 1) {
					vec.emplace_back(i, j);
				}
			}
		}
		sort(vec.begin(), vec.end(), [&](const auto& a, const auto& b) {
			return forest[a.first][a.second] < forest[b.first][b.second];
			});
		int x = 0;
		int y = 0;
		int ans = 0;
		for (int i = 0; i < vec.size(); ++i) {  // 找到每个位置的路径
			int temp = bfs(x, y, vec[i].first, vec[i].second, forest);
			if (temp == -1) {
				return -1;
			}
			else {
				ans += temp;
				x = vec[i].first;
				y = vec[i].second;
			}
		}
		return ans;
	}

	// bfs能保证从矩阵中走过去是最短路径
	int bfs(int x, int y, int target_x, int target_y, vector<vector<int>>& forest) {
		int p = forest.size();
		int q = forest[0].size();
		if (x == target_x && y == target_y) {
			return 0;
		}
		queue<PII> que;
		vector<vector<int>> visited(p, vector<int>(q, 0));
		que.push({ x,y });
		visited[x][y] = 1;
		int step = 0;
		while (!que.empty()) {
			int n = que.size();
			step++;
			for (int i = 0; i < n; ++i) {
				auto temp = que.front();
				que.pop();
				for (int j = 0; j < 4; ++j) {
					int new_x = temp.first + steps[j][0];
					int new_y = temp.second + steps[j][1];
					if (new_x >= 0 && new_x < p && new_y >= 0 && new_y < q) {
						if (!visited[new_x][new_y] && forest[new_x][new_y] > 0) {
							if (new_x == target_x && new_y == target_y) {
								return step;
							}
							visited[new_x][new_y] = 1;
							que.push({ new_x, new_y });
						}
					}
				}
			}
		}
		return -1;
	}

};


//467. 环绕字符串中唯一的子字符串
int case467(string p) {
	vector<int> dp(26, 0);
	int n = p.size();
	int temp = 1;
	for (int i = 1; i < n; ++i) {
		if ((p[i] == 'a' && p[i - 1] == 'z') || (p[i] - 'a') == (p[i - 1] - 'a' + 26 + 1) % 26) {
			// 连续
			temp++;
		}
		else { // 不连续
			temp = 1;
		}
		dp[p[i] - 'a'] = max(dp[p[i] - 'a'], temp);
	}
	return accumulate(dp.begin(), dp.end(), 0);
}



// 795. 区间子数组个数
int numSubarrayBoundedMax(vector<int>& nums, int left, int right) {
	std::function<int(vector<int>&, int)> atMostK = [&](vector<int>& nums, int k)->int {
		int ans = 0;
		int temp = 0;
		int n = nums.size();
		for (int i = 0; i < n; ++i) {
			if (nums[i] <= k) {
				temp++;
			}
			else {
				temp = 0;
			}
			ans += temp;
		}
		return ans;
	};
	return atMostK(nums, right) - atMostK(nums, left - 1);
}


// 904. 水果成篮
int totalFruit(vector<int>& fruits) {
	int n = fruits.size();
	unordered_map<int, int> mp;
	int j = 0; // 记录前一种水果的起始下标
	int k = 2;
	int ans = 0;
	for (int i = 0; i < n; ++i) {
		if (mp[fruits[i]] == 0) { // 没使用过
			--k;
		}
		mp[fruits[i]]++;
		while (k < 0) {  // 有新水果进来，篮子已超出了两个
			// 删除前一种水果
			mp[fruits[j]]--;
			if (mp[fruits[j]] == 0) {
				++k;
			}
			j++;  // 删除多少个就往后移动多少位，因为是连续的，所以间接等同于移到前一种水果的最后一个位置
		}
		ans = max(ans, i - j + 1);
	}
	return ans;
}


// 992. K 个不同整数的子数组
int subarraysWithKDistinct992(vector<int>& nums, int k) {
	auto mostEqualK = [&](vector<int>& nums, int k)->int {
		unordered_map<int, int> mp;
		int j = 0; // 保存最早出现的数字
		int n = nums.size();
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			if (mp[nums[i]] == 0) {  // 没出现过，加进去
				--k; // 剩余的不同数目减一
			}
			mp[nums[i]]++;
			// 没出现过的数字加进去后发现超出了不同数字的要求值
			while (k < 0) {
				mp[nums[j]]--;
				if (mp[nums[j]] == 0) {  // 直至清空前面任何一个数字
					// 因为删除空的数字之前的那部分也是不能使用的，因为不连续
					++k; // k从-1归0， 此时正好用完
				}
				++j;
			}
			ans += (i - j + 1);  // 按照之前求countArray的方法
		}
		return ans;
	};
	return mostEqualK(nums, k) - mostEqualK(nums, k - 1);
}


// 1109. 航班预订统计
// 差分数组
vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
	// 差分数组
	vector<int> nums(n + 1, 0);
	for (int i = 0; i < i < bookings.size(); ++i) {
		nums[bookings[i][0] - 1] += bookings[i][2];
		nums[bookings[i][1]] -= bookings[i][2];
	}
	for (int i = 1; i < n; ++i) {
		nums[i] += nums[i - 1];
	}
	nums.pop_back();
	return nums;
}


// 6079. 价格减免
class Solution6079 {
public:
	string discountPrices(string sentence, int discount) {
		// stringstream 拆分字符串
		stringstream ss;
		string temp;
		vector<string> str;
		ss << sentence;
		

		auto judge = [&](const string& s) -> bool{
			if (s[0] != '$') {
				return false;
			}
			else {
				if (s.size() == 1) {
					return false;
				}
				int n = s.size();
				for (int i = 1; i < n; ++i) {
					if (s[i] < '0' || s[i] > '9') {
						return false;
					}
				}
				return true;
			}
			return false;
		};

		string ans = "";
		bool begin = 1;
		while (ss>>temp) {
			if (!begin) {
				ans += " ";
			}
			begin = 0;
			if (judge(temp)) {  // 当前是价格
				long num = 0;
				for (int j = 1; j < temp.size(); ++j) {
					num = num * 10 + (temp[j] - '0');
				}

				int d = 100 - discount;
				num = num * d;
				ans += '$' + to_string(num / 100) + '.';
				// ans = ans + '$' + to_string(num / 100) + '.';
				// 没写+=   直接超时超到死
				num %= 100;  // 取后两位
				if (num == 0) {
					ans += "00";
				}
				else {
					ans += to_string(num / 10) + to_string(num % 10);
				}
			}
			else {
				ans += temp;
			}
		}
		return ans;
	}
};


// 468. 验证IP地址
// 正则化 regex
string validIPAddress(string queryIP) {
	regex ipv4("((25[0-5]|2[0-4][0-9]?|1[0-9]?[0-9]?|0).){3}(25[0-5]|2[0-4][0-9]?|1[0-9]?[0-9]?|0)");
	regex ipv6("(([0-9a-fA-F]{1,4}):){7}([0-9a-fA-F]{1,4})");
	smatch mt;
	bool ret4 = regex_match(queryIP, mt, ipv4);
	bool ret6 = regex_match(queryIP, mt, ipv6);
	return ret4 == true ? "IPv4" : ret6 == true ? "IPv6" : "Neither";
}


// 剑指 Offer II 114. 外星文字典
// 经典拓扑排序  +  经典dfs
// 好题
class Solution114 {
public:
	unordered_map<char, vector<char>> edges;  // 邻接表存储有向边
	unordered_map<char, int> visited;  // 记录点的访问情况
	bool valid = true;
	int index;
	string ans;
	
	string alienOrder(vector<string>& words) {
		// 初始化结点
		for (auto& word : words) {
			int n = word.size();
			for (int i = 0; i < n; ++i) {
				if (!edges.count(word[i])) {
					edges[word[i]] = vector<char>();
				}
			}
		}
		// 加入边
		for (int i = 1; i < words.size(); ++i) {
			addEdge(words[i - 1], words[i]);
		}

		this->ans = string(edges.size(), ' ');
		this->index = edges.size() - 1;

		for (auto& [node, _] : edges) {  // 遍历结点
			if (!visited.count(node)) {  // 没访问过
				dfs(node);
				if (!valid) {
					return "";
				}
			}
		}
		return ans;
	}

	void addEdge(string pre, string last) {
		int n = min(pre.size(), last.size());
		int index = 0;
		while (index < n) {
			if (pre[index] != last[index]) {
				edges[pre[index]].emplace_back(last[index]);
				break;
			}
			++index;
		}
		if (index == n && pre.size() > last.size()) {
			valid = false;
		}
	}
	
	// dfs 总能找到字典序最大的那个字母
	void dfs(char bg) {
		visited[bg] = 2;  //2 代表正在访问
		for (char temp : edges[bg]) {
			if (!visited.count(temp)) {  // 没访问过
				dfs(temp);
				if (!valid) {
					return;
				}
			}
			else if(visited[temp] == 2) {  // 形成了环
				valid = false;
				return;
			}
		}
		// 标记当前节点访问过了
		// 走到这里，说明后面已经没边了
		visited[bg] = 1;
		ans[index] = bg;  //
		--index;  // 下标减1
	}
};


// 473. 火柴拼正方形
// dfs 
class Solution473 {
public:
	bool makesquare(vector<int>& matchsticks) {
		int sum = accumulate(matchsticks.begin(), matchsticks.end(), 0);
		if (sum % 4) {
			return false;
		}
		int edge = sum / 4;
		vector<int> edges(4, edge);  //  存储还缺多少
		sort(matchsticks.begin(), matchsticks.end(), greater<int>());

		std::function<bool(vector<int>&, vector<int>&, int index, int target)> dfs = [&](vector<int>& matchsticks, vector<int>& edges, int index, int target)->bool {
			if (index == matchsticks.size()) {
				return true;
			}
			for (int i = 0; i < edges.size(); ++i) {
				edges[i] -= matchsticks[index];  // 选当前的
				if (edges[i] >= 0) {  // 可以选
					bool temp = dfs(matchsticks, edges, index + 1, edge);
					if (temp) {
						return true;
					}
				}
				edges[i] += matchsticks[index];  // 不能选，还原
			}
			return false;
		};

		return dfs(matchsticks, edges, 0, edge);
	}
};

// 698. 划分为k个相等的子集
// 经典dfs
class Solution698 {
public:
	bool dfs(vector<int>& nums, vector<int>& sub, int index, int target) {
		if (index == nums.size()) {
			return true;
		}
		// 考虑每个数组的选取情况
		for (int i = 0; i < sub.size(); ++i) {
			sub[i] += nums[index];
			if (sub[i] <= target) { // 说明当前这个数字可以选
				bool temp = dfs(nums, sub, index + 1, target); // 开始遍历后续数字
				// 因为若本次能成功，dfs会一直走到底，这样temp为真说明情况可以
				if (temp) {
					return true;
				}
			}
			// 说明这个数字选不了,则减去
			sub[i] -= nums[index];
			if (sub[i] == 0) {   // 剪枝剪的好  ！！！！！！！！！！
			// 如果连一个数都装不下，那么说明这个数组不满足
				return false;
			}
		}
		return false;
	}

	bool canPartitionKSubsets(vector<int>& nums, int k) {
		// 经典题目
		int sum = accumulate(nums.begin(), nums.end(), 0);
		if (sum % k) {
			return false;
		}
		int each = sum / k;
		int index = 0;
		vector<int> sub(k, 0); // 保存每个数组当前的值
		// 从大开始选
		sort(nums.begin(), nums.end(), greater<int>());
		return dfs(nums, sub, index, each);
	}
};


// 实现堆的建立、插入和删除
class HeapBID {
public:
	// 数组下标从0开始
	void shiftDown(vector<int>& nums, int k) {  // 向下调整
		int leftChild = k * 2 + 1, rightChild = k * 2 + 2;
		int maxIndex = k; //假设在当前节点，及其左、右子节点，共三个节点中，最大的是当前这个节点。后序我们就要更新max，看到底哪个才是最大的，把最大的那个和当前节点交换
		if (leftChild < nums.size() && nums[leftChild] > nums[maxIndex])
			maxIndex = leftChild;
		if (rightChild < nums.size() && nums[rightChild] > nums[maxIndex])
			maxIndex = rightChild;
		if (maxIndex != k)
		{
			swap(nums[maxIndex], nums[k]);
			shiftDown(nums, maxIndex); // 如果原k节点调整了位置(上一步swap调整)，那么就要将k继续做shiftDown操作，直到它比它的左、右孩子都大
		}
		
	}
	//  建立的是大顶堆
	// 向上调整，删除堆顶时候使用
	void shiftUp(vector<int>& nums) {  // 向上调整
		// 首先将最后一个孩子节点插入到头节点
		int child = nums.size() - 1;
		int parent = (child - 1) / 2;
		while (child > 0) {
			if (nums[parent] < nums[child]) {
				swap(nums[parent], nums[child]);
			}
			else {
				break;
			}
			child = parent;
			parent = (child - 1) / 2;  // 换层
		}
	}

	void buildMaxHeap(vector<int>& nums) {
		int n = nums.size();
		for (int i = 0; i < n/2; ++i) {  // 从第一个非叶子节点开始，从下往上，将每棵子树调整成最大堆
			shiftDown(nums, i);
		}
	}

	void inHeap(vector<int>& nums, int x) {
		nums.push_back(x);
		shiftUp(nums);
	}

	void pollHeap(vector<int>& nums) {
		int oldVal = nums[0]; // 头元素
		nums[0] = nums[nums.size() - 1];
		nums.pop_back(); // 弹出末尾
		shiftDown(nums, 0);
	}
};


/*
int main() {
	vector<int> nums{ 1,4,9,7,2,6,8 };
	HeapBID h;
	h.buildMaxHeap(nums);
	cout << "build heap" << endl;
	for (auto& a : nums) {
		cout << a << " ";
	}
	cout << endl;
	cout << "insert" << endl;
	h.inHeap(nums,5);
	for (auto& a : nums) {
		cout << a << " ";
	}

	cout << endl;
	cout << "pop " << nums[0] << endl;
	h.pollHeap(nums);
	for (auto& a : nums) {
		cout << a << " ";
	}

	cout << endl;
	cout << "pop all" << endl;
	int n = nums.size();
	for (int i = 0; i < n; ++i) {
		cout << nums[0] << " ";
		h.pollHeap(nums);
	}
	return 0;
}
*/

// 508. 出现次数最多的子树元素和
class Solution508 {
public:
	unordered_map<int, int> mp; // 保存值出现的次数
	int ans = 0;
	vector<int> findFrequentTreeSum(TreeNode* root) {
		dfs(root);
		vector<int> rt;
		for (auto [key, value] : mp) {
			if (value == ans) {  // 找到出现次数为max的值
				rt.emplace_back(key);
			}
		}
		return rt;
	}

	int dfs(TreeNode* root) {  // 求一个节点和其子树和
		if (root == nullptr) {
			return 0;
		}
		int sum = root->val;
		sum = sum + dfs(root->left) + dfs(root->right);
		mp[sum]++;  // 记录一下当前值
		ans = max(ans, mp[sum]); // 记录目前出现最多次数的值
		return sum;
	}
};


// 929. 独特的电子邮件地址
// 正则表达式
// ?=   零宽断言
//  就是一种条件
int numUniqueEmails(vector<string>& emails) {
	// 考虑没有+号的情况
	// (\.)(?=.*@)     去掉@之前的所有.
	// 考虑有+号的情况
	// (\+.*)(?=@)  +号之后@之前的所有字符作废
	// 两种情况求并集
	regex pattern("(\\.)(?=.*@)|(\\+.*)(?=@)");
	unordered_set<string> st;
	for (auto& email : emails) {
		st.insert(regex_replace(email, pattern, ""));
	}
	return st.size();
}


// sscanf 实现字符串的替换
// 1108. IP 地址无效化
// #include <stdio.h>
// int sscanf(const char* str, const char* format, …);
string defangIPaddr(string address) {
	int a, b, c, d;
	sscanf(address.c_str(), "%d.%d.%d.%d", &a, &b, &c, &d);
	return to_string(a) + "[.]" + to_string(b) + "[.]" + to_string(c) + "[.]" + to_string(d);
}



// 线段树解决区间覆盖和删除问题
// 715. Range 模块
struct SegTree715 {
	SegTree715 *left, *right;
	bool covered;
	int l, r; // 左右边界
	SegTree715(int left, int right) :l(left), r(right), covered(false), left(nullptr), right(nullptr) {};
	SegTree715(int left, int right, bool covered) :l(left), r(right), covered(covered), left(nullptr), right(nullptr) {};
};

// 半开区间 [left, right)
class RangeModule {
public:
	SegTree715 *sg;
	RangeModule() { // 初始化数据结构的对象
		// 初始化线段树区间为数据范围
		sg = new SegTree715(0, (int)1e9);
	}

	void addRange(int left, int right) {  // 添加 半开区间 
		// 添加时若只是部分间断覆盖，那么需要递归进行部分区间更新
		addNum(sg, left, right);
	}

	void addNum(SegTree715 *sg, int left, int right) {
		// 在此线段树中添加新的区间[left,right)
		if (sg->covered) {  // 当前线段树的区间已经全部被覆盖，那么不需要再更新
			return;
		}
		int l = sg->l, r = sg->r;
		int mid = l + (r - l) >> 1;  // 二分查找区间
		if (left <= l && right >= r) {  // 添加的区间可以覆盖当前最大区间，那么可以删除当前区间的所有子树
			// 因为已经设置了线段树的最大区间，因此不会超过预设区间
			sg->left = nullptr;
			sg->right = nullptr;
			sg->covered = true;
		}
		else if(right<=mid) {  // 去更新左侧区间
			addNum(sg->left, left, right);
		}
		else if (left >= mid) { //去更新右侧区间
			addNum(sg->right, left, right);
		}
		else {  // l <= left <= mid <= right <=r
			// 两侧都需要更新
			if (sg->left == nullptr) {
				sg->left = new SegTree715(left, mid, true);
			}
			if (sg->right == nullptr) {
				sg->right = new SegTree715(mid, right, true);
			}
			addNum(sg->left, left, mid);
			addNum(sg->right, mid, right);
		}
	}

	bool queryRange(int left, int right) {  // [left,right)全被覆盖的话，返回true，否则false
		return judgeCover(sg, left, right);
	}

	bool judgeCover(SegTree715* sg, int left, int right) {
		// 判断  [left, right) 范围的所有数字是否都被覆盖
		if (sg == nullptr) {
			return false; //  边界条件
		}
		int l = sg->l, r = sg->r;
		int mid = l + (r - l) >> 1;
		// 边界条件
		if (sg->covered) {
			return true;
		}
		if (right <= mid) {  // 在左侧
			return judgeCover(sg->left, left, right);
		}
		else if (mid <= left) {  // 右侧
			return judgeCover(sg->right, left, right);
		}
		else {  // 两侧
			return judgeCover(sg->left, left, mid) && judgeCover(sg->right, mid, right);
		}
	}

	void removeRange(int left, int right) {  // 删除当前区间
		removeNum(sg, left, right);
	}

	void removeNum(SegTree715* sg, int left, int right) {
		// 注意是删除区间内的所有数字，那么子区间也得删除
		if (sg == nullptr) {
			// 没有，那么数字也不存在
			return;
		}
		int l = sg->l, r = sg->r;
		int mid = l + (r - l) >> 1;
		if (left <= l && right >= r) {  // 完全覆盖此区间
			sg->left = nullptr;
			sg->right = nullptr;
		}
		else if (sg->covered) { // 区间所有点都有，直接正常删除
			if (right <= mid) {  // 在seg的左部分
				removeNum(sg->left, left, mid);
			}
			else if (mid <= left) {  //  右部分
				removeNum(sg->right, mid, right);
			}
			else {
				// 左右部分都需要删除
				removeNum(sg->left, left, mid);
				removeNum(sg->right, mid, right);
			}
		}
		else {  // 线段树内的数字并不是全都存在
			if (right <= mid) {
				removeNum(sg->left, left,right);
			}
			else if (mid <= left) {
				removeNum(sg->right, left, right);
			}
			else {
				removeNum(sg->left, left, mid);
				removeNum(sg->right, mid, right);
			}
		}
		sg->covered = false;
	}
};


// 241. 为运算表达式设计优先级
// 分治法
// 考虑每一个运算符的左右部分，添加括号。如2*3-4*5
// 添加括号可得    (2) * (3-4*5)  ->   (2)*((3)-(4*5))   ->  (2)*((3)-((4)*(5)))   
// 其余情况类似，这样一个 完整的式子就可以使用子式子计算求和
// 为了防止重复，可以加入hash去重
/*
class Solution241 {
public:
	vector<int> diffWaysToCompute(string expression) {
		// ****分*****
		int n = expression.size();
		for (int i = 0; i < n;++i) {
			if (a == '+' || a == '-' || a == '*') {
				string left = expression.substr()
			}
		}
	}
};*/

//6109. 知道秘密的人数
// O(N^2)硬推
class Solution6109 {
public:
	int MOD = 1e9 + 7;
	int peopleAwareOfSecret(int n, int delay, int forget) {
		vector<long long> vec(n + 1);
		vec[1] = 1;
		for (int i = 2; i <= n; ++i) {
			long long temp = 0;
			for (int j = max(i - forget + 1, 0); j <= max(i - delay, 0); ++j) {
				temp = (temp + vec[j] + MOD) % MOD;
			}
			vec[i] = temp;
		}
		long long ans = 0;
		for (int i = max(n - forget + 1, 0); i <= n; ++i) {
			ans = (ans + vec[i] + MOD) % MOD;
		}
		return ans;
	}

	// 6109 前缀和压缩到O(N)
	// 前缀和的下标很恶心
	int peopleAwareOfSecret_case2(int n, int delay, int forget) {
		// 使用前缀和
		vector<long long> presum(n + 1, 0);
		// 第一个人
		presum[1] = 1;
		for (int i = 2; i <= n; ++i) {
			int l = max(0, i - forget); //最先忘记的人
			int r = max(0, i - delay); //最后可以告诉他人的人
			presum[i] = (presum[i - 1] + presum[r] - presum[l] + MOD) % MOD;
		}
		long long ans = (presum[n] - presum[n - forget] + MOD) % MOD;
		return ans;
	}
};



class Solution6110 {
public:
	int MOD = 1e9 + 7;
	vector<vector<int>> step{ {0,1},{0,-1},{1,0},{-1,0} };
	vector<vector<long long>> mem;
	int m, n;
	long long dfs(vector<vector<int>>& grid, int x, int y) {
		if (mem[x][y] > 1) {
			return mem[x][y];
		}
		for (int i = 0; i < 4; ++i) {
			int new_x = x + step[i][0];
			int new_y = y + step[i][1];
			if (new_x >= 0 && new_x < m && new_y >= 0 && new_y<n && grid[new_x][new_y]>grid[x][y]) {  // 范围内
				mem[x][y] = (mem[x][y] + dfs(grid, new_x, new_y) + MOD) % MOD;
			}
		}
		return mem[x][y];
	}

	int countPaths(vector<vector<int>>& grid) {
		// 记忆化搜索
		// 用mem[i][j] 表示到这个点的路径个数
		m = grid.size();
		n = grid[0].size();
		long long ans = 0;
		mem.resize(m, vector<long long>(n, 1));
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				ans = (ans + dfs(grid, i, j) + MOD) % MOD;
			}
		}
		return (int)ans;
	}
};


// 496. 下一个更大元素 I
class Solution496496 {
public:
	vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
		// 先把nums2中每个数右侧下一个更大元素找出来
		int n = nums2.size();
		stack<int> st;
		unordered_map<int, int> mp;
		for (int i = n - 1; i >= 0; --i) {
			while (!st.empty() && st.top() < nums2[i]) {
				st.pop();
			}
			mp[nums2[i]] = st.empty() ? -1 : st.top();
			st.push(nums2[i]);
		}
		vector<int> ans((int)nums1.size(), -1);
		for (int i = 0; i < (int)nums1.size(); ++i) {
			ans[i] = mp[nums1[i]];
		}
		return ans;
	}
};

// 503. 下一个更大元素 II
class Solution503503 {
public:
	vector<int> nextGreaterElements(vector<int>& nums) {
		// 和 I 一样的解法，不同的是按照数组长度翻倍处理
		stack<int> st;
		int n = nums.size();
		vector<int> ans(n, -1);
		for (int i = 2 * n - 1; i >= 0; --i) {
			while (!st.empty() && nums[i % n] >= st.top()) {
				st.pop();
			}
			ans[i % n] = st.empty() ? -1 : st.top();
			st.push(nums[i % n]);
		}
		return ans;
	}
};

// 556. 下一个更大元素 III
class Solution556 {
public:
	int nextGreaterElement(int n) {
		string str = to_string(n);
		// 从右往左找到第一个不满足从左到右为降序的数字
		// 然后与它右侧大于它的最小数字交换就行
		int len = str.size();
		int fst_min = -1;
		for (int i = len; i > 0; --i) {
			if (str[i] > str[i - 1]) {
				fst_min = i - 1;
				break;
			}
		}
		if (fst_min < 0) {
			return -1;
		}
		// 记录下来位置，然后找大于它的最小数字
		int rt_min = fst_min + 1;
		for (int j = fst_min + 1; j < len; ++j) {
			if (str[j] > str[fst_min]) {
				rt_min = str[rt_min] < str[j] ? rt_min : j;
			}
		}
		swap(str[rt_min], str[fst_min]);

		// 把fst_min右侧其余数字升序排列
		sort(str.begin() + fst_min + 1, str.end());
		return stoll(str) > INT_MAX ? -1 : stoll(str);
	}
};


// 729. 我的日程安排表 I
class MyCalendar729 {
public:
	// 区间不能有交叉。set倒着插入，找第一个大于start的right
	set<pair<int, int>> st;
	MyCalendar729() {
	}

	bool book(int start, int end) {
		auto fnd = st.upper_bound({ start + 1,-1 });
		if (fnd != st.end()) {
			// 判断是否有重叠
			if (end <= fnd->second) {
				st.insert({ end,start });
				return true;
			}
			else {
				return false;
			}
		}
		else {  // 没有点的右区间大于start，说明start最大
			st.insert({ end,start });
			return true;
		}
	}
};

//735. 行星碰撞
vector<int> asteroidCollision(vector<int>& asteroids) {
	// stack
	stack<int> st;
	for (size_t i = 0; i < asteroids.size(); ++i) {
		if (i == 0) {
			st.push(asteroids[i]);
			continue;
		}
		int tag = asteroids[i] < 0 ? 0 : 1;
		int tp_tag;
		if (!st.empty()) {
			tp_tag = st.top() < 0 ? 0 : 1;
		}
		int equ = 0;
		while (!st.empty() && (tag == 0 && tp_tag == 1) && i < asteroids.size()) {  //  左正右负才炸
			if (abs(asteroids[i]) > st.top()) {
				st.pop();
			}
			else if (abs(asteroids[i]) == st.top()) {
				st.pop();
				equ = 1;
				break;
			}
			else if (abs(asteroids[i]) < st.top()) {
				equ = 1;
				break;
			}
			if (!st.empty()) {  // 记录一下看看后续能否继续炸
				tp_tag = st.top() < 0 ? 0 : 1;
			}
		}
		// 不炸
		if (!equ) {
			st.push(asteroids[i]);
		}
	}
	vector<int>ans;
	while (!st.empty()) {
		ans.emplace_back(st.top());
		st.pop();
	}
	reverse(ans.begin(),ans.end());
	return ans;
}




// 二维数组的花式遍历
// 48.旋转图像
// 54.矩阵螺旋
// 59.螺旋矩阵


// 749. 隔离病毒
/*
	计算最大连通的病毒区域，用不同的常量标记各个区域，使用一个state递减实现
	已经围过
*/
class Solution749 {
private:
	const vector<vector<int>> steps{ {-1,0},{1,0},{0,-1},{0,1} };
	int state = -2;
	int wall_count;
	vector<vector<int>> flags;
	int m, n;
public:
	int containVirus(vector<vector<int>>& isInfected) {
		m = isInfected.size();
		n = isInfected[0].size();
		flags.resize(m, vector<int>(n, 0));
	}

	int find_max_union(vector<vector<int>>& isInfected) {
		int ans = 0;
		int curMaxArea = 0;
		int target_x = 0, target_y = 0;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (isInfected[i][j] == 1 && flags[i][j] == 0) {
					wall_count = 0;
					int temp_area = dfs(isInfected, state, i, j);  // 使用 state 标记区域，以便于统计edge
					// 为了复用state，--state保证区域的划分
					if (temp_area > curMaxArea) {
					
					}
				}
			}
		}
	}

	int dfs(vector<vector<int>>& isInfected, int state, int x, int y) {
		flags[x][y] = 1;
		int edge_count = 0;
		for (int i = 0; i < 4; ++i) {
			int new_x = x + steps[i][0];
			int new_y = y + steps[i][1];
			// 负的state表示不同的模块，  0 表示未访问过，1表示访问过
			if (new_x >= 0 && new_x < m && new_y >= 0 && new_y < n && flags[new_x][new_y]!=1) {
				// 不是病毒
				if (isInfected[new_x][new_y] != 1) {
					// 说明走到了病毒区域的边界，边缘可以统计
					++wall_count;
					// 墙的个数和edge的个数不一样，如果在一个左下角空缺的拐角，那么是有一个外围edge模块，但是需要使用两块墙
					if (flags[new_x][new_y] != state) {
						flags[new_x][new_y] = state;
						edge_count++;
					}
				}
				else {  //  是病毒，继续递归
					edge_count += dfs(isInfected, state, new_x, new_y);
				}

			}
		}
	}
};



// 452. 用最少数量的箭引爆气球
// 一枝箭射气球问题，此时区间交集为1，气球就可以破
class Solution452 {
public:
	int findMinArrowShots(vector<vector<int>>& points) {
		// 求尽可能多的区间交集
		if (points.size() == 0) {
			return 0;
		}
		sort(points.begin(), points.end(), [](vector<int>& a, vector<int>& b)->bool {
			return a[1] < b[1];
			});

		int ans = 1;
		int end = points[0][1];
		for (int i = 1; i < points.size(); ++i) {
			if (points[i][0] > end) {
				end = points[i][1];
				++ans;
			}
		}
		return ans;
	}
};




// 757. 设置交集大小至少为2
// 和射气球问题的区别是，这里的区间交集至少为2，而气球为 1 即可
// 左右融合区间时，之前的区间如何考虑
// s不一定连续，小区间满足大区间满足
// 从大到小融合
class Solution757 {
public:
	int intersectionSizeTwo(vector<vector<int>>& intervals) {
		// 集合s不连续
		sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b)->bool {
			return a[0] == b[0] ? a[1] > b[1] : a[0] < b[0];
			});
		int ans = 2;
		int n = intervals.size();
		int cur = intervals[n - 1][0], next = intervals[n - 1][0] + 1;
		for (int i = n - 2; i >= 0; --i) {
			// 下一个区间，左侧肯定<=当前区间左侧，按照排序来说，若右侧大于当前，肯定出现大区间包含小区间的情况，因此不用管
			if (intervals[i][1] >= next) {
				continue;
			}
			else if (intervals[i][1] < cur) {  // 没重叠
				ans += 2;
				cur = intervals[i][0];
				next = intervals[i][0] + 1;
			}
			else if (intervals[i][1] >= cur && intervals[i][1] < next) {   // 部分重叠，如何考虑重叠个数？
				// 默认重叠一个，ans计数只加1, cur和next怎么动
				// 取个重叠点，取个新区间的左端点
				// s不连续！！！
				next = cur;
				cur = intervals[i][0];
				++ans;   // 新区间左端点记个数
			}
		}
		return ans;
	}
};









int main() {
	vector<int> vec{ 1,-1,-2,-2 };
	
	return 0;
}




	//Solution587 sol;
	//vector<pair<float, float>> vec{ {0,2},{0,1},{0,0},{1,0},{2,0},{1,1} };
	//vector<vector<int>> vec{ {1, 1}, { 2, 2 }, { 2, 0 }, { 2, 4 }, { 3, 3 }, { 4, 2 }};
	//sol.outerTrees(vec);
	//convexHull(vec);


	//string input = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext";
	//lengthLongestPath(input);

	//string str = "I speak Goat Latin";
	//toGoatLatin(str);

	//vector<int> v{ 5,7,2,4,6,8,9,23,6,8,9,54,4, 5,7,2,4,6,8,9,23,6,8,9,54,4 };
	//reservoir(v, 8);

	//Solution1036 sol;
	//vector<vector<int>> blocked{ {10,9},{9,10},{10,11},{11,10} };
	//vector<int> source{ 0, 0 };
	//vector<int> target{ 10, 10 };
	//sol.isEscapePossible(blocked, source, target);

	////Solution306 sol;
	//string s = "101";
	//sol.isAdditiveNumber(s);

	//vector<int> vec{ 0,0,1,3,4,2,6,8 };
	//SolutionArrayResort sa;
	//sa.findOriginalArray(vec);
	// 
	// 
	//string s = "/a/../../b/../c//.//";
	//Solution71 sol;
	//sol.simplifyPath_case2(s);


	//vector<string> vec = { "cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat" };
	//Solution472 sol;
	//sol.findAllConcatenatedWordsInADict(vec);

	//string s = "abcabcccdd";
	//string t = "ccc";
	//RabinKarp rb;
	//rb.rabinkarp(s, t);
	// 
	// 
	//Solution1034 sol;
	//vector<vector<int>> vecs{ {1,2,2},{2,3,2} };
	//sol.colorBorder(vecs, 0, 1, 3);
	//clock_t start, end;
	//start = clock();

	//vector<int> weight = { 3,4,5,3,6 };
	//vector<int> value = { 4,5,6,3,5 };

	//bagProblemcomp(weight, value, 5, 10);


	//numSquares(12);


	//vector<int> piles = { 5,3,4,5 };
	//Stone sto;
	//sto.stoneGame(piles);

	//vector<vector<int>> s = { {0,0},{0,0},{0,0} };
	//vector<vector<int>> t = { {1,1},{1,1},{1,1} };
	//maxCompatibilitySum(s,t);


	//vector<vector<int>> time{ {2,1,1},{2,3,1},{3,4,1} };
	//networkDelayTime(time, 4, 2);

	//smallest_K sk;
	//vector<int> vec{ 1,3,5,7,2,4,6,8 };
	//sk.smallestK(vec, 4);

	//vector<int> nums{ 4,-4,1,-3,1,-3 };
	//choose_num(nums);


	//vector<vector<char>> boards{ {'1'},{'2'} };
	//isValidSudoku(boards);

	
	//cnt(111);
	//cnt(30);
	//cnt(1);
	
	//windy wd;
	//wd.digdp(1, 10);
	//wd.digdp(25, 50);

	//string s = "()())()";
	//Solution301 s301;
	//s301.removeInvalidParentheses(s);


	//divide(10, 3);

	//vector<int> vec{ 0,1,0,2,1,0,1,3,2,1,2,1 };
	//trap(vec);

	//Solution282 sol;
	//sol.addOperators("2147483648", -2147483648);


	//countSort(vec);

	///auto a = std::make_shared<A>();   //此时a是一个指针
	///auto funcds = std::bind(&A::add, std::placeholders::_1, 3, 4);
	///std::cout << funcds(a) << std::endl;   // cout  7

	// vector<string> aa = { "abcw","baz","foo","bar","xtfn","abcdef" };
	// maxProduct(aa);

	// fractionToDecimal(7, 12);

	//vector<vector<int>> matrix = { {1, 4, 7, 11, 15},{2, 5, 8, 12, 19},{3, 6, 9, 16, 22},{10, 13, 14, 17, 24},{18, 21, 23, 26, 30} };
	//searchMatrix(matrix, 5);

	//TransNumtoWord tw;
	//tw.numberToWords(12345);

	//toHex(26);	

	//trie_example1 sa;
	//vector<vector<char>> board = { {'o', 'a' , 'a', 'n'},{'e', 't', 'a', 'e'},{'i', 'h', 'k', 'r'},{'i', 'f', 'l', 'v'} };
	//vector<string> words = { "oath", "pea", "eat", "rain" };
	//sa.findWords(board, words);

	//vector<int> a{ 1,2,3 }, b{ 0,1,2 };
	//findMaximizedCapital(3, 0, a, b);

	//vector<int> primes{ 2,7,13,19 };
	//nthSuperUglyNumber(12, primes);
	//nthSuperUglyNumber_multi_pointer(12, primes);


//	pathInZigZagTree(14);

	//vector<vector<int>> vec{ {2,1},{3,4},{3,2} };
	//restoreArray(vec);



	//vector<vector<int>> vec{ {1,2,3},{1,5,1},{3,1,1} };
	//maxPointss(vec);

	//vector<int> vec = { 1,2,4 };
	//maxFrequency(vec, 5);

	//vector<vector<int>> routes = { {0, 1, 6, 16, 22, 23},{4, 10, 12, 20, 24, 28, 33},{14, 15, 24, 32},{1, 10, 11, 19, 27, 33},{15, 20, 21, 23, 29},{29} };
	//numBusesToDestination(routes, 4, 21);

	//AAA a;
	//a.countPalindromicSubsequence("nhzosdwmwomlevcctvopoiiayudhvauitqutiboveumsqvbulhbfbynzogtejuwi");

	//string formula = "K4(ON(SO3)2)2";
	//countOfAtoms(formula);


	//vector<vector<string>> orders = { {"David", "3", "Ceviche"},{"Corina", "10", "Beef Burrito"},{"David", "3", "Fried Chicken"},{"Carla", "5", "Water"},{"Carla", "5", "Ceviche"},{"Rous", "3", "Ceviche"} };
	//displayTable(orders);

	//int n = 5;
	//vector<vector<int>> relation = { {0, 2},{2, 1},{3, 4},{2, 3},{1, 4},{2, 0},{0, 4} };
	//int k = 3;
	//numWays(n, relation, k);


	/*vector<vector<int>> vec = {
		{-1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1 },
		{-1, -1, -1, -1, -1, -1},
		{ -1, 35, -1, -1, 13, -1 },
		{-1, -1, -1, -1, -1, -1},
		{ -1, 15, -1, -1, -1, -1 }};
	snakesAndLadders(vec);*/

	//slidingPuzzle(vec);

	//string a = "aabb";
	//wonderfulSubstrings(a);

	//vector<string> lock = { "8888" };
	//string s = "0009";
	//openLock(lock, s);

	//ssss s;
	//vector<vector<int>> g1 = { {1, 0, 1, 0, 1},{1, 1, 1, 1, 1},{0, 0, 0, 0, 0},{1, 1, 1, 1, 1},{1, 0, 1, 0, 1} };
	//vector<vector<int>> g2 = { {0, 0, 0, 0, 0} , {1, 1, 1, 1, 1},{0, 1, 0, 1, 0},{0, 1, 0, 1, 0},{1, 0, 0, 0, 1} };
	///s.countSubIslands(g1, g2);


	//vector<int> nums = { 1,3,4,8 };
	//vector<vector<int>> queries = { {0,1},{1,2},{2,3},{0,3} };
	//minDifference(nums, queries);

	//vector<int>nums = { 1,5,1 };
	//nextPermutation(nums);

	//Perms pe;
	//string s = "abc";
	//pe.permutation(s);

	//BinaryWatch bw;
	//bw.readBinaryWatch(7);

	////vector<string> arr = { "abcdefghijklm", "bcdefghijklmn", "cdefghijklmno", "defghijklmnop", "efghijklmnopq", "fghijklmnopqr", "ghijklmnopqrs", "hijklmnopqrst", "ijklmnopqrstu", "jklmnopqrstuv", "klmnopqrstuvw", "lmnopqrstuvwx", "mnopqrstuvwxy", "nopqrstuvwxyz", "opqrstuvwxyza", "pqrstuvwxyzab" };
	//StrLength sl;
	//sl.maxLength_1(arr);


	//string s = "1000000000000000000";
	//smallestGoodBase(s);

	//string s = "3.";
	//Legal_Number ln;
	//ln.isNumber(s);


	//vector<vector<int>> matrix = { {8,1, 6},{3,5,7},{4,9,2},{7,10,9} };
	//largestMagicSquare(matrix);


	//vector<vector<int>> node = { {0, 1},{0, 2},{2, 3},{3, 4} };
	//largestPathValue("abaca", node);

	//vector<int> cost = { 2,4,6,2,4,6,4,4,4 };
	//int target = 5;
	//largestNumber(cost, target);


	//int amount = 5;
	//vector<int> coins = { 1,2,5 };
	//change(amount, coins);

	//int n = 10, minProfit = 5;
	//vector<int> group = { 2, 3, 5 }, profit = { 6, 7, 8 };
	//profitableSchemes(n, minProfit, group, profit);

	//minFlips("11100");

	//vector<string> strs = { "10", "0001", "111001", "1", "0" };
	//findMaxForm(strs, 5, 3);

	//vector<int> nums = { 5,1,3 };
	//reductionOperations(nums);

	//vector<int> stones = { 1,1,4,2,2 };
	//lastStoneWeightII(stones);

	//targetsum tg;
	//vector<int> nums{ 1,1,1,1,1 };
	//tg.findTargetSumWays_dp(nums, 3);

	//vector<int> nums = { 23,2,4,6,6 };
	//int k = 7;
	//checkSubarraySum(nums, k);



	//vector<int> nums = { 0,1,0,1,0,1 };
	//findMaxLength(nums);

	//vector<int> servers = { 74,57,61,82,67,97,67,21,61,79,21,50,14,88,48,52,76,64 }, tasks = { 21,100,48,64,20,8,28,10,3,63,7 };
	//TasksProcessing tp;
	//tp.assignTasks(servers,tasks);


	//vector<int> nums = { 4,14,2 };
	//totalHammingDistance(nums);


	//vector<int> nums = { 1,2,3 };
	//Combination_Sum cb;
	//cb.combinationSum4_2(nums, 4);


	//vector<int> nums = { 1,2,3,4 };
	//productExceptSelf(nums);

	//string s = "010010"; 
	//int minJump = 2, maxJump = 3;
	//canReach(s, minJump, maxJump);

	//string s = "(ed(et(oc))el)";
	//reverseParentheses(s);


	//string s = "aaabbb";
	//strangePrinter(s);

	//string s = "111000";
	//checkZeroOnes(s);

	//vector<int> a = { 1,1,100000 };
	//double hours = 2.01;
	//minSpeedOnTime(a, hours);

	//vector<int> nums1 = { 1,4,2 };
	//vector<int> nums2 = { 1,2,4 };
	//maxUncrossedLines(nums1, nums2);

/*
	auto cmp = [](const auto& a, const auto& b) {
		return a > b;
	};
	priority_queue< int, vector<int>, decltype(cmp)> que(cmp);  
	//优先队列的排序函数是反着来的，定义的return a>b,但是却是小根堆
	
	vector<int> nums = { 1,6,8,9,3,54,6,8,9,3,2 };
	for (auto a : nums) {
		que.push(a);
	}*/


	
	//vector<string> words = { "i", "love", "leetcode", "i", "love", "coding" };
	//int k = 2;
	//topKFrequent(words,k);


	//vector<int> nums = { 3,2,1,5,6,4 };
	//quickSelect qs;
	//qs.findKthLargest(nums,4);

	//vector<int> nums = { 2,3,1,6,7 };
	//countTriplets(nums);


	//vector<int> nums = { 3, 10, 5, 25, 2, 8 };
	//findMaximumXOR(nums);

	//rearrangeSticks(5, 4);


	//string s = "010";
	//minSwaps(s);

	//vector<int> nums = { 5,1,6 };
	//subsetXORSum(nums);


	//vector<vector<string>>box = { {"#", "#", "*", ".", "*", "."},
		//{"#", "#", "#", "*", ".", "." },
		//{"#", "#", "#", ".", "#", "."} };

	//rotateTheBox(box);

	//vector<int> g = { 8,5,2,9,1 };
	//vector<int> s = { 0,0,0,0,0 };
	//findContentChildren(g, s);


	//vector<int> nums = { 1,1,3,6,8 };
	//minOperations(nums);

	//string s = "-43";
	//myAtoi(s);


	//string s1 = "great", s2 = "rgeat";
	//scramble sc;
	//sc.isScramble2(s1, s2);

	//AAA A;
	//A.maxSumMinProduct(n1);

	//vector<int> nums = { 2,3,3,1,2 };
	//maxSumMinProduct(nums);

	//vector<int> encoded = { 6,5,4,6 };
	//decode(encoded);


	//vector<int> bloomDays = { 1,10,3,10,2 };
	//Flowers fl;
	//fl.minDays(bloomDays,3,1);

	//Frog fr;
	//vector<int> stones = { 0,1,3,5,6,8,12,17 };
	//fr.canCross(stones);

	//vector<int> n1 = { 55,30,5,4,2 };
	//vector<int> n2 = { 100,20,10,10,5 };
	//maxDistance(n1, n2);

	//vector<vector<int>> logs = { {1982, 1998}, {2013, 2042},{2010, 2035},{2022, 2050},{2047, 2048} };
	//maximumPopulation(logs);


	//vector<int> jobs = { 1,2,4,7,8 };
	//Jobs jbs;
	//jbs.minimumTimeRequired(jobs, 2);


	//vector<vector<int>> walls = { {1, 2, 2, 1},{3, 1, 2},{1, 3, 2},{2, 4},{3, 1, 2},{1, 3, 1, 1} };
	//leastBricks(walls);


	//vector<int> nums = { 3,4,2 };
	//deleteAndEarn(nums);


	//judgeSquareSum(8);

	//vector<int> nums = { 9,18,54,90,108,180,360,540,720 };
	//largestDivisibleSubset(nums);


	//vector<vector<int>> matrix = { {1, 0, 1},{0, -2, 3} };
	//int k = 2;
	//maxSumSubmatrix(matrix, k);


	//vector<vector<int>> tasks = { {1,2},{2,4},{3,2},{4,1} };
	//getOrder(tasks);


	//nthUglyNumber_case2(10);

	//vector<int> nums = { 2,3,2,6,2,5,9,0,1,3 };
	//rob(nums);



	//vector<int> nums = { 0,2,2,1,0,3,0,3,0,1,3,1,1,0,1,3,1,1,1,0,2,0,0,3,3,0,3,2,2,0,0,3,3,3,0,0,2,0,0,3,3,0,3,3,0,0,3,1,0,1,0,2,3,1,1,0,3,3,0,3,1,3,0,2,2,0,1,3,0,1,0,3,0,1,3,1,2,2,0,0,3,0,1,3,2,3,2,1,0,3,2,2,0,3,3,0,3,0,0,1,0 };
	//minSideJumps(nums);


	//findTheWinner(5, 2);


	//vector<int> nums = {5,1,3};
	//search(nums, 3);

	//vector<int> m1 = { 0,0,1,1,1 };
	//numRabbits(m1);

	//Perm pm;
	//vector<int> nums = { 1,2,3 };
	//pm.subsets(nums);


	//vector<int> nums = { 3, 10, 5, 25, 2, 8 };
	//findMaximumXOR(nums);


	//Solution22 sol22;
	/*vector<vector<char>> board =
		{
			{'S', 'F', 'C', 'S'},
			{'A', 'D', 'E', 'E'},
			{'S', 'F', 'C', 'S'},
			{'A', 'D', 'E', 'E'},
			{'A', 'B', 'C', 'E'},
			{'S', 'F', 'C', 'S'},
			{'A', 'D', 'E', 'E'}
		};*/
//sol22.exist(board,"ABCBEFG");

	//vector<int> nums = { 1,0,1,-4,-3 };
	//find132pattern(nums);



	//maxValue(2,1,21);

	//vector<vector<int>> orders = { {7, 1000000000, 1},{15, 3, 0},{5, 999999995, 0},{5, 1, 1} };
	//getNumberOfBacklogOrders(orders);

	//vector<string> s = { "10","6","9","3","+","-11","*","/","*","17","+","5","+" };
	//evalRPN(s);


	//int n = 12222333;
	//nextGreaterElement(n);

	//string text1 = "abcdef", text2 = "def";
	//longestCommonSubsequence(text1, text2);

	//string s = "rabbbit";
	//string t = "rabbit";
	//dpAndMemory DM;
	//DM.numDistinct(s, t);

	//vector<int> a = { 1,2,1,7,8,9 };
	//nextGreaterElements(a);

	//vector<vector<int>> env = { {4,5},{4,6},{6,7},{2,3},{1,1} };
	//maxEnvelopes(env);

	//vector<int> nums = { 7,7,7,7,7,7,7,7,7,7 };
	//lengthOfLIS(nums);

	//string s = "9,#,92,#,#";
	//isValidSerialization(s);

	//string S = "abbacd";
	//removeDuplicates(S);

	//string s = "3+2*2";
	//calculate2(s);

	//string s = " 2-1 + 2 ";
	//calculate(s);

	//string s = "aabcb";
	//beautySum(s);

	//string s = "aab";
	//minCut(s);

	//string s = "cdbcbbaaabab";
	//int x = 4, y = 5;
	//maximumGain(s, x, y);
	
	//int n = 8, m = 2;
	//vector<int> group = { -1, -1, 1, 0, 0, 1, 0, -1 };
	//vector<vector<int>> beforeItems = { {},{6},{5},{6},{3, 6},{},{},{} };
	//sortItems(n, m, group, beforeItems);

	//vector<vector<int>> edges = { {1,2}, {2,3}, {2,3},{3,4},{1,4},{1,5} };
	//findRedundantConnection(edges);


	//int n = 5;
	//vector<vector<int>> edges = { {0, 1, 1},{1, 2, 1},{2, 3, 2},{0, 3, 2},{0, 4, 3},{3, 4, 3},{1, 4, 6} };
	//findCriticalAndPseudoCriticalEdges(n, edges);

	//string s = "AABABBABABABABA";
	//int k = 1;
	//characterReplacement(s, k);
	//vector<int> A = { 0,1,0 };
	//int K = 1;
	//minKBitFlips(A, K);
	//vector<int> nums = { 1,3,-1,-3,5,3,6,7 };
	//int k = 3;
	//medianSlidingWindow(nums, k);
	

	//string s = "anryddgaqpjdw";
	//string t = "zjhotgdlmadcf";
	//int cost = 5;
	//equalSubstring(s, t, cost);


//int n = 5;
//vector<vector<int>> connections = { {0, 1},{0, 2},{3,4},{2,3} };
//makeConnected(n, connections);





//vector<int> source = { 1,2,3,4 };
	//vector<int> target = { 1,3,2,4 };
	//vector<vector<int>> allowedSwaps = { };
	//minimumHammingDistance(source, target, allowedSwaps);

//vector<vector<int>> isConnected = { {1, 1, 0},{1, 1, 0},{0, 0, 1} };
	//findCircleNum(isConnected);

/*
vector<int> a = { 0,0,0,0,0,0,0,0 };
	waysToSplit(a);
*/

/*
leetcode 399
图论
vector<vector<string>> equations = { {"a", "b"},{"b", "c"} };
	vector<double> values = {2.0, 3.0};
	vector<vector<string>> queries = { {"a", "c"},{"b", "a"},{"a", "e"},{"a", "a"},{"x", "x"} };

	calcEquation(equations, values, queries);

*/



/*
vector<int> a = { 2,14,11,5,1744,2352,0,1,1300,2796,0,4,376,1672,73,55,2006,42,10,6,0,2,2,0,0,1,0,1,0,2,271,241,1,63,1117,931,3,5,378,646,2,0,2,0,15,1 };
	countPairs(a);
*/

//vector<int> a = { 9,11 };
	//int k = 2;
	//maxSlidingWindow(a, k);


	//vector<int> a = { 0,0,1,0,1 };int n = 1;
	//canPlaceFlowers(a, n);

	//vector<int> ratings = { 1,2,5,4,3,2,1 };
	//candy(ratings);
	
	
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

	//vector<int> vec;
	//string s = "123456579";
	//splitIntoFibonacci(s);

	//vector<int> num = { 5,2,9,1,6 };
	//containsDuplicate(num);

	
	//vector<string> a = { "eat", "tea", "tan", "ate", "nat", "bat" };
	//groupAnagrams(a);

	//vector<int> nums = { 7,90,5,1,100,10,10,2  };
	//PredictTheWinner(nums);
	
	//vector<int> stones = { 5,3,1,4,2 };
	//stoneGameVII(stones);
	
	
	//monotoneIncreasingDigits(120);
	//string  pattern = "abba", str = "dog cat cat dog";
	//wordPattern(pattern, str);
	
	//string s = "abcd", t = "abcde";
	//findTheDifference(s, t);

	//vector<vector<int>> matrix = {
	//							 {1,2,3},
	//							 {4,5,6},
	//							 {7,8,9} };
	//rotate(matrix);
	
	//vector<int> a = { 5,2,1,2,5,2,1,2,5 };
	//maximumUniqueSubarray(a);

	//string number = "9964-";
	//reformatNumber(number);



