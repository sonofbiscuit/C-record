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
#include "time.h"
#include "StlStringChange.h"
#include <numeric>

/* #pragma GCC optimize(2) //O2优化
*/
using namespace std;
using namespace strtool;

/*
string s1 = "hiya";    // 拷贝初始化
string s2("hello");    // 直接初始化
string s3(10, 'c');    // 直接初始化
*/
//多用直接初始化，少用拷贝初始化

//循环时，前自增大于后自增，前自增返回的是自增后的自己，后自增返回的是自增前自己的副本(临时变量)


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


//最长严格递增子序列
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

	for (int i = 0; i < nums2.size(); i++) {
		while (!st.empty() && st.top() < nums2[i]) { //因为是找第一个比它大的，所以找到后就可以出栈了
			mp[st.top()] = nums2[i]; //
			st.pop();
		}
		st.push(nums2[i]);
	}

	for (int j = 0; j < n; j++) {
		if (mp.count(nums1[j])) { //里面有
			ans.emplace_back(mp[nums1[j]]);
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


class Solution22 {
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
int findMaxForm(vector<string>& strs, int m, int n) {
	return 0;
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



//丑数II   264
int nthUglyNumber(int n) {
	if (n == 1) {
		return 1;
	}
	int count = 1;
	//如何确保顺序?
	//priority_queue每次取出最小值
	priority_queue<int,vector<int>,greater<int>> pr_que;
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
	for (int i = 1; i < n; ++i) {
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



//474 一和零
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
	return one;
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





//1857. 有向图中最大颜色值
int largestPathValue(string colors, vector<vector<int>>& edges) {
	int node_size = colors.size();
	vector<int> node_in(node_size,0);
	unordered_map<int, vector<int>> graph(node_size);

	for (auto a : edges) {
		node_in[a[1]]++; //保存所有结点的入度
		graph[a[0]].emplace_back(a[1]); //存图
	}
	
	return 0;
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

	/*
	* 组合总和IV  377
	* 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
	* 注意：
		   顺序不同的序列被视作不同的组合
	*/



};


/**/
int main() {
	//clock_t start, end;
	//start = clock();









	//end = clock();
	//cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
	return 0;
}


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



