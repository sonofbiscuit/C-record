#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <typeinfo>
using namespace std;


/*402.给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。 */
/*316.给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。*/
/*321.
给定长度分别为 m 和 n 的两个数组，其元素由 0-9 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 k (k <= m + n) 个数字拼接成一个新的数，
要求从同一个数组中取出的数字保持其在原数组中的相对顺序。
求满足该条件的最大数。结果返回一个表示该最大数的长度为 k 的数组。

说明: 请尽可能地优化你算法的时间和空间复杂度。
*/

class Solution {
public:
	//402
	string removeKdigits(string num, int k) {
		vector<char> ans_vec;
		for (auto temp : num) {
			while (ans_vec.size() > 0 && temp < ans_vec.back() && k) {
				ans_vec.pop_back();
				k -= 1;
			}
			ans_vec.push_back(temp);
		}

		//没删够k个数
		for (; k > 0; k--) {
			ans_vec.pop_back();
		}

		//倘若第一个是0
		int first_zero_tag = 1;
		string ans_str="";
		for (auto a : ans_vec) {
			if (first_zero_tag==1 && a == '0') {
				continue;
			}
			first_zero_tag = 0;
			ans_str += a;;
		}
		return ans_str == "" ? "0" : ans_str;
	}

	//316
	string removeDuplicateLetters(string s) {
		string ans_str="0";
		vector<bool> tag(128, false);
		unordered_map<char,int> ans_set;
		for (auto a : s) {
			ans_set[a] += 1;
		}// count

		for (char str : s) {
			ans_set[str]--;
			if (tag[str]==true) {
				continue;
			}
			while (str < ans_str.back() && ans_set[ans_str.back()]>0) {
				tag[ans_str.back()] = false;
				ans_str.pop_back();
			}
			ans_str.push_back(str);
			tag[str] = true;
		}
		ans_str.erase(0, 1);
		for (auto a : ans_str)
			cout << a << endl;
		
		return ans_str;
	}
};
/*
int main() {
	Solution sol;
	/*string num = "10";
	int k = 1;
	sol.removeKdigits(num, k);
	string s = "abacb";
	sol.removeDuplicateLetters(s);


	return 0;
}
*/