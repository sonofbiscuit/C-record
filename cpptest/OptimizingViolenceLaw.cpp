/*
e.g
Given a (0-indexed) integer array nums and two integers low and high, return the number of nice pairs.

#defination
A nice pair is a pair (i, j) where 0 <= i < j < nums.length and low <= (nums[i] XOR nums[j]) <= high.

1 <= nums.length <= 2 * 10^4
1 <= nums[i] <= 2 * 10^4
1 <= low <= high <= 2 * 10^4
*/


/*
_mm<bit_width>_<name>_<data_type>

<bit_width> 表明了向量的位长度，对于128位的向量，这个参数为空，对于256位的向量，这个参数为256。
<name>描述了内联函数的算术操作。
<data_type> 标识函数主参数的数据类型。
-ps 包含float类型的向量
pd 包含double类型的向量


epi8/epi16/epi32/epi64 包含8位/16位/32位/64位的有符号整数
epu8/epu16/epu32/epu64 包含8位/16位/32位/64位的无符号整数
si128/si256 未指定的128位或者256位向量
m128/m128i/m128d/m256/m256i/m256d 当输入向量类型与返回向量的类型不同时，标识输入向量类型

__m128	包含4个float类型数字的向量
__m128d	包含2个double类型数字的向量
__m128i	包含若干个整型数字的向量
__m256	包含8个float类型数字的向量
__m256d	包含4个double类型数字的向量
__m256i	包含若干个整型数字的向量


__v8si   __v16hi  __v32qi  __v32qu
*/

/*
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

#include<immintrin.h>  //AVX2
#define us unsigned short
//65535  



const int N = 10000;
unsigned short a[N] __attribute__((aligned(32)));
#define _mm256_cmple_epi16(a,b) ((__m256i)((__v16hi)a<=(__v16hi)b))
#define _mm256_cmpge_epi16(a,b) ((__m256i)((__v16hi)a>=(__v16hi)b))



using namespace std; 




class Sol {
private:
	us _a1[20001];
public:
	int violence(vector<int> _a, int l, int  r) {
		int ans = 0;
		int n = _a.size();
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				int temp = _a[i] ^ _a[j];
				ans += (temp >= l) && (temp <= r);
			}
		}
		return ans;
	}

	//op1
	//vector=>array
	int violence1(vector<int> _a, int l, int  r) {
		int *a = &_a[0];
		int ans = 0;
		int n = _a.size();
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				int temp = a[i] ^ a[j];
				ans += (temp >= l) && (temp <= r);
			}
		}
		return ans;
	}


	//op2
	//unsigned short  and cpu parallel computing
	int violence2(vector<int> _a, int l, int  r) {
		int ans = 0;
		int n = _a.size();
		for (int i = 0; i < n; i++) {
			_a1[i] = _a[i];
		}
		for (int* i = &_a[0], *end = &_a[0] + n; i != end; i++) {
			int* j = i + 1;
			int* _end = end - 8;
			for (; j < _end; j += 8) { //still the violence law, but 8 at a time
				cout << *j << endl;
				us x = *i ^ *j;
				us x1 = *i ^ *(j + 1);
				us x2 = *i ^ *(j + 2);
				us x3 = *i ^ *(j + 3);
				us x4 = *i ^ *(j + 4);
				us x5 = *i ^ *(j + 5);
				us x6 = *i ^ *(j + 6);
				us x7 = *i ^ *(j + 7);

				ans += (x >= l && x <= r) + (x1 >= l && x1 <= r) + (x2 >= l && x2 <= r) + (x3 >= l && x3 <= r) + (x4 >= l && x4 <= r) +
					(x5 >= l && x5 <= r) + (x6 >= l && x6 <= r) + (x7 >= l && x7 <= r);
				
			}
			//end for when j>=_end , while j still < end
			for (; j < end; j++) {
				ans += (*i ^ *j) >= l && (*i ^ *j) <= r;
			}
		}
		cout << ans << endl;
		return ans;
	}


	
	//AVX2 Instruction Set
	__attribute__((no_sanitize_address, no_sanitize_memory))
	__attribute__((target("avx2")))
	int violence2(vector<us> _a, us l, us  r) {
		int n = _a.size();
		int ans = 0;
		int d = 0;
		for (int i = 0; i < n; i++) {
			a[i] = _a[i];
		}
		sort(a, a + n);
		//storage
		__m256i L = _mm256_set1_epi16(l),
				R=_mm256_set1_epi16(r),
				mask = _mm256_set1_epi16(1);
		for (int i = 0; i < n; i++) {
			if (i && a[i] == a[i - 1]) {
				ans += d;
				continue;
			}
			us x = a[i];
			int j = i + 1;
			d = 0;
			__m256i X = __mm256_set1_epi16(x),
				res = __mm256_setzero_si256();
			for (; (j & 15) && j < n; j++) {
				d += (x ^ a[j]) >= l && (x ^ a[j]) <= r;
			}
			if (j < n) {
				for (__m256i* I = (__m256i*)(a + j), *end = (__m256i*)(a + (n & ~15)); I != end; I++) {
					__m256i Y = _mm256_xor_si256(X, *I);
					res = _mm256_add_epi16(res, _mm256_and_si256(_mm256_and_si256(_mm256_cmpge_epi16(Y, L), _mm256_cmple_epi16(Y, R)), mask));
				}
			}
			for (int k = 0; k < 16; k++) {
				d += ((us*)&res)[k];
			}
			for (j = max(j, n & ~15); j < n; j++) {
				d += (x ^ a[j]) >= l && (x ^ a[j]) <= r;
			}
			ans += d;
		}
		return ans;
	}



};
*/


/*
int main() {
	//warning:函数使用了堆栈的40028个字节，超过了/analyze:stacksize:16384
	//原因:数组定义过大
	//vector<int> a = { 9,8,4,2,1,9,8,4,2,1 };
	//int low = 5, high = 14;
	//Sol sol;
	//sol.violence2(a, low, high);

}*/