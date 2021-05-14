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

using namespace std;

/*
   迭代器辅助函数	                    功能
   advance(it, n)	         it 表示某个迭代器，n 为整数。该函数的功能是将 it 迭代器前进或后退 n 个位置。
distance(first, last)	     first 和 last 都是迭代器，该函数的功能是计算 first 和 last 之间的距离。
   begin(cont)	             cont 表示某个容器，该函数可以返回一个指向 cont 容器中第一个元素的迭代器。
   end(cont)	             cont 表示某个容器，该函数可以返回一个指向 cont 容器中最后一个元素之后位置的迭代器。
   prev(it) / prev(it,n)	 it 为指定的迭代器，该函数默认可以返回一个指向上一个位置处的迭代器。注意，it 至少为双向迭代器。n为距离，左+，右-，左移2位是2，右-2
   next(it)	                 it 为指定的迭代器，该函数默认可以返回一个指向下一个位置处的迭代器。注意，it 最少为前向迭代器。

*/


/*
* 判断序列中的元素是否满足条件
any_of()  任何一个元素都满足时，返回true
all_of()  所有元素都满足条件时，返回true
none_of() 没有元素满足条件时，返回true

vector<int> nums;
any_of(nums.begin(),nums.end(),[](const auto& a){
	return a!=0;  //是否任何一个元素都不为0
});

*/



/*
sort(edges.begin(), edges.end(), [](const auto& e1, const auto& e2) {
	auto&& [x1, x2, x3] = e1;
	auto&& [y1, y2, y3] = e2;
	return x3 < y3;
});*/

/*
//sort内置函数
less<type>()    //从小到大排序 <
grater<type>()  //从大到小排序 >
less_equal<type>()  //  <=
gtater_equal<type>()//  >=
*/
/*
======string to int======
//实现任意类型的转换
template<typename in_type, typename out_type>
out_type convert(const& in_type t){
	stringstream stream;
	stream<<t; //向流中传值
	out_type ans; //存值
	stream>>ans; //向ans中写入值
	return ans;
}

//方法二
//1、先将string转为char数字型字符串数组
//2、使用stoi转换为int

c_str()的作用是以char*类型返回string内字符串


*/


/*
substr(i,length);  //xxx[i]~xxx[i+length-1]
substr(i);    //xxx[i]~xxx.end()
*/
/*
lower_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于或等于num的数字，找到返回该数字的地址，
不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

upper_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于num的数字，
找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

*/

//双端队列实现单调队列 ,  自带priority_queue<>
//priority_queue<int, vector<int>, greater<int> > que; //顶部为最小值
//priority_queue<int> que;   顶部为最大值
class MonotonicQueue {
private:
	deque<int> deq;
public:
	void push(int x) {
		while (deq.back() < x && !deq.empty()) {
			deq.pop_back();
		}
		deq.emplace_back(x);
	}

	int max_ele() {
		return deq.front();
	}

	void delete_ele(int x) { // delete可根据要求更改
		if (!deq.empty() && deq.front() == x) {
			deq.pop_front();
		}
	}

};


//============================排序==========================================
//随机化版本的快排
int partion(vector<int>& vec, int low, int high) {
	if (low >= high) {
		return low;
	}
	swap(vec[low], vec[low + rand() % (high - low + 1)]);//更改pivot的选取
	int pivot = vec[low];
	int i = low, j = high;
	while (i < j) {
		while (i<j&&vec[j]>pivot) {
			j--;
		}
		vec[i] = vec[j];
		while (i < j&&vec[i] < pivot) {
			i++;
		}
		vec[j] = vec[i];
	}
	vec[i] = pivot;
	return j;
}

void quicksort(vector<int>& vec, int low, int high) {
	int pivot = partion(vec, low, high);
	partion(vec, low, pivot - 1);
	partion(vec, pivot + 1, high);
}



//============================并查集==================================================

//初始化
int fa[100];
int ranks[100];

inline void init(int n) {
	for (int i = 0; i < n; i++) {
		fa[i] = i;//i的父结点设为自己
	}
}

//查询
int find(int x) {
	if (fa[x] == x) {
		return x;
	}
	return find(fa[x]);
}

//合并
inline void merge(int i, int j) {
	fa[find(i)] = find(j); //将前者的父结点设为后者
}//直接合并不是非常合理，因为单个元素合并会造成单链形状


//路径压缩合并
//把沿途每个节点的父结点都设为根节点
/*
int find(int x){
	if(x==find(x)){
		return x;
	}else{
		fa[x]=find(fa[x]);
		return fa[x];
	}
}
*/
int find3(int x) {
	return x == fa[x] ? x : (fa[x] = find3(fa[x]));
}

//按秩合并
//把简单的树往复杂的树上合并，这样合并后，到根节点距离变长的节点个数比较少
void inline init1(int n) {
	for (int i = 0; i < n; i++) {
		fa[i] = i;
		ranks[i] = 1;//rank表示结点深度
	}
}

inline void merge1(int i, int j) {
	int x = find(i), y = find(j);//先找到i，j的父结点
	if (ranks[x] <= ranks[y]) {
		fa[x] = y;
	}
	else {
		ranks[y] = x;
	}
	if (ranks[x] == ranks[y] && x != y) {//深度相同但根节点不同
		ranks[y]++;
	}
}


//================================平衡二叉树==================================================
//multiset内部实现为平衡二叉树









//哈希表 hashset,仅仅存储对象
//此处使用链地址法，基准质数取1023

class MyHashSet {
private:
	vector<list<int>> myHash;
	static const int base = 997; //constant, read-only,  new MyHashSet will not reload base
	static int hash(int x) {
		return x % base;
	}
public:
	/** Initialize vector*/
	MyHashSet() :myHash(base) {}

	void add(int key) {
		int temp = hash(key);
		for (auto i = myHash[temp].begin(); i != myHash[temp].end(); i++) {
			if (*i == key)
				return;
		}
		myHash[temp].push_back(key);
	}

	void remove(int key) {
		int temp = hash(key);
		for (auto i = myHash[temp].begin(); i != myHash[temp].end(); i++) {
			if (*i == key)
				myHash[temp].erase(i);
			return;
		}
	}

	/** Returns true if this set contains the specified element */
	bool contains(int key) {
		int temp = hash(key);
		for (auto i = myHash[temp].begin(); i != myHash[temp].end(); i++) {
			if (*i == key)
				return true;
		}
		return false;
	}
};


//HashMap,存储的键值对
class MyHashMap {
private:
	vector<list<pair<int, int>>> myHash;
	static const int base = 997; //constant, read-only,  new MyHashSet will not reload base
	static int hash(int x) {
		return x % base;
	}
public:
	/** Initialize your data structure here. */
	MyHashMap() :myHash(base) {

	}

	/** value will always be non-negative. */
	void put(int key, int value) {
		int temp = hash(key); //key的映射
		for (auto i = myHash[temp].begin(); i != myHash[temp].end(); i++) {
			if (((*i).first) == key){
				(*i).second = value;
				return;
			}
		}
		myHash[temp].push_back(make_pair(key, value));
	}

	/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
	int get(int key) {
		int temp = hash(key);
		for (auto i = myHash[temp].begin(); i != myHash[temp].end(); i++) {
			if ((*i).first == key)
				return (*i).second;
		}
		return -1;
	}

	/** Removes the mapping of the specified value key if this map contains a mapping for the key */
	void remove(int key) {
		int temp = hash(key);
		for (auto i = myHash[temp].begin(); i != myHash[temp].end(); i++) {
			if ((*i).first == key) {
				myHash[temp].erase(i);
				return;
			}
		}
	}
};


//=========================================字典树/前缀树=============================================
class Trie {
private:
	bool isEnd;
	Trie* next[26];
public:
	/** Initialize your data structure here. */
	Trie() {
		isEnd = false;
		memset(next, 0, sizeof(next));
	}

	/** Inserts a word into the trie. */
	void insert(string word) {
		Trie* node = this;
		for (auto a : word) {
			if (node->next[a - 'a'] == NULL) {
				node->next[a - 'a'] = new Trie();
			}
			node = node->next[a - 'a'];
		}
		node->isEnd = true;
	}

	/** Returns if the word is in the trie. */
	bool search(string word) {
		Trie* node = this;
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
		Trie* node = this;
		for (auto a : prefix) {
			if (node->next[a - 'a'] == NULL) {
				return false;
			}
			node = node->next[a - 'a'];
		}
		return true;
	}
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */



