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


/*
C++中强制类型转换操作符有static_cast、dynamic_cast、const_cast、reinterpert_cast四个
static_cast<int>(x) 作用等同于 (int)x
dynamic_cast是将一个基类对象指针（或引用）转换到继承类指针，dynamic_cast会根据基类指针是否真正指向继承类指针来做相应处理。


static_cast 也能进行指针的转换， 比如基类和派生类


把派生类的指针或引用转换成基类表示，称之为上行转换。
把基类指针或引用转换成派生类表示，称之为下行转换。

在类层次间进行上行转换时，dynamic_cast和static_cast的效果是一样的；
在进行下行转换时，dynamic_cast具有类型检查的功能，比static_cast更安全。

例子：
class B{
public:
	int m_iNum;
	virtual void foo();
}

class D:public B{
public:
	char* m_szName[100];
}
 void func(){
	D* pd1 = static_cast<D*>(pd);
	D* pd2 = dynamic_cast<D*>(pd);
 }

  如果pd指向的是派生类D， 那么pd1和pd2是一样的
  如果pd指向的是基类B，那么pd1将是一个指向该对象的指针，pd2将是一个空指针

  对于对于dynamic_cast的转换， 基类（父类）中必须要定义一个虚函数，不然会报错。
  虚函数不仅仅是实现多态性的一个重要标志，同时也是dynamic_cast转换能够进行的前提条件。
   
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


//============================随机化快速排序==========================================
//随机化版本的快排
int partion(vector<int>& nums, int left, int right) {
	srand((int)time(0));
	int randidx = left + (rand() % (right - left + 1));
	swap(nums[left], nums[left + (rand() % (right - left + 1))]);
	int pivot = nums[left];
	int i = left, j = right;
	while (i < j) {
		while (i < j and nums[j] >= pivot) {
			j -= 1;
		}
		nums[i] = nums[j];
		while (i < j and nums[i] <= pivot) {
			i += 1;
		}
		nums[j] = nums[i];
	}
	nums[i] = pivot;
	return i;
}

void quicksort(vector<int>& vec, int low, int high) {
	int pivot = partion(vec, low, high);
	partion(vec, low, pivot - 1);
	partion(vec, pivot + 1, high);
}

//============================快速选择==========================================
//快速选择算法
class quickSelect {
public:
	int findKthLargest(vector<int>& nums, int k) {
		int n = nums.size();
		return quickselect(nums, 0, n - 1, n - k);
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
				swap(nums[++i], nums[j]); //很巧妙
			}
		}
		swap(nums[right], nums[i + 1]);
		return i + 1;
	}

};




//======================================并查集==================================================

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


//================================二叉搜索树==================================================
template<typename V>
struct BSTNode {
	BSTNode* _lchild;
	BSTNode* _rchild;
	V val;

	BSTNode(const V& value)
		: _lchild(nullptr),
		_rchild(nullptr),
		val(value) {}

};

template<typename V>
class BST {
typedef BSTNode<V> node;
private:
	node* _root;
public:
	

	BST() :_root(nullptr) {}

	//查找
	node* find(V value) {
		return _find(_root, value);
	}

	node* _find(node* root, const V& value) {
		if (root == nullptr) {
			return nullptr;
		}
		if (root->val > value) {
			return _find(root->_lchild, value);
		}
		else if (root->val < value) {
			return _find(root->_rchild, value);  
		}
		else {
			return root;
		}
	}

	//插入
	bool insert(V value) {
		return _insert(_root, value);
	}

	bool _insert(node* root, const V& value) {
		if (root == nullptr) {
			root = new BSTNode(value);
			return true;
		}

		if (root->val > value) {
			_insert(root->_lchild, value);
		}
		else if (root->val < value) {
			_insert(root->_rchild, value);
		}
		else {
			return false;
		}
	}

	//删除
	bool delete_node(V value) {
		return _delete(_root, value);
	}

	bool _delete(node* root, const V& value) {
		if (root == nullptr) {
			return false;
		}
		if (root->_lchild == nullptr && root->_rchild == nullptr) { // 只有root
			root == nullptr;
			return true;
		}

		if (root->val > value) {
			return _delete(root->_lchild, value);
		}
		else if (root->value < value) {
			return _delete(root->_rchild, value);
		}
		else { //equal
			node* temp = nullptr;
			if (root->_lchild == nullptr && root->_rchild == nullptr) { //删除节点为叶子节点
				delete root;
				root == nullptr;
				return true;
			}
			else if (root->_lchild == nullptr) { //带有右孩子的结点
				temp = root;
				root = root->_rchild;
				delete(temp);
				temp = nullptr;
				return true;
			}
			else if (root->_rchild == nullptr) { //带有左孩子的结点
				temp = root;
				root = root->_lchild;
				delete(temp);
				temp = nullptr;
				return true;
			}
			else { //左右节点均不为空
				node* right_first;
				right_first = root->_rchild;
				while (right_first->_rchild) {
					right_first = right_first->_lchild;
				}
				//已经到达了最左的结点
				swap(root->val, right_first->val);
				_delete(root->_rchild, value);
				return true;
			}
		}
	}

};





//================================平衡二叉树==================================================
//multiset内部实现为平衡二叉树











//========================================哈希表 hashset,仅仅存储对象============================
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


//===============================================HashMap,存储的键值对========================================
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


//==========================================小根堆、小顶堆========================================================
template<typename T>
class minHeap {
private:
	T* heap;
	int currentSize; //当前堆中的元素数量
	int maxSize; //所允许的最大元素数
	static const int default_size = 30;

public:
	//初始化一个空堆
	minHeap(int sz = default_size) {
		maxSize = (sz >= default_size) ? sz : default_size;
		heap = new T[maxSize];
		currentSize = 0;
	}
	// 构造函数，通过一个数组建立堆
	minHeap(T arr[]) {
		int n = length(arr);
		maxSize = (default_size < n) ? n : default_size;
		heap = new T[maxSize];
		for (int i = 0; i < n; ++i) {
			heap[i] = arr[i]; //初始化
		}
		currentSize = n;
		int current_pos = (currentSize - 2) / 2;
		while (current_pos) {
			adjustDown(current_pos, currentSize - 1);
			current_pos--; //逐步向前
		}
		//找初始位置。
	}

	//插入
	bool insert(const T& x) {
		if (currentSize == maxSize) {
			cout << "heap full" << endl;

		}
		heap[currentSize] = x; //插入到最后一个位置
		adjustUp(currentSize);
		currentSize++;
		return true;
	}

	//移除
	bool removeMin() {
		if (currentSize == 0) {
			cout << "heap empty" << endl;
			return false;
		}
		T x = heap[0];
		heap[0] = heap[currentSize - 1];
		currentSize--;
		adjustDown(0, currentSize - 1);
		return true;
	}

	void output()
	{
		for (int i = 0; i < currentSize; i++)
		{
			cout << heap[i] << " ";
		}
		cout << endl;
	}

	//向下调整法
	void adjustDown(int start, int end) {
		int cur = start;
		int min_child = 2 * cur + 1; //取左孩子
		T temp_root = heap[cur]; //取当前孩子的父结点，为start处的结点
		while (min_child <= end) {//没到最后一个结点
			if (min_child<end && heap[min_child] > heap[min_child + 1]) {
				//找出左右孩子结点的最小的一个结点
				min_child++; //右节点比左节点小，则变到右节点
			}
			if (heap[min_child] >= temp_root) {
				//满足父结点小于最小的子节点
				return;
			}
			else { //不满足,那么得交换
				heap[cur] = heap[min_child]; //子节点上移到父结点位置
				cur = min_child; //cur下移到子节点的位置
				min_child = min_child * 2 + 1; //找下面的子节点
			}
		}
		//找到 temp_root的正确位置
		heap[cur] = temp_root;
	}


	//向上调整法
	//从start到0，调整为最小堆
	void adjustUp(int start) { //找到存放temp的合适位置
		int cur = start;
		int cur_parent = (cur - 1) / 2;
		T temp = heap[cur];
		while (cur > 0) {
			if (temp > heap[cur_parent]) {
				break;
			}
			else { //不满足小顶堆的性质,开始交换
				heap[cur] = heap[cur_parent];
				cur = cur_parent; //上移
				cur_parent = (cur_parent - 1) / 2;
			}
		}
		//跳出while的条件，cur==0 或者 temp > heap[cur_parent]
		//当 temp>heap[cur_parent]时，cur处是没有更新的
		heap[cur] = temp;
	}
};

/*
调用
minHeap<int> h;
	h.insert(8);
	h.insert(5);
	h.insert(7);
	h.insert(9);
	h.insert(6);
	h.insert(12);
	h.insert(15);
	h.output();

	int out;
	cout << static_cast<int> (h.RemoveMin(out)) << endl;
	h.output();

	int arr[10] = { 15,19,13,12,18,14,10,17,20,11 };
	MinHeap<int> h1(arr,10);
	h1.output();
*/


//======================================================实现next_permutation==========================
/* 31. 下一个排列
比如有序列{1,2,3}
那么下一个排列依次为
			1 2 3
			1 3 2
			2 1 3
			2 3 1
			3 1 2
			3 2 1
next_permutation，就是每次找到一个大于当前序列的排列，且变大的幅度尽可能小  比如：1 2 3 =>> 1 3 2

void nextPermutation(vector<int>& nums) {
	int n = nums.size();
	int first_less = nums.size()-1;
	int first_large = nums.size()-1;
	while (first_less > 0 && nums[first_less] > nums[first_less - 1]) {
		first_less--;
	}
	if (first_less >= 1) {
		while (first_large >= first_less && nums[first_large] < nums[first_less - 1]) {
			first_large--;
		}
		swap(nums[first_large], nums[first_less]);
	}
	reverse(nums.begin() + first_less, nums.end());
	
}*/
