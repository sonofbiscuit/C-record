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
   ��������������	                    ����
   advance(it, n)	         it ��ʾĳ����������n Ϊ�������ú����Ĺ����ǽ� it ������ǰ������� n ��λ�á�
distance(first, last)	     first �� last ���ǵ��������ú����Ĺ����Ǽ��� first �� last ֮��ľ��롣
   begin(cont)	             cont ��ʾĳ���������ú������Է���һ��ָ�� cont �����е�һ��Ԫ�صĵ�������
   end(cont)	             cont ��ʾĳ���������ú������Է���һ��ָ�� cont ���������һ��Ԫ��֮��λ�õĵ�������
   prev(it) / prev(it,n)	 it Ϊָ���ĵ��������ú���Ĭ�Ͽ��Է���һ��ָ����һ��λ�ô��ĵ�������ע�⣬it ����Ϊ˫���������nΪ���룬��+����-������2λ��2����-2
   next(it)	                 it Ϊָ���ĵ��������ú���Ĭ�Ͽ��Է���һ��ָ����һ��λ�ô��ĵ�������ע�⣬it ����Ϊǰ���������

*/


/*
* �ж������е�Ԫ���Ƿ���������
any_of()  �κ�һ��Ԫ�ض�����ʱ������true
all_of()  ����Ԫ�ض���������ʱ������true
none_of() û��Ԫ����������ʱ������true

vector<int> nums;
any_of(nums.begin(),nums.end(),[](const auto& a){
	return a!=0;  //�Ƿ��κ�һ��Ԫ�ض���Ϊ0
});

*/



/*
sort(edges.begin(), edges.end(), [](const auto& e1, const auto& e2) {
	auto&& [x1, x2, x3] = e1;
	auto&& [y1, y2, y3] = e2;
	return x3 < y3;
});*/


/*
//sort���ú���
less<type>()    //��С�������� <
grater<type>()  //�Ӵ�С���� >
less_equal<type>()  //  <=
gtater_equal<type>()//  >=
*/


/*
======string to int======
//ʵ���������͵�ת��
template<typename in_type, typename out_type>
out_type convert(const& in_type t){
	stringstream stream;
	stream<<t; //�����д�ֵ
	out_type ans; //��ֵ
	stream>>ans; //��ans��д��ֵ
	return ans;
}

//������
//1���Ƚ�stringתΪchar�������ַ�������
//2��ʹ��stoiת��Ϊint

c_str()����������char*���ͷ���string���ַ���


*/


/*
substr(i,length);  //xxx[i]~xxx[i+length-1]
substr(i);    //xxx[i]~xxx.end()
*/
/*
lower_bound( begin,end,num)���������beginλ�õ�end-1λ�ö��ֲ��ҵ�һ�����ڻ����num�����֣��ҵ����ظ����ֵĵ�ַ��
�������򷵻�end��ͨ�����صĵ�ַ��ȥ��ʼ��ַbegin,�õ��ҵ������������е��±ꡣ

upper_bound( begin,end,num)���������beginλ�õ�end-1λ�ö��ֲ��ҵ�һ������num�����֣�
�ҵ����ظ����ֵĵ�ַ���������򷵻�end��ͨ�����صĵ�ַ��ȥ��ʼ��ַbegin,�õ��ҵ������������е��±ꡣ

*/


/*
C++��ǿ������ת����������static_cast��dynamic_cast��const_cast��reinterpert_cast�ĸ�
static_cast<int>(x) ���õ�ͬ�� (int)x
dynamic_cast�ǽ�һ���������ָ�루�����ã�ת�����̳���ָ�룬dynamic_cast����ݻ���ָ���Ƿ�����ָ��̳���ָ��������Ӧ����


static_cast Ҳ�ܽ���ָ���ת���� ��������������


���������ָ�������ת���ɻ����ʾ����֮Ϊ����ת����
�ѻ���ָ�������ת�����������ʾ����֮Ϊ����ת����

�����μ��������ת��ʱ��dynamic_cast��static_cast��Ч����һ���ģ�
�ڽ�������ת��ʱ��dynamic_cast�������ͼ��Ĺ��ܣ���static_cast����ȫ��

���ӣ�
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

  ���pdָ�����������D�� ��ôpd1��pd2��һ����
  ���pdָ����ǻ���B����ôpd1����һ��ָ��ö����ָ�룬pd2����һ����ָ��

  ���ڶ���dynamic_cast��ת���� ���ࣨ���ࣩ�б���Ҫ����һ���麯������Ȼ�ᱨ��
  �麯����������ʵ�ֶ�̬�Ե�һ����Ҫ��־��ͬʱҲ��dynamic_castת���ܹ����е�ǰ��������
   
*/





//˫�˶���ʵ�ֵ������� ,  �Դ�priority_queue<>
//priority_queue<int, vector<int>, greater<int> > que; //����Ϊ��Сֵ
//priority_queue<int> que;   ����Ϊ���ֵ
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

	void delete_ele(int x) { // delete�ɸ���Ҫ�����
		if (!deq.empty() && deq.front() == x) {
			deq.pop_front();
		}
	}

};


//============================�������������==========================================
//������汾�Ŀ���
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

//============================����ѡ��==========================================
//����ѡ���㷨
class quickSelect {
public:
	int findKthLargest(vector<int>& nums, int k) {
		int n = nums.size();
		return quickselect(nums, 0, n - 1, n - k);
	}

	//quickselect�ж��±�
	//random �����漴�±�
	//partion���룬����ÿ��������
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
				swap(nums[++i], nums[j]); //������
			}
		}
		swap(nums[right], nums[i + 1]);
		return i + 1;
	}

};




//======================================���鼯==================================================

//��ʼ��
int fa[100];
int ranks[100];

inline void init(int n) {
	for (int i = 0; i < n; i++) {
		fa[i] = i;//i�ĸ������Ϊ�Լ�
	}
}

//��ѯ
int find(int x) {
	if (fa[x] == x) {
		return x;
	}
	return find(fa[x]);
}

//�ϲ�
inline void merge(int i, int j) {
	fa[find(i)] = find(j); //��ǰ�ߵĸ������Ϊ����
}//ֱ�Ӻϲ����Ƿǳ�������Ϊ����Ԫ�غϲ�����ɵ�����״


//·��ѹ���ϲ�
//����;ÿ���ڵ�ĸ���㶼��Ϊ���ڵ�
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

//���Ⱥϲ�
//�Ѽ򵥵��������ӵ����Ϻϲ��������ϲ��󣬵����ڵ����䳤�Ľڵ�����Ƚ���
void inline init1(int n) {
	for (int i = 0; i < n; i++) {
		fa[i] = i;
		ranks[i] = 1;//rank��ʾ������
	}
}

inline void merge1(int i, int j) {
	int x = find(i), y = find(j);//���ҵ�i��j�ĸ����
	if (ranks[x] <= ranks[y]) {
		fa[x] = y;
	}
	else {
		ranks[y] = x;
	}
	if (ranks[x] == ranks[y] && x != y) {//�����ͬ�����ڵ㲻ͬ
		ranks[y]++;
	}
}


//================================����������==================================================
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

	//����
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

	//����
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

	//ɾ��
	bool delete_node(V value) {
		return _delete(_root, value);
	}

	bool _delete(node* root, const V& value) {
		if (root == nullptr) {
			return false;
		}
		if (root->_lchild == nullptr && root->_rchild == nullptr) { // ֻ��root
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
			if (root->_lchild == nullptr && root->_rchild == nullptr) { //ɾ���ڵ�ΪҶ�ӽڵ�
				delete root;
				root == nullptr;
				return true;
			}
			else if (root->_lchild == nullptr) { //�����Һ��ӵĽ��
				temp = root;
				root = root->_rchild;
				delete(temp);
				temp = nullptr;
				return true;
			}
			else if (root->_rchild == nullptr) { //�������ӵĽ��
				temp = root;
				root = root->_lchild;
				delete(temp);
				temp = nullptr;
				return true;
			}
			else { //���ҽڵ����Ϊ��
				node* right_first;
				right_first = root->_rchild;
				while (right_first->_rchild) {
					right_first = right_first->_lchild;
				}
				//�Ѿ�����������Ľ��
				swap(root->val, right_first->val);
				_delete(root->_rchild, value);
				return true;
			}
		}
	}

};





//================================ƽ�������==================================================
//multiset�ڲ�ʵ��Ϊƽ�������











//========================================��ϣ�� hashset,�����洢����============================
//�˴�ʹ������ַ������׼����ȡ1023

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


//===============================================HashMap,�洢�ļ�ֵ��========================================
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
		int temp = hash(key); //key��ӳ��
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


//=========================================�ֵ���/ǰ׺��=============================================
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


//==========================================С���ѡ�С����========================================================
template<typename T>
class minHeap {
private:
	T* heap;
	int currentSize; //��ǰ���е�Ԫ������
	int maxSize; //����������Ԫ����
	static const int default_size = 30;

public:
	//��ʼ��һ���ն�
	minHeap(int sz = default_size) {
		maxSize = (sz >= default_size) ? sz : default_size;
		heap = new T[maxSize];
		currentSize = 0;
	}
	// ���캯����ͨ��һ�����齨����
	minHeap(T arr[]) {
		int n = length(arr);
		maxSize = (default_size < n) ? n : default_size;
		heap = new T[maxSize];
		for (int i = 0; i < n; ++i) {
			heap[i] = arr[i]; //��ʼ��
		}
		currentSize = n;
		int current_pos = (currentSize - 2) / 2;
		while (current_pos) {
			adjustDown(current_pos, currentSize - 1);
			current_pos--; //����ǰ
		}
		//�ҳ�ʼλ�á�
	}

	//����
	bool insert(const T& x) {
		if (currentSize == maxSize) {
			cout << "heap full" << endl;

		}
		heap[currentSize] = x; //���뵽���һ��λ��
		adjustUp(currentSize);
		currentSize++;
		return true;
	}

	//�Ƴ�
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

	//���µ�����
	void adjustDown(int start, int end) {
		int cur = start;
		int min_child = 2 * cur + 1; //ȡ����
		T temp_root = heap[cur]; //ȡ��ǰ���ӵĸ���㣬Ϊstart���Ľ��
		while (min_child <= end) {//û�����һ�����
			if (min_child<end && heap[min_child] > heap[min_child + 1]) {
				//�ҳ����Һ��ӽ�����С��һ�����
				min_child++; //�ҽڵ����ڵ�С����䵽�ҽڵ�
			}
			if (heap[min_child] >= temp_root) {
				//���㸸���С����С���ӽڵ�
				return;
			}
			else { //������,��ô�ý���
				heap[cur] = heap[min_child]; //�ӽڵ����Ƶ������λ��
				cur = min_child; //cur���Ƶ��ӽڵ��λ��
				min_child = min_child * 2 + 1; //��������ӽڵ�
			}
		}
		//�ҵ� temp_root����ȷλ��
		heap[cur] = temp_root;
	}


	//���ϵ�����
	//��start��0������Ϊ��С��
	void adjustUp(int start) { //�ҵ����temp�ĺ���λ��
		int cur = start;
		int cur_parent = (cur - 1) / 2;
		T temp = heap[cur];
		while (cur > 0) {
			if (temp > heap[cur_parent]) {
				break;
			}
			else { //������С���ѵ�����,��ʼ����
				heap[cur] = heap[cur_parent];
				cur = cur_parent; //����
				cur_parent = (cur_parent - 1) / 2;
			}
		}
		//����while��������cur==0 ���� temp > heap[cur_parent]
		//�� temp>heap[cur_parent]ʱ��cur����û�и��µ�
		heap[cur] = temp;
	}
};

/*
����
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


//======================================================ʵ��next_permutation==========================
/* 31. ��һ������
����������{1,2,3}
��ô��һ����������Ϊ
			1 2 3
			1 3 2
			2 1 3
			2 3 1
			3 1 2
			3 2 1
next_permutation������ÿ���ҵ�һ�����ڵ�ǰ���е����У��ұ��ķ��Ⱦ�����С  ���磺1 2 3 =>> 1 3 2

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
