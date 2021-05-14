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


//============================����==========================================
//������汾�Ŀ���
int partion(vector<int>& vec, int low, int high) {
	if (low >= high) {
		return low;
	}
	swap(vec[low], vec[low + rand() % (high - low + 1)]);//����pivot��ѡȡ
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



//============================���鼯==================================================

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


//================================ƽ�������==================================================
//multiset�ڲ�ʵ��Ϊƽ�������









//��ϣ�� hashset,�����洢����
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


//HashMap,�洢�ļ�ֵ��
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



