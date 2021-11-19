/*
* lambda�ĵݹ�
* 
std::function<void(int, int)> backtrack = [&](int index, int rec) {  // index��ʾ��ǰλ��, rec��¼�ַ�ʹ�������������
	if (index == (int)unique_subarr.size()) { //�߽�����
		ans = max(ans, (int)__builtin_popcount(rec));
		return;
	}
	//ѡ
	if ((unique_subarr[index] & rec) == 0) { //=0��ʾ���ظ�Ԫ��
		backtrack(index + 1, rec | unique_subarr[index]);
	}
	//��ѡ
	backtrack(index + 1, rec);
};
*/


/*
map
//map<key, value>
//unique keys
//ordered

multimap
//Multiple elements in the container can have equivalent keys.
//ordered
//multimap<key,value>

unordered_map
unordered_multimap
//unordered
//...

//unordered_map��count��find
count�����ҵ�������
find���ص�����, auto a = xxx.find();  a->first, a->second...


map�ŵ㣺

�����ԣ�����map�ṹ�����ŵ㣬��Ԫ�ص��������ںܶ�Ӧ���ж���򻯺ܶ�Ĳ���
��������ڲ�ʵ��һ�������ʹ��map�ĺܶ������lgn��ʱ�临�Ӷ��¾Ϳ���ʵ�֣����Ч�ʷǳ��ĸ�
ȱ�㣺 �ռ�ռ���ʸߣ���Ϊmap�ڲ�ʵ���˺��������Ȼ���������Ч�ʣ�������Ϊÿһ���ڵ㶼��Ҫ���Ᵽ�游�ڵ㡢���ӽڵ�ͺ�/�����ʣ�ʹ��ÿһ���ڵ㶼ռ�ô����Ŀռ�

���ô���������Щ��˳��Ҫ������⣬��map�����ЧһЩ

unordered_map��

�ŵ㣺 ��Ϊ�ڲ�ʵ���˹�ϣ�����������ٶȷǳ��Ŀ�
ȱ�㣺 ��ϣ��Ľ����ȽϺķ�ʱ��
���ô������ڲ������⣬unordered_map����Ӹ�ЧһЩ����������������⣬���ῼ��һ����unordered_map
=================================================================

set   //unique key
multiset //multi key
//The value of an element is also the key used to identify it.
//ordered


unordered_set
unordered_multiset  //multi
//unordered
//The value of an element is also the key used to identify it.

��ʱʹ��set

������Ҫ��������ݡ�
���ǽ����ò���ӡ/�������ݣ�������˳�򣩡�
������ҪԪ�ص�ǰ��/��̡�
����set������ģ�������ǿ�����setԪ����ʹ��binary_search������lower_bound������upper_bound����֮��ĺ�������Щ����������unordered_set������ʹ�á�

��ʱʹ��unordered_set
������Ҫ����һ�鲻ͬ��Ԫ�أ����Ҳ���Ҫ����
������Ҫ��Ԫ�ط��ʣ����ޱ�����

unordered_set�Ĺ��췽����
	unordered_set<int> set1; //������set1
	unordered_set<int> set2(set1);   //��������
	unordered_set<int> set3(set1.begin(),set1.end()); //����������
	unordered_set<int> set4(arr,arr+5);  //���鹹��
	unordered_set<int> set5(move(set1)); //�ƶ�����
	unordered_set<int> set6 = {1,2,3,4,5};  //ʹ��initializer_list��ʼ��


push_back()������������ĩβ�����Ԫ��ʱ�����ȴ�����Ԫ�أ�Ȼ���ٽ���Ԫ���ƶ����߿����������У�
emplace_back()�����ĵײ�ʵ����ֱ��������β����������Ԫ�أ������ڿ��������ƶ�Ԫ�صĹ��̡�
*/


/*
* �����ھ�Ϊchar����
isalpha()    �ж��ַ��Ƿ�Ϊ��ĸ
isdigit()    �Ƿ�������
isalnum()    �Ƿ��� ��ĸ������
islower()    �Ƿ�ΪСд��ĸ
isupper()    �Ƿ�Ϊ��д��ĸ

����ǣ����ط�0
������Ƿ���0
*/





//decltype()  ���ز���������������

// &��ֵ����    &&��ֵ����   �磺int &&k=i+k����C++ 11��


/*int* p = new int;
	*p = 3;
	cout << "p address: " << p << endl;
	delete p;
	p = nullptr;
	cout << "p address after delete: " << p << endl;
	long* p1 = new long;
	*p1 = 100;
	cout << "p address after creat p1: " << p << endl;
	cout << "p1 address: " << p1 << endl;
	p = new int;
	*p = 2;
	cout << "p address after revalued: " << p << endl;
	cout << "p1 address after p was revalued: " << p1 << endl;
	delete p1;
	
	
	//nullptr ��ַΪ0x00000000
	
	*/



//========================================================Ԥ����===============================
/*
Ԥ����ָ����#��ͷ��#Ϊ��һ���ַ�

#include

==>>>   #include <xxx.h>  ��Ҫ�������ļ��Լ�������������Ԥ����������ϵͳĬ��Ŀ¼���������ڵ�·�����в��ң������ڰ���ϵͳ���Դ��Ĺ����ļ�
		#include "xxx.h" ��Ҫ�������ļ���˫������������Ԥ���������ڳ���Դ�ļ����ڵ�Ŀ¼���в��ң���δ�ҵ���ȥϵͳĬ��Ŀ¼���ң������ڰ����Լ���д��˽��ͷ�ļ�

#include ���ܻ�������ذ��������⡣һ�����������a.h��b.h����ͷ�ļ�����a.h��b.h����ͬʱ�ֶ�������c.h�����Ǹó���Ͱ���������c.h������һЩ�����»ᵼ�³���Ĵ���
��ͨ������������н��


#define, #undef   �궨�壬ȡ���궨��

#if, #elif, #else ,#endif  ������if elseif else

#ifdef, #idndef, #endif  ����ж��壬���û���壬����
	#ifndef MYHEAD_H
	#define MYHEAD_H
	#include "myHead.h"
	#endif



__FILE__��ʾ����������ڵ�Դ�ļ����ļ���
__LINE__��ʾ�����������Դ�ļ��е�λ����Ϣ
#lineָ���������������������ֵ
#line number["filename"]  �ڶ��������ļ����ǿ�ʡ�Եģ�������ָ�����к���ʵ�ʵ���һ�����Żᷢ�����á�
void test(){
	cout<<"current file: "<<__FILE__<<endl; //current file: d:\test.cpp
	cout<<"current line: "<<__LINE__<<endl; //50

	#line 1000 "abc"
	cout<<"current file: "<<__FILE__<<endl; //current file: d:\abc
	cout<<"current line: "<<__LINE__<<endl; //1001
}


#error  ָ���ڱ���ʱ������������Ϣ�����Է�������ֵĴ���
void test() {
#define OPTION 2
#if OPTION  ==1
	cout << "op1" << endl;
#elif OPTION==2
	cout << "op2" << endl;
#else
#error ILLEGAL OPTION!
#endif
}


#pragmaָ��
��ָ���������趨��������״̬������ָʾ���������һЩ�ض��Ķ���

*/



//append
/*
1. ��string�������C-string
string s = "123";
const char* a = "456";
s.append(a); => output:123456
//���C-string��һ����
s.append(a,2);//���a��ǰ2���ַ�  => output:12345

2.��string����� string
string s = "123";
string a = "456789";
s.append(a); =>ouput: 123456789
s.append(a,2,2); a�д�2��ʼ��2���ַ����ӵ�s��  =>ouput: 12367

s.append(a.begin()+1,a.end()) =>ouput: 12356789

3.��string����Ӷ���ַ�
string s = "123";
s.append(5,'!'); =>output:123!!!!!
*/

//==========================================================C++�е�0 
/*
0     int
0L    long
0LL   long long
0.0   double
0.0f  float
0.0L  long double

��:  
vector<int> vec;
//vec�е�ֵ��ͻᳬ��int�ķ�Χ����ô�������µ�д��
accumulate(vec.begin(),vec.end(),0LL);
*/    



//================================================================���ֲ����е�ϸ��===================================
/*
���ڶ��ֲ��ҵ�mid=(left+right)/2
���дΪ mid = left+(right - left)/2
��ֹleft+right���
*/



//=================================================������ʽ==========================================
/*

===============�޶���?
abcd?
	��ʾd����ַ�����0�λ���1��
	abcd? ����ƥ�䵽abc��abcd


===============�޶���*
* ƥ�� 0~����ַ�
	��ab*c ����ƥ��ac  abc  abbbbbbbc


===============�޶���+
+ ƥ�� 1~����ַ�
	ab+c  =>>>>> abc   abbbbbbc   �޷�ƥ��ac

===============�޶���{}
{} ab{5}c
	ƥ�� abbbbbc =>>>b������Ϊ5
   ab{2,5}c
	ƥ��abbc ~ abbbbbc   =>>>>>b��
	ƥ��ab...c��  b���ִ�������2������Ϊ2~5
   ab{2,}c

����ƥ�� ��γ��ֵ�ab
��ôʹ��(ab)����޶���  =>>>>> (ab)  ����(ab)+


==============="��"����� |

a (cat|dog)
	��ʾƥ�� a cat ���� a dog
	���Ų�����

===============�ַ���
[abc]+ ��ʾƥ�������������������ַ�   =>>>����aabbcc  abc  ababcccbcbcbababb����ƥ��
[a-zA-Z0-9]+ �������е�Ӣ���ַ�������
�ڷ�����ǰ����^  ���ʾƥ����˼����������������

===============Ԫ�ַ�
\d �����ַ�
\w �����ַ���Ӣ�ġ����ּ��»��ߣ�
\s �հ׷�������Tab�ͻ��з���

\b �����н�����

\D ���� **��**�����ַ�
\W ���� **��**�����ַ�
\S ���� **��**�հ��ַ�

. ���������ַ������ǲ��������з�

^ ��ƥ������    =>>>>>����^aֻ��ȥƥ�����׵ĵ�һ��a     ��abbbbbbbacba �� ֻ��ƥ�����׵�һ��a
$ ƥ����β      =>>>>> a$ֻ��ȥƥ����β��a   ��abbbbbbbaca  ֻ��ƥ������a  �� aaabbac�򲻻�ƥ�䵽a


===============̰��������ƥ��
* + {}��ƥ���ַ�����ʱ�򣬻ᾡ���ܶ��ȥƥ���ַ�

���� <span><b>test sample text</b></span>
	��ô <.+>�����̰��ƥ�䣬��ʾ<>֮�䣬���Գ��������ַ�����ô����ƥ����Ϊ��һ�� < �����һ�� >
	�� <.+>��Ϊ����ƥ�䣺
		<.+?>��ʾ < >֮��������ַ�����0�λ���1�Σ���ô�ͻ�ƥ�����е�<span>  <b>  </b>  </span>

*/

/*
ʵ��1��
	RGB��ɫƥ��

	����#00
		#ffffff
		#ffaaff
		#00hh00
		#aabbcc
		#000000
		#ffffffff

	#[a-fA-F0-9]{6}\b  ������ȷ��ƥ��
	#����ͷ��#��[a-fA-F0-9]�������ƥ��a-f  A-F  0-9 ���ַ��� {6}������Ϊ6   \b�����н�����

ʵ��2��
	IPv4�ĵ�ַƥ��

	����
		123
		255.255.255.0
		192.168.1.1
		0.0.0.0
		256.1.1.1
		123.123.0
		999.999.99.999

	\d+\.\d+\.\d+\.\d+  .Ϊ�����ַ������Ҫʹ��\����ת�롣
	���regex����ƥ�� xxx.xxx.xxx.xxx ����xΪ���֣�������Ϊip��ַ�ķ�Χ��СΪ0-255�������Ҫ���з�Χ����

	\b((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)\b
	��λ������\b��������β
	(25[0-5]|2[0-4]\d|[01]?\d\d?)��ʾ
		��25��ͷ�Ļ�����������ֻ����0-5
		��2��ͷ�Ļ���������������0-4
		��0��1��ͷ�Ļ������������ֿ���Ϊ��������ֵ

		��Ϊip��ַ����Ϊ1λ��2λ��3λ����� [01]?\d\d? ������һλ����λ����0��1��ͷ����λ���������25��2��ͷ�������ǰ��

	((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}
		�����\.��ʾƥ��ip��ַ��.    {3}��ʾ���ָ�ʽ�������Σ���ƥ���˷���Ҫ���ip��ַ��ǰ����

	�������һ�Σ������ĶΣ�ֱ�Ӹ���(25[0-5]|2[0-4]\d|[01]?\d\d?)����

	���ѿ�ͷ�ͽ�β����\b��ƥ���ַ��ı߽�

*/


//[captures] (params) mutable-> return-type{...} //lambda ���ʽ��������ʽ
/*
����captures�е����ݣ������¼���
	Ϊ��                     �������ڿ���ʹ��lambda���ڷ�Χ�ڵ����оֲ�����
	=						�������ڿ���  ��ֵ���ݷ�ʽ  ʹ��lambda���ڷ�Χ�ڵ����оֲ�����
	&						�������ڿ���  �����ô��ݷ�ʽ  ʹ��lambda���ڷ�Χ�ڵ����оֲ�����
	This					�������ڿ���ʹ��lambda����  �෶Χ��  �����оֲ�����
	a(����ֲ���������)        �� a ��ֵ���д���,�������� �����޸� ���ݽ����� a �Ŀ�������ΪĬ������º����� const �ģ�Ҫ�޸Ĵ��ݽ����Ŀ������������ mutable ���η���
	&a						�� a �����÷�ʽ���д���,�������� �����޸� ���ݽ����� a ��ֵ
	a,&b					�� a ��ֵ���д��ݣ� b �����÷�ʽ���д���
	=,&a,&b                 ���� a �� b �������÷�ʽ���ݣ� lambda���ڷ�Χ�ڵ������������� ֵ���� ��ʽ���д���
	&,a,b                   ���� a �� b ����ֵ���ݷ�ʽ���ݣ� ������������ ���ô��� ��ʽ���д���
	...

params�����б�
	��C++����ͨ�����Ĳ����б���һ����˼

mutable 
    �ùؼ�����Ϊһ�����η�����Ĭ�ϵ�����£�lambda�ķ���ֵΪconst��������mutable������ȡ���䳣�����ʡ�
	��ʹ����mutable����ô�����б��Ǳز����ٵģ���ʹ��Ϊ�ա�
	
return-type
	�����ķ������ͣ���C++����ͨ�����ķ�������һ������Ҫ������׷��lambda�ķ���ֵ�����͡���lambdaû�з���ֵʱ���ɲ�д��

...
	�����塣�������ڳ��˿���ʹ�ò����б��еĲ���֮�⣬������ʹ��capture����ı���
*/


//============================== vector  array   ����=======���ߵ���ͬ��Ͳ�ͬ��
/*
* ��ͬ��
1) �����������ƣ�������ʹ�ñ�׼����ı�ʾ����������ÿ��Ԫ��(vector��array�����±������[]����������)
2) ���ߵĴ洢���������ģ����Խ����������

* ��ͬ��
1)  �����ǲ���ȫ�ģ�array��vector�Ƚϰ�ȫ����Ч�ı�����Խ�������(vector��1.5/2�����ݣ���������йء�array��α���Խ�磿ʹ��at()?)
2)  array���������洢����ͬ���ڴ�����ջ���У�vector����߳������ɴ洢�����ѣ��С�
3)  array���Խ�һ������ֵ����һ��array����vector�ƺ�Ҳ�У������������鲻�С�
4)  vector���ڱ䳤�����������Ը������ݵĲ����ɾ�����¹�����������������array������Ϊ������
5)  vector��array�ṩ�˸��õķ��ʻ��ƣ�������ʹ��front(),back(),at()�ȷ��ʷ�ʽ��at()���Ա���a[-1]����Խ������⣩��ʹ�÷��ʸ��Ӱ�ȫ��
	������ֻ��ʹ���±���ʣ����׳���Խ�����
6)  vector��array�ṩ�˸��õı������ƣ�������������ͷ����������
7)  vector��array�ṩ��size()��empty(),������ֻ��ͨ��sizeof()/strlen()�Լ�������������ȡ��С���Ƿ�Ϊ�ա�
8)  vector��array�ṩ������������������ݽ�������swap()�Ļ��ƣ���������ڽ���ֻ��ͨ�������ķ�ʽ���������
9)  array�ṩ�˳�ʼ�����г�Ա�ķ���fill() ��
10) ����vector�Ķ�̬�ڴ�仯���ƣ��ڲ����ɾ��ʱ����Ҫ���ǵ������Ƿ��ʧЧ�����⡣
11) vector��array������������������������ɺ󣬻��Զ����ͷ�����ռ�õ��ڴ档
    �������飬�����new[]/malloc����Ŀռ䣬����ʹ�ö�Ӧ��delete[]��free���ͷ��ڴ档

*/

/*

*c++ ��queueû��clear���ַ���

���queueʵ����գ������¼��ַ���

1)	
	queue<int> q1;
	...
	q1 = queue<int>();

2)
	while(!q.empty()){
		q.pop();
	}

3)
	void clear(queue<int>& q){  <==���Ч��ͬʱҲ������STL�ķ��
		queue<int> temp;
		swap(temp,q);
	}
	
	����
	template<typename T>
	void clear(queue<T>& q) {
		queue<T> temp;
		swap(temp, q);
	}
*/


//============================================unique_ptr����ָ��=======================
/*
#include <iostream>
#include <memory>

using namespace std;
struct Task {
	int tid;
	Task(int id) :tid(id) {
		std::cout << "Task::constructor" << endl;
	}
	~Task() {
		cout << "Task::destructor" << endl;
	}

};

int main() {
	// �ն��� unique_str
	std::unique_ptr<int> ptr1;  //C++ 11

	//���ptr1�Ƿ�Ϊ��
	if (!ptr1) {
		cout << "ptr1 is empty" << endl;
	}

	//���ptr1�Ƿ�Ϊ��
	if (ptr1 == nullptr) {
		std::cout << "ptr1 is empty" << endl;
	}

	//�����µ�unique_ptr����
	//!!!!!!����ͨ����ֵ�ķ�����������
	//std::unique_ptr<Task> ptr2 = new Task(); //compile error
	//�ڴ�������ʱ���乹�캯���д���ԭʼָ��
	unique_ptr<Task> ptr2(new Task(2));
	//����
	std::unique_ptr<Task> ptr3(new std::unique_ptr<Task>::element_type(3));   //elemwnt_type����˼

	//C++14  ����unique_ptr����    std::make_unique
	unique_ptr<Task> ptr4 = make_unique<Task>(4);
	//����ptr4����Ӧ��tid
	cout << "id for ptr4: " << ptr4->tid << endl;
	//ʹ��get()��ȡ��������ָ��
	Task* ptr44 = ptr4.get();
	cout << "use get() to cout ptr44->id" << ptr44->tid << endl;

	//����ptr4
	if (ptr4 != nullptr) {
		cout << "ptr4 is not empty" << endl;
	}
	cout << "use reset() to reset ptr4" << endl;
	//ptr4.release()
	//ptr4�����������ָ����Ŀ���Ȩ�������ر����ָ�룬��ptr4�ÿգ����ͷ��ڴ�
	//reset()���ͷ�ptr4������ָ����Ȼ������ptr4��ֵ
	ptr4.reset();
	cout << "ptr4 already reset()" << endl;
	if (ptr4 == nullptr) {
		cout << "after reset(), ptr4 is empty" << endl;
	}

	//unique_ptr���󲻿ɸ��ƣ� �����ƶ�
	//��ͨ��moveת��unique_ptr�Ķ���
	//ͨ��ԭʼָ�봴��ptr5
	unique_ptr<Task> ptr5(new Task(5));
	//��ptr5�й���ָ�������Ȩ����ptr55
	unique_ptr<Task> ptr55(move(ptr5));
	//����д��  unique_ptr<Task> ptr55 = move(ptr5);
	cout << "ptr5 move to ptr55: " << ptr55->tid << endl;
	if (ptr5 == nullptr) {
		cout << "after move(ptr5), ptr5 is empty" << endl;
	}
	if (ptr4 == nullptr) {
		cout << "ptr4 is empty" << endl;
	}
	cout << "move ptr4 to ptr55" << endl;
	ptr55 = move(ptr4);
	if (ptr55 == nullptr) {
		cout << "ptr55 is empty" << endl;
	}
	
	return 0;
}

*/
/*
====================================unique_ptr�Ĳ���
unique_ptr�ĵ�һ������Ϊָ���������ͣ��ڶ�������Ϊ��ָ���Զ����������
functionָʾ�ص�����(����������)��Ȼ����ָ���ʼ��ʱָ���������������
*/
//ֱ��ʹ��lambda���ʽ��Ϊ��������

/*
#include <iostream>
#include <memory>
#include <functional>

using namespace std;

std::unique_ptr<int, std::function<void(int*)>> testPtr(
		new int[5], [](int* p) {
		if (p) {
			delete[] p;
		}
	});


auto delInt = [](int* pData) {
	if (pData) {
		cout << "destructor " << endl;
		delete[] pData;
	}
};

std::unique_ptr<int, decltype(delInt)> dataPtr(new int[2], delInt);

//�����Զ�������ݽṹ���������ڴ�����������unique_ptrָ��ʱ��Ҳ�ɶ�������������

struct cur {
	int curid;
	cur(int id) :curid(id) {
		cout << "constructor" << endl;
	}
	~cur() {
		cout << "destructor" << endl;
	}

};

void cur_free(struct cur* p) {
	delete(p);
}
struct cur* headers = NULL;   //��ָ��headers
unique_ptr<struct cur, function<void(struct cur*)>>headerPtr(
	headers, [](struct cur* p) {
	if (p != nullptr) {
		cur_free(p);  //������������
	}
});

//headerPtr = make_unique<cur>(5);
//cout << headerPtr->curid << endl;


auto delcur = [](struct cur* pHeaders) {
	if (pHeaders != nullptr) {
		cur_free(pHeaders);    //libcurl����������
	}
};
unique_ptr<struct curl_slist, decltype(delcur)> headersPtr;

*/


//==================================explicit ��ֹ��ʽ����
/*
class Test1
{
public:
	Test1(int n)
	{
		num = n;
	}//��ͨ���캯��
private:
	int num;
};
class Test2
{
public:
	explicit Test2(int n)      //<------���ڵ������Ĺ��캯����������explicit����ֹ��ʽ����
	{
		num = n;
	}//explicit(��ʽ)���캯��
private:
	int num;
};

int main() {
	Test1 t1 = 2;   //<----��ʽ���óɹ�
	//Test2 t2 = 3;  compile error    explicit��ֹ����ʽ����
	Test2 t3(3);   //��ʽ����
}
*/


//==================================vector��һЩ��һ���Ĺ��췽��
/*
constructor

// default(1)  Ĭ�Ϲ��캯��
explicit vector(const allocator_type&  alloc = allocator_type());


//fill(2)  ������СΪn��vector��ÿ��Ԫ�ظ�ֵΪval
explicit vector(size_type n, const value_type& val = value_type(),const allocator_type& alloc = allocator_type());


//fill(3)  �����������������󣨻�Ϊָ�룩�������ߵ����ݿ�����vector������ǰ�ṹ���Ӧ��С��������
template<class InputIterator>
vector(InputIterator first, InputIterator last, const allocator_type& alloc = alloctor_type());


//copy(4)  ����vector���󣬽��п���
vector(const vector& x);


//vector�ڲ����Լ���allocator���ܹ�ʵ�ֶ�̬�ڴ�ķ������ͷţ�һ�㲻��ֱ��ʹ��new��delete������ʹ���ڴ�������ͷŸ��Ӱ�ȫ
vector& operator= (const vector& x);

*/


/*
vector��ֱ�ӽ��д�С�Ƚ�
	vector<int> a{ 1,1,1,0,0 };
	vector<int> b{ 1,1,1,1,0 };
	cout << (a > b) << endl;      <--- ��� 0
	cout << (a < b) << endl;      <--- ��� 1

	��aΪ{2,1,1,0,0}
	��a>b���1
*/


//�������� partial_sort  ��  ��n��Ԫ�� nth_element
/*
partial_sort���������أ�
	partial_sort(�������ʼλ�ã�����Ľ���λ�ã����ҵĽ���λ��)
	partial_sort(�������ʼλ�ã�����Ľ���λ�ã����ҵĽ���λ�ã��Զ�������򷽷�)

bool func(int a, int b){
	return a>b;  //����
}
partial_srot(vec.begin(),vec.begin()+5,vec.end(),func)

partial_sort(vec.begin(), vec.begin() + 4, vec.end(), [](const auto& a, const auto& b) {
		return a > b;
	});


nth_elemen()ò�ƾ��ǿ���
*/











