/*
* lambda的递归
* 
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

//unordered_map的count和find
count返回找到的数量
find返回迭代器, auto a = xxx.find();  a->first, a->second...


map优点：

有序性，这是map结构最大的优点，其元素的有序性在很多应用中都会简化很多的操作
红黑树，内部实现一个红黑书使得map的很多操作在lgn的时间复杂度下就可以实现，因此效率非常的高
缺点： 空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间

适用处：对于那些有顺序要求的问题，用map会更高效一些

unordered_map：

优点： 因为内部实现了哈希表，因此其查找速度非常的快
缺点： 哈希表的建立比较耗费时间
适用处：对于查找问题，unordered_map会更加高效一些，因此遇到查找问题，常会考虑一下用unordered_map
=================================================================

set   //unique key
multiset //multi key
//The value of an element is also the key used to identify it.
//ordered


unordered_set
unordered_multiset  //multi
//unordered
//The value of an element is also the key used to identify it.

何时使用set

我们需要有序的数据。
我们将不得不打印/访问数据（按排序顺序）。
我们需要元素的前任/后继。
由于set是有序的，因此我们可以在set元素上使用binary_search（），lower_bound（）和upper_bound（）之类的函数。这些函数不能在unordered_set（）上使用。

何时使用unordered_set
我们需要保留一组不同的元素，并且不需要排序。
我们需要单元素访问，即无遍历。

unordered_set的构造方法：
	unordered_set<int> set1; //创建空set1
	unordered_set<int> set2(set1);   //拷贝构造
	unordered_set<int> set3(set1.begin(),set1.end()); //迭代器构造
	unordered_set<int> set4(arr,arr+5);  //数组构造
	unordered_set<int> set5(move(set1)); //移动构造
	unordered_set<int> set6 = {1,2,3,4,5};  //使用initializer_list初始化


push_back()函数在向容器末尾添加新元素时，会先创建该元素，然后再将该元素移动或者拷贝到容器中；
emplace_back()函数的底层实现是直接在容器尾部创建该新元素，不存在拷贝或者移动元素的过程。
*/


/*
* 括号内均为char类型
isalpha()    判断字符是否为字母
isdigit()    是否是数字
isalnum()    是否是 字母或数字
islower()    是否为小写字母
isupper()    是否为大写字母

如果是，返回非0
如果不是返回0
*/





//decltype()  返回操作数的数据类型

// &左值引用    &&右值引用   如：int &&k=i+k；（C++ 11）


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
	
	
	//nullptr 地址为0x00000000
	
	*/



//========================================================预编译===============================
/*
预处理指令以#开头且#为第一个字符

#include

==>>>   #include <xxx.h>  将要包含的文件以尖括号括起来，预处理程序会在系统默认目录或者括号内的路径进行查找，常用于包含系统中自带的公共文件
		#include "xxx.h" 将要包含的文件以双引号引起来，预处理程序会在程序源文件所在的目录进行查找，若未找到再去系统默认目录查找，常用于包含自己编写的私有头文件

#include 可能会产生多重包含的问题。一个程序包含了a.h和b.h两个头文件，但a.h和b.h可能同时又都包含了c.h，于是该程序就包含了两次c.h，这在一些场合下会导致程序的错误
可通过条件编译进行解决


#define, #undef   宏定义，取消宏定义

#if, #elif, #else ,#endif  类似于if elseif else

#ifdef, #idndef, #endif  如果有定义，如果没定义，结束
	#ifndef MYHEAD_H
	#define MYHEAD_H
	#include "myHead.h"
	#endif



__FILE__表示本行语句所在的源文件的文件名
__LINE__表示本行语句所在源文件中的位置信息
#line指令可以重设这两个变量的值
#line number["filename"]  第二个参数文件名是可省略的，并且其指定的行号在实际的下一行语句才会发生作用。
void test(){
	cout<<"current file: "<<__FILE__<<endl; //current file: d:\test.cpp
	cout<<"current line: "<<__LINE__<<endl; //50

	#line 1000 "abc"
	cout<<"current file: "<<__FILE__<<endl; //current file: d:\abc
	cout<<"current line: "<<__LINE__<<endl; //1001
}


#error  指令在编译时输出编译错误信息，可以方便检查出现的错误
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


#pragma指令
该指令用来来设定编译器的状态或者是指示编译器完成一些特定的动作

*/



//append
/*
1. 向string后面添加C-string
string s = "123";
const char* a = "456";
s.append(a); => output:123456
//添加C-string的一部分
s.append(a,2);//添加a的前2个字符  => output:12345

2.向string后添加 string
string s = "123";
string a = "456789";
s.append(a); =>ouput: 123456789
s.append(a,2,2); a中从2开始的2个字符连接到s后  =>ouput: 12367

s.append(a.begin()+1,a.end()) =>ouput: 12356789

3.向string后面加多个字符
string s = "123";
s.append(5,'!'); =>output:123!!!!!
*/

//==========================================================C++中的0 
/*
0     int
0L    long
0LL   long long
0.0   double
0.0f  float
0.0L  long double

如:  
vector<int> vec;
//vec中的值求和会超出int的范围，那么按照如下的写法
accumulate(vec.begin(),vec.end(),0LL);
*/    



//================================================================二分查找中的细节===================================
/*
对于二分查找的mid=(left+right)/2
最好写为 mid = left+(right - left)/2
防止left+right溢出
*/



//=================================================正则表达式==========================================
/*

===============限定符?
abcd?
	表示d这个字符出现0次或者1次
	abcd? 可以匹配到abc和abcd


===============限定符*
* 匹配 0~多个字符
	如ab*c 可以匹配ac  abc  abbbbbbbc


===============限定符+
+ 匹配 1~多个字符
	ab+c  =>>>>> abc   abbbbbbc   无法匹配ac

===============限定符{}
{} ab{5}c
	匹配 abbbbbc =>>>b的数量为5
   ab{2,5}c
	匹配abbc ~ abbbbbc   =>>>>>b的
	匹配ab...c中  b出现次数大于2的数量为2~5
   ab{2,}c

若想匹配 多次出现的ab
那么使用(ab)后跟限定符  =>>>>> (ab)  或者(ab)+


==============="或"运算符 |

a (cat|dog)
	表示匹配 a cat 或者 a dog
	括号不能少

===============字符类
[abc]+ 表示匹配出现在括号里的所有字符   =>>>比如aabbcc  abc  ababcccbcbcbababb均能匹配
[a-zA-Z0-9]+ 代表所有的英文字符和数字
在方括号前加上^  则表示匹配除了尖括号内以外的数字

===============元字符
\d 数字字符
\w 单词字符（英文、数字及下划线）
\s 空白符（包含Tab和换行符）

\b 代表行结束符

\D 代表 **非**数字字符
\W 代表 **非**单词字符
\S 代表 **非**空白字符

. 代表任意字符，但是不包含换行符

^ 会匹配行首    =>>>>>比如^a只会去匹配行首的第一个a     如abbbbbbbacba ， 只会匹配行首第一个a
$ 匹配行尾      =>>>>> a$只会去匹配行尾的a   如abbbbbbbaca  只会匹配最后的a  ， aaabbac则不会匹配到a


===============贪婪与懒惰匹配
* + {}在匹配字符串的时候，会尽可能多的去匹配字符

若有 <span><b>test sample text</b></span>
	那么 <.+>会进行贪婪匹配，表示<>之间，可以出现任意字符，那么最终匹配结果为第一个 < 到最后一个 >
	将 <.+>变为懒惰匹配：
		<.+?>表示 < >之间的任意字符出现0次或者1次，那么就会匹配所有的<span>  <b>  </b>  </span>

*/

/*
实例1：
	RGB颜色匹配

	若有#00
		#ffffff
		#ffaaff
		#00hh00
		#aabbcc
		#000000
		#ffffffff

	#[a-fA-F0-9]{6}\b  可做正确的匹配
	#代表开头的#，[a-fA-F0-9]代表可以匹配a-f  A-F  0-9 的字符， {6}代表长度为6   \b代表行结束符

实例2：
	IPv4的地址匹配

	若有
		123
		255.255.255.0
		192.168.1.1
		0.0.0.0
		256.1.1.1
		123.123.0
		999.999.99.999

	\d+\.\d+\.\d+\.\d+  .为特殊字符，因此要使用\进行转译。
	这个regex可以匹配 xxx.xxx.xxx.xxx 其中x为数字，但是因为ip地址的范围大小为0-255，因此需要进行范围限制

	\b((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)\b
	首位的两个\b限制了首尾
	(25[0-5]|2[0-4]\d|[01]?\d\d?)表示
		以25开头的话，后续数字只能是0-5
		以2开头的话，后续的数字是0-4
		以0或1开头的话，后续的数字可以为两个任意值

		因为ip地址可以为1位、2位、3位，因此 [01]?\d\d? 包含了一位、两位、以0、1开头的三位的情况，以25和2开头的情况在前面

	((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}
		加入的\.表示匹配ip地址的.    {3}表示这种格式出现三次，即匹配了符合要求的ip地址的前三段

	对于最后一段，即第四段，直接复制(25[0-5]|2[0-4]\d|[01]?\d\d?)即可

	最后把开头和结尾加上\b来匹配字符的边界

*/


//[captures] (params) mutable-> return-type{...} //lambda 表达式的完整形式
/*
对于captures中的内容，有以下几种
	为空                     函数体内可以使用lambda所在范围内的所有局部变量
	=						函数体内可以  以值传递方式  使用lambda所在范围内的所有局部变量
	&						函数体内可以  以引用传递方式  使用lambda所在范围内的所有局部变量
	This					函数体内可以使用lambda所在  类范围内  的所有局部变量
	a(任意局部变量均可)        将 a 按值进行传递,函数体内 不能修改 传递进来的 a 的拷贝，因为默认情况下函数是 const 的，要修改传递进来的拷贝，可以添加 mutable 修饰符。
	&a						将 a 按引用方式进行传递,函数体内 可以修改 传递进来的 a 的值
	a,&b					将 a 按值进行传递， b 按引用方式进行传递
	=,&a,&b                 除了 a 和 b 按照引用方式传递， lambda所在范围内的其他变量按照 值传递 方式进行传递
	&,a,b                   除了 a 和 b 按照值传递方式传递， 其他变量按照 引用传递 方式进行传递
	...

params参数列表
	和C++中普通函数的参数列表是一个意思

mutable 
    该关键字作为一个修饰符。在默认的情况下，lambda的返回值为const，当加了mutable，可以取消其常量性质。
	若使用了mutable，那么参数列表是必不可少的，即使它为空。
	
return-type
	函数的返回类型，与C++中普通函数的返回类型一样，主要是用来追踪lambda的返回值的类型。当lambda没有返回值时，可不写。

...
	函数体。函数体内除了可以使用参数列表中的参数之外，还可以使用capture捕获的变量
*/


//============================== vector  array   数组=======三者的相同点和不同点
/*
* 相同点
1) 都和数组相似，都可以使用标准数组的表示方法来访问每个元素(vector和array都对下标运算符[]进行了重载)
2) 三者的存储都是连续的，可以进行随机访问

* 不同点
1)  数组是不安全的，array和vector比较安全，有效的避免了越界等问题(vector有1.5/2倍扩容，与编译器有关。array如何避免越界？使用at()?)
2)  array对象和数组存储在相同的内存区域（栈）中，vector对象催出在自由存储区（堆）中。
3)  array可以将一个对象赋值给另一个array对象（vector似乎也行？），但是数组不行。
4)  vector属于变长的容器，可以根据数据的插入和删除重新构造容器容量。但是array和数组为定长。
5)  vector和array提供了更好的访问机制，即可以使用front(),back(),at()等访问方式（at()可以避免a[-1]访问越界的问题），使得访问更加安全。
	而数组只能使用下标访问，容易出现越界错误。
6)  vector和array提供了更好的遍历机制，有正向迭代器和反向迭代器。
7)  vector和array提供了size()和empty(),而数组只能通过sizeof()/strlen()以及遍历计数来获取大小和是否为空。
8)  vector和array提供了两个容器对象的内容交换，即swap()的机制，而数组对于交换只能通过遍历的方式逐个交换。
9)  array提供了初始化所有成员的方法fill() 。
10) 由于vector的动态内存变化机制，在插入何删除时，需要考虑迭代器是否会失效的问题。
11) vector和array在声明变量后，在声明周期完成后，会自动地释放其所占用的内存。
    对于数组，如果用new[]/malloc申请的空间，必须使用对应的delete[]和free来释放内存。

*/

/*

*c++ 的queue没有clear这种方法

想对queue实现清空，有以下几种方法

1)	
	queue<int> q1;
	...
	q1 = queue<int>();

2)
	while(!q.empty()){
		q.pop();
	}

3)
	void clear(queue<int>& q){  <==最高效，同时也保持了STL的风格
		queue<int> temp;
		swap(temp,q);
	}
	
	或者
	template<typename T>
	void clear(queue<T>& q) {
		queue<T> temp;
		swap(temp, q);
	}
*/


//============================================unique_ptr智能指针=======================
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
	// 空对象 unique_str
	std::unique_ptr<int> ptr1;  //C++ 11

	//检查ptr1是否为空
	if (!ptr1) {
		cout << "ptr1 is empty" << endl;
	}

	//检查ptr1是否为空
	if (ptr1 == nullptr) {
		std::cout << "ptr1 is empty" << endl;
	}

	//创建新的unique_ptr对象
	//!!!!!!不能通过赋值的方法创建对象
	//std::unique_ptr<Task> ptr2 = new Task(); //compile error
	//在创建对象时在其构造函数中传递原始指针
	unique_ptr<Task> ptr2(new Task(2));
	//或者
	std::unique_ptr<Task> ptr3(new std::unique_ptr<Task>::element_type(3));   //elemwnt_type的意思

	//C++14  创建unique_ptr对象    std::make_unique
	unique_ptr<Task> ptr4 = make_unique<Task>(4);
	//访问ptr4所对应的tid
	cout << "id for ptr4: " << ptr4->tid << endl;
	//使用get()获取管理对象的指针
	Task* ptr44 = ptr4.get();
	cout << "use get() to cout ptr44->id" << ptr44->tid << endl;

	//重置ptr4
	if (ptr4 != nullptr) {
		cout << "ptr4 is not empty" << endl;
	}
	cout << "use reset() to reset ptr4" << endl;
	//ptr4.release()
	//ptr4会放弃对它所指对象的控制权，并返回保存的指针，将ptr4置空，不释放内存
	//reset()会释放ptr4对它所指对象，然后重置ptr4的值
	ptr4.reset();
	cout << "ptr4 already reset()" << endl;
	if (ptr4 == nullptr) {
		cout << "after reset(), ptr4 is empty" << endl;
	}

	//unique_ptr对象不可复制， 智能移动
	//可通过move转移unique_ptr的对象
	//通过原始指针创建ptr5
	unique_ptr<Task> ptr5(new Task(5));
	//把ptr5中关联指针的所有权交给ptr55
	unique_ptr<Task> ptr55(move(ptr5));
	//或者写成  unique_ptr<Task> ptr55 = move(ptr5);
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
====================================unique_ptr的参数
unique_ptr的第一个参数为指针数据类型，第二个参数为该指针自定义的析构器
function指示回调函数(即析构函数)，然后在指针初始化时指定具体的析构函数
*/
//直接使用lambda表达式作为析构函数

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

//对于自定义的数据结构和析构，在创建该类对象的unique_ptr指针时，也可定义其析构函数

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
struct cur* headers = NULL;   //裸指针headers
unique_ptr<struct cur, function<void(struct cur*)>>headerPtr(
	headers, [](struct cur* p) {
	if (p != nullptr) {
		cur_free(p);  //定义析构函数
	}
});

//headerPtr = make_unique<cur>(5);
//cout << headerPtr->curid << endl;


auto delcur = [](struct cur* pHeaders) {
	if (pHeaders != nullptr) {
		cur_free(pHeaders);    //libcurl的析构函数
	}
};
unique_ptr<struct curl_slist, decltype(delcur)> headersPtr;

*/


//==================================explicit 防止隐式调用
/*
class Test1
{
public:
	Test1(int n)
	{
		num = n;
	}//普通构造函数
private:
	int num;
};
class Test2
{
public:
	explicit Test2(int n)      //<------对于单变量的构造函数，尽量加explicit，防止隐式调用
	{
		num = n;
	}//explicit(显式)构造函数
private:
	int num;
};

int main() {
	Test1 t1 = 2;   //<----隐式调用成功
	//Test2 t2 = 3;  compile error    explicit防止了隐式调用
	Test2 t3(3);   //显式调用
}
*/


//==================================vector的一些不一样的构造方法
/*
constructor

// default(1)  默认构造函数
explicit vector(const allocator_type&  alloc = allocator_type());


//fill(2)  构建大小为n的vector，每个元素赋值为val
explicit vector(size_type n, const value_type& val = value_type(),const allocator_type& alloc = allocator_type());


//fill(3)  传入两个迭代器对象（或为指针），将二者的内容拷贝到vector（拷贝前会构造对应大小的容器）
template<class InputIterator>
vector(InputIterator first, InputIterator last, const allocator_type& alloc = alloctor_type());


//copy(4)  传入vector对象，进行拷贝
vector(const vector& x);


//vector内部有自己的allocator，能够实现动态内存的分配与释放，一般不会直接使用new和delete，这样使得内存分配与释放更加安全
vector& operator= (const vector& x);

*/


/*
vector可直接进行大小比较
	vector<int> a{ 1,1,1,0,0 };
	vector<int> b{ 1,1,1,1,0 };
	cout << (a > b) << endl;      <--- 输出 0
	cout << (a < b) << endl;      <--- 输出 1

	若a为{2,1,1,0,0}
	则a>b输出1
*/


//部分排序 partial_sort  和  第n个元素 nth_element
/*
partial_sort的两个重载：
	partial_sort(排序的起始位置，排序的结束位置，查找的结束位置)
	partial_sort(排序的起始位置，排序的结束位置，查找的结束位置，自定义的排序方法)

bool func(int a, int b){
	return a>b;  //降序
}
partial_srot(vec.begin(),vec.begin()+5,vec.end(),func)

partial_sort(vec.begin(), vec.begin() + 4, vec.end(), [](const auto& a, const auto& b) {
		return a > b;
	});


nth_elemen()貌似就是快排
*/











