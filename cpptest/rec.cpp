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





push_back()函数在向容器末尾添加新元素时，会先创建该元素，然后再将该元素移动或者拷贝到容器中；
emplace_back()函数的底层实现是直接在容器尾部创建该新元素，不存在拷贝或者移动元素的过程。
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
	匹配abbc ~ abbbbbc   =>>>>>b的数量为2~5
   ab{2,}c
	匹配ab...c中  b出现次数大于2的

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


//[captures] (params) mutable-> type{...} //lambda 表达式的完整形式
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
*/


