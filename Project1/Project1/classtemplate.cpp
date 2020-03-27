// classtemplate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<iostream>
#include  "mystack.h"
#include "myAdd.h"      ////<------------方法2
using namespace std;


//template<typename T>//template<class T>
//template<typename T, int max_size>

//函数和类分开，不能把模板函数写到源文件，应该写到头文件
//使用模板类时,定义可以直接写到类里，以免在头文件中再声明


//若想在此源文件中使用testAdd().cpp中的add()函数，方法有两种

//1、extern int add(int a , int b);    <----不推荐,函数特别多时，一个个声明很麻烦
//2、建立一个头文件，然后把add()函数的源文件包含进这个头文件


//extern int add(int a, int b);  方法一写法
int main() {
	cout << "result: " << add(1, 2) << endl;   



	Stack<int, 10> stack;//Stack<int , max_size>
	stack.push(1);
	stack.push(2);
	stack.push(3);
	stack.push(4);
	while (stack.size()) {
		cout << stack.top() << endl;
		stack.pop();
	}
	system("pause");
}