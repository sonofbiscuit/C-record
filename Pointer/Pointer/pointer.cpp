#include<iostream>

using namespace std;

int main() {

	cout << "不同指针大小：" << endl;
	int* pInt = nullptr;
	char* pChar = nullptr;
	short* pShort = nullptr;

	cout << sizeof(pInt) << endl;
	cout << sizeof(pChar) << endl;
	cout << sizeof(pShort) << endl;

	cout << "--------------------" << endl;
	int arr[5] = { 1,3,5,7,9 };

	for (int i = 0; i < 5; i++) {
		cout << arr[i] << endl;
	}
	cout << "-----------------" << endl;

	//arr可以用来表示一个指针
	cout << "arr:" << arr << "\t&arr[0]:" << &arr[0] << endl;
	for (int* p = arr; p<&arr[5] ; p++) {//循环条件可以把(p-arr)<5改位p<&arr[5]
		cout << *p << endl;
	}


	//nullptr可以用来表示一个空白指针
	int* p = nullptr;//如果不做初始化，那么*p可能为任意值
}