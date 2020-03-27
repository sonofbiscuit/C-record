#include <string.h>
#include <iostream>

using namespace std;



//int main() {
void func() {
	//C语言的string相当于"\0"结束的数组
	char str1[200] = { 0 };
	char str2[200] = { 0 };

	sprintf_s(str1, "hello");//sprintf为字符串初始化 
	cout << "str1 initial: " << str1 << endl;

	strcpy_s(str2, str1);//strcpy(char* strDestination , char* strSource)
	cout << "str2:" << str2 << endl;
	//springf_s和strcpy_s更安全
	cout << "str1 length:" << strlen(str1) << endl;
	//sizeof(str1)输出的是整个数组的大小
	cout << "str1 array:" << sizeof(str1) << endl;
	//strcmp
	strcat_s(str1, "world");

	if (0 == strcmp(str1, str2)) {
		cout << "same" << endl;
	}
	else {
		cout << "not same" << endl;
	}

	sprintf_s(str1, "123");
	sprintf_s(str2, "123.5");
	//atoi将字符串转换为整数
	cout << "atoi(): " << atoi(str1) + 1 << endl;
	//atof将字符串转换为double
	cout << "atof(): " << atof(str2) + 1 << endl;
}
//}