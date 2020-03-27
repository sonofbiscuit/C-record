#include<iostream>
#include<string>

using namespace std;

int main() {
	string str1 = "abc";
	for (string::iterator it = str1.begin(); it != str1.end(); it++) {
		cout << *it << "|" ;
	}
	cout << endl << "--------------" << endl;
	cout << "str1[1]: " << str1[1] << endl;
	cout << "str1.at(1): " << str1.at(1) << endl;
	//.at  <-------get character in string

	cout << "str1 after append: " << str1.append("123") << endl;
	//find函数返回下标
	//string1.compare(string2)    判断相等

	char str2[100] = {0};
	//string_character.copy(p , n , size_type_off = 0)
	//从string类型对象中拷贝n个字符到字符指针p指向的空间中,默认从0开始
	//p指向的空间容量要足够保存n个字符
	str1.copy(str2, sizeof(str1));//string类型str1中拷贝所有元素到str2中
	cout << "str2：" << str2 << endl;


	char str3[100] = { 0 };
	strcpy_s(str3, str1.c_str());//c_str()返回一个以0结尾的字符串
	cout << "str3: " << str3 << endl;
	cout << "strlen(str1) " << strlen(str1.c_str()) << endl;

}