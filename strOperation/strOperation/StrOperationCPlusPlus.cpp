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
	//find���������±�
	//string1.compare(string2)    �ж����

	char str2[100] = {0};
	//string_character.copy(p , n , size_type_off = 0)
	//��string���Ͷ����п���n���ַ����ַ�ָ��pָ��Ŀռ���,Ĭ�ϴ�0��ʼ
	//pָ��Ŀռ�����Ҫ�㹻����n���ַ�
	str1.copy(str2, sizeof(str1));//string����str1�п�������Ԫ�ص�str2��
	cout << "str2��" << str2 << endl;


	char str3[100] = { 0 };
	strcpy_s(str3, str1.c_str());//c_str()����һ����0��β���ַ���
	cout << "str3: " << str3 << endl;
	cout << "strlen(str1) " << strlen(str1.c_str()) << endl;

}