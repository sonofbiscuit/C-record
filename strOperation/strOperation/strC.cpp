#include <string.h>
#include <iostream>

using namespace std;



//int main() {
void func() {
	//C���Ե�string�൱��"\0"����������
	char str1[200] = { 0 };
	char str2[200] = { 0 };

	sprintf_s(str1, "hello");//sprintfΪ�ַ�����ʼ�� 
	cout << "str1 initial: " << str1 << endl;

	strcpy_s(str2, str1);//strcpy(char* strDestination , char* strSource)
	cout << "str2:" << str2 << endl;
	//springf_s��strcpy_s����ȫ
	cout << "str1 length:" << strlen(str1) << endl;
	//sizeof(str1)���������������Ĵ�С
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
	//atoi���ַ���ת��Ϊ����
	cout << "atoi(): " << atoi(str1) + 1 << endl;
	//atof���ַ���ת��Ϊdouble
	cout << "atof(): " << atof(str2) + 1 << endl;
}
//}