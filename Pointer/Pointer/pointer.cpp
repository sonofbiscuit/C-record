#include<iostream>

using namespace std;

int main() {

	cout << "��ָͬ���С��" << endl;
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

	//arr����������ʾһ��ָ��
	cout << "arr:" << arr << "\t&arr[0]:" << &arr[0] << endl;
	for (int* p = arr; p<&arr[5] ; p++) {//ѭ���������԰�(p-arr)<5��λp<&arr[5]
		cout << *p << endl;
	}


	//nullptr����������ʾһ���հ�ָ��
	int* p = nullptr;//���������ʼ������ô*p����Ϊ����ֵ
}