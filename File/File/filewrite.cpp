#include<iostream>
#include<fstream>

using namespace std;



void test01() {
	//1������ͷ�ļ�fstream
	//2������������
	ofstream ofs1;
	//3��ָ���򿪷�ʽ
	ofs1.open("test.txt", ios::out);
	//4��д����
	ofs1 << "����������" << endl;
	ofs1 << "�Ա���" << endl;
	ofs1 << "���䣺18" << endl;
	//5���ر��ļ�
	ofs1.close();
}

//���ļ�
void test02() {
	//1������ͷ�ļ�fstream
	//2������������
	ifstream ifs1;
	ifs1.open("test.txt", ios::in);
	if (! ifs1.is_open()) {
		cout << "�ļ���ʧ��" << endl;
		return;
	}

	//��һ��:
	char buf[1024];
	while (ifs1 >> buf) {
		
		cout << buf << endl;
	}
	
	//�ڶ���:
/*	char buf[1024] = { 0 };
	while (ifs1.getline(buf, sizeof(buf))) {
		cout << buf << endl;

		//getline (char* s, streamsize n, char delim );
		//getline (char* s, streamsize n );
		//��istream�ж�ȡ����n���ַ�(����������Ƿ�)������s��Ӧ�������С���ʹ��û����n���ַ����������delim �� �����ﵽ���ƣ����ȡ��ֹ
		
	}*/

	//������
/*	char c;//���Ƽ������� ��Ϊchar����һ����������
	while ((c = ifs1.get()) != EOF) {
		cout << c ;
		//EOF   end of file
	}*/



	ifs1.close();
}


int main() {
	//test01();
	test02();
	system("pause");
	return 0;
}