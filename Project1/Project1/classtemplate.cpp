// classtemplate.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
//

#include<iostream>
#include  "mystack.h"
#include "myAdd.h"      ////<------------����2
using namespace std;


//template<typename T>//template<class T>
//template<typename T, int max_size>

//��������ֿ������ܰ�ģ�庯��д��Դ�ļ���Ӧ��д��ͷ�ļ�
//ʹ��ģ����ʱ,�������ֱ��д�����������ͷ�ļ���������


//�����ڴ�Դ�ļ���ʹ��testAdd().cpp�е�add()����������������

//1��extern int add(int a , int b);    <----���Ƽ�,�����ر��ʱ��һ�����������鷳
//2������һ��ͷ�ļ���Ȼ���add()������Դ�ļ����������ͷ�ļ�


//extern int add(int a, int b);  ����һд��
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