#include <iostream>
#include<string>

using namespace std;


class Animal {
public://����
	int length_;
	float height_;
	string name_;
	void breath() {
		cout << name_ << " is breathing" << endl;
	}

	int c;

private://˽��     �����˽�в��ɱ��̳�
	int a;
	int type_;

protected://���ɱ��ⲿʹ�ã����ǿ��Ա�����ʹ��   <-----������Ķ���ֻ�������������������
	int phone_;

};

/*
��Ա��������				�Ƿ��ܹ��������߷���				�Ƿ��ܹ����̳�			
public								true										true
private								false										false
protected							false										true
*/




//�����public�ŵ����൱����Ϊpublic��protected�ŵ���������Ϊprotected�������class Dog : private Animal   ��ô��������м̳г�Ա������Ϊprotected
//public���Ϊprotected ��protected��ȻΪprotected ��private�����Ϊprotected
            //:�̳з�ʽ   ��������             �̳з�ʽ��ָ�����Ա���̳е������Ժ�ķ���Ȩ�ޣ����԰�public��Ϊprivate��protected  �����ܷ��ű任
class Dog : public Animal {//�̳�Animal
public:
	void bark() {//Dog�е�����
		cout << name_ << " is barking" << endl;
		//cout << type_ << endl;//Animal��private�޷����������
		cout << phone_ << endl;																	
	}
};

class Fish : public Animal {//�̳�Animal
public:
	void diving() {
		cout << name_ << " is diving" << endl;
	}
};

int main() {
	Dog wangcai;
	wangcai.name_ = "����";
	Fish marry;
	marry.name_ = "����";
	wangcai.breath();
	marry.breath();
	wangcai.bark();
	marry.diving();


	//wangcai.type_=10;//Animal��private�޷����̳�

	Animal animal;
	//b.a;//�޷�����private
	animal.c;//���Է���public


}