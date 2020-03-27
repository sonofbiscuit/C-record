#include<iostream>

using namespace std;


//���캯��
//һ�����󱻴�����ʱ����ִ�еĺ���

//Animal����  �ƶ� ���� ����
class Animal {
public:
	Animal() {
		cout << "Animal console " << endl;
	}

	virtual void move() = 0;//<-----���麯������һ���൱�߼��ı�﷽ʽ ,û��Ĭ��ʵ�֣���Ҫ��������ʵ��                   
	//���virtual���Ժ��дAnimal�������ʱ��֪��Ӧ�ø�����Щ��������ߴ���Ŀ�ά����

	virtual void move1() {
		cout << "�麯��move1��Ĭ��ʵ��" << endl;
	}
	//�麯���������������и��ǣ���˿��Բ�ʵ�֣�д������=0����ʱ�������麯����

	virtual void grow() = 0;

	virtual void die() = 0;
};


class Dog : public Animal {
public:
	Dog() {
		cout << "Dog console" << endl;
	}

	void run(){
		cout << "Dog is running" << endl;
	}
	void move() override {     //override��ʾ���麯����������д
		run();
	}

	//���麯����д
	void move1() override{
		cout << "move1 override " << endl;
	}


	void grow() override {
		cout << "Dog is growing" << endl;
	}
	
	void die() override {
		cout << "Dog is dead" << endl;
	}
};


class Fish : public Animal {
public:
	Fish() {
		cout << "Fish console" << endl;
	}

	void run() {
		cout << "Fish is running" << endl;
	}
	void move() override {
		run();
	}

	void grow() override {
		cout << "Fish is growing" << endl;
	}

	void die() override {
		cout << "Fish is dead" << endl;
	}
};


int main() {
	Dog wangcai;
	wangcai.run();
	wangcai.grow();
	wangcai.die();
	wangcai.move1();
	cout << "-------------" << endl;
	Fish marry;
	marry.run();
	marry.grow();
	marry.die();
	marry.move1();
}