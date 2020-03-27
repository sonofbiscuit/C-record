#include<iostream>

using namespace std;


//构造函数
//一个对象被创建的时候所执行的函数

//Animal共性  移动 长大 死亡
class Animal {
public:
	Animal() {
		cout << "Animal console " << endl;
	}

	virtual void move() = 0;//<-----纯虚函数，是一种相当高级的表达方式 ,没有默认实现，需要在子类中实现                   
	//标记virtual让以后编写Animal的子类的时候知道应该覆盖哪些函数，提高代码的可维护性

	virtual void move1() {
		cout << "虚函数move1的默认实现" << endl;
	}
	//虚函数允许子类对其进行覆盖，因此可以不实现，写法令其=0，此时叫做纯虚函数；

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
	void move() override {     //override表示对虚函数进行了重写
		run();
	}

	//对虚函数重写
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