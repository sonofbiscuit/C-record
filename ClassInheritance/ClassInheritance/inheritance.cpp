#include <iostream>
#include<string>

using namespace std;


class Animal {
public://共有
	int length_;
	float height_;
	string name_;
	void breath() {
		cout << name_ << " is breathing" << endl;
	}

	int c;

private://私有     父类的私有不可被继承
	int a;
	int type_;

protected://不可被外部使用，但是可以被子类使用   <-----比如你的东西只想给儿子玩而不想给外人
	int phone_;

};

/*
成员变量类型				是否能够被调用者访问				是否能够被继承			
public								true										true
private								false										false
protected							false										true
*/




//父类的public放到子类当中作为public，protected放到子类中作为protected，如果是class Dog : private Animal   那么父类的所有继承成员都设置为protected
//public会变为protected ，protected仍然为protected ，private不会变为protected
            //:继承方式   父类名称             继承方式是指父类成员被继承到子类以后的访问权限，可以把public变为private和protected  而不能反着变换
class Dog : public Animal {//继承Animal
public:
	void bark() {//Dog有的属性
		cout << name_ << " is barking" << endl;
		//cout << type_ << endl;//Animal的private无法被子类访问
		cout << phone_ << endl;																	
	}
};

class Fish : public Animal {//继承Animal
public:
	void diving() {
		cout << name_ << " is diving" << endl;
	}
};

int main() {
	Dog wangcai;
	wangcai.name_ = "旺财";
	Fish marry;
	marry.name_ = "玛丽";
	wangcai.breath();
	marry.breath();
	wangcai.bark();
	marry.diving();


	//wangcai.type_=10;//Animal的private无法被继承

	Animal animal;
	//b.a;//无法访问private
	animal.c;//可以访问public


}