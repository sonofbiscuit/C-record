#include <iostream>
#include <string>
using namespace std;

//友元函数不可被继承和传递
class Human {
public:
	Human(int age, bool gender)
		:age_(age)
		, gender_(gender) {}

	void walk() {
		cout << name_ << "is walking" << endl;
	}
	void say(string content) {
		cout << name_ << "is saying" << content << endl;
	}

	//get/set也可以作为接口访问成员，包括private
	void set_name(string name) {
		name_ = name;
	}
	string set_name() {
		return name_;
	}
	int get_age() {
		return age_;
	}
	bool get_gender() {
		return gender_;
	}
	
	//友元函数的声明
	//friend void Marry(Human human1, Human human2);
	friend class minzhengju;//友元类  则minzhengju可访问private的name_

private:
	string name_;	
protected:
	int age_;
	bool gender_;
};

//若将Marry封装到minzhengju这个类
class minzhengju {
public:
	void Marry(Human human1, Human human2) {
		cout << human1.name_ 
			<< "  and  " 
			<< human2.name_ 
			<< "  get married  " 
			<< endl;
	}
};




int main() {
	Human lihua(25, true), han(25, false);
	lihua.set_name("李华");
	han.set_name("梅");
	minzhengju mzj;
	mzj.Marry(lihua, han);
	return 0;
}