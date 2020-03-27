#include <iostream>

using namespace std;


//this
//	它是一个指针
//	它定义在成员函数(非静态)中
//	指向当前的对象
class	Human {
public :
	void set_age(int age) {
		this->age_ = age;
	}
	int get_age() {
		return this->age_;
	}

	Human* get_this() {//返回一个Human类型的指针
		return this;
	}
	//静态成员函数没有this指针

	//若非要使用，则添加一个指针参数，利用指针来访问非静态成员
	static int static_get_this(Human* obj) {
		return obj->age_;
	}

private:
	int age_;
};




int main() {
	Human jack, alice;
	jack.set_age(25);
	alice.set_age(21);
	cout << jack.get_age() << endl;
	cout << alice.get_age()<< endl;

	cout << "&jack" << &jack << "\tthis:" << jack.get_this() << endl;//说明this指针是指向对象本身的
	cout << "&alice" << &alice << "\tthis:" << alice.get_this() << endl;

	cout << "jack's age: " << jack.static_get_this(&jack) << endl;
}