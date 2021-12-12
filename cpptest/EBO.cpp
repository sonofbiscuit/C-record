#include <iostream>

using namespace std;

class A {};

class B1 {
private:
	A a;
};

class B2 {
private:
	A a;
	int x;
};

class C:private A {
private:
	int x;
};
//sizeof(A) == 1
//sizeof(B1) == 1
//sizeof(B2) == 8
//sizeof(C) == 5

class Empty {};

class Empty1 : public Empty {};

class Empty2 : public Empty1 {};

class Empty3 : public Empty,Empty1 {};
//sizeof(Empty) == 1
//sizeof(Empty1) == 1
//sizeof(Empty2) == 1
//sizeof(Empty3) == 1


class Empty11 {
public:
	void p11() {
		std::cout << "&Empty11= " << this << endl;
	}
};

class Empty12:public Empty11 {
public:
	void p12() {
		std::cout << "&Empty12= " << this << endl;
	}
};

class Empty13:public Empty11, public Empty12 {
public:
	int x;
	void p13() {
		std::cout << "&Empty13= " << this << endl;
	}

	void p13_12() {
		std::cout << "Empty13::Empty12::p12 = "; Empty12::p12();
	}

	void p13_11(){
		std::cout << "Empty13::Empty11::p11 = "; Empty11::p11();
	}

	void p13_12_11() {
		std::cout << "Empty13::Empty12::p11 = "; Empty12::p11();
	}
};

int main() {

	//cout << sizeof(A) << endl;
	//cout << sizeof(B1) << endl;
	//cout << sizeof(B2) << endl;
	//cout << sizeof(C) << endl;


	//cout << sizeof(Empty) << endl;
	//cout << sizeof(Empty1) << endl;
	//cout << sizeof(Empty2) << endl;
	//cout << sizeof(Empty3) << endl;


	Empty13 e13;
	e13.p13();
	e13.p13_12();
	e13.p13_11();
	e13.p13_12_11();

	cout << sizeof(Empty11) << endl;
	cout << sizeof(Empty12) << endl;
	cout << sizeof(Empty13) << endl;

	return 0;
}