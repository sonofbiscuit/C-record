#include <istream>
#include <iostream>

using namespace std;

//			全局变量					    VS										静态全局变量
//	作用范围   整个程序唯一											作用范围小		当前源文件

extern int add(int a, int b);
extern int get_global();

static int global1=30;

//局部静态变量
//生命周期：贯穿整个程序执行
//作用域：和普通局部变量一样
void frequency() {
	static int count_ = 0;
	count_++;
	cout << "第" << count_ << "次调用" << endl;
}

class MyClass {
public:
	int a;
	static int b;
		
	//静态成员函数,和静态变量一样与成员无关
	static void staticFunction(MyClass* obj) {//<--------------静态无法直接访问非静态 , 该函数无法直接访问a，访问的话修改如左  添加MyClass* obj
		cout << obj->a << endl;
		cout << "私有成员vp=" << obj->vp << endl;
	}
	//可将静态函数当作友元函数看待，因为其可以访问类中的私有成员

	void set_vp(int v) {
		vp = v;
	}

private:
	int vp;
};

int MyClass::b = 70;//对b的初始化
	
int main() {
	cout <<"global:"<< global1 << endl;
	cout << "add里的global:" <<get_global()<< endl;
	frequency();
	frequency();
	frequency();
	frequency();

	MyClass jack, alice;
	cout << "初始化的b:" << MyClass::b<<endl;
	alice.a = 10;
	jack.a = 20;
	jack.b = 30;
	cout << "jack的a:" << jack.a << "\t alice的a" << alice.a << endl;
	cout << "jack的b:" << jack.b << "\t alice的b" << alice.b << endl;
	alice.a = 40;
	alice.b = 60;
	cout << "jack的a:" << jack.a << "\t alice的a" << alice.a << endl;
	cout << "jack的b:" << jack.b << "\t alice的b" << alice.b << endl;
	//静态局部变量和成员无关
	MyClass::b = 100;
	cout << "MyClass::b：" << MyClass::b << endl;

	jack.staticFunction(&jack);//<--------静态函数通过对象来访问非静态变量
	MyClass::staticFunction(&jack);//《---------方法二

	jack.set_vp(100);

}