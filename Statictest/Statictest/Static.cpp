#include <istream>
#include <iostream>

using namespace std;

//			ȫ�ֱ���					    VS										��̬ȫ�ֱ���
//	���÷�Χ   ��������Ψһ											���÷�ΧС		��ǰԴ�ļ�

extern int add(int a, int b);
extern int get_global();

static int global1=30;

//�ֲ���̬����
//�������ڣ��ᴩ��������ִ��
//�����򣺺���ͨ�ֲ�����һ��
void frequency() {
	static int count_ = 0;
	count_++;
	cout << "��" << count_ << "�ε���" << endl;
}

class MyClass {
public:
	int a;
	static int b;
		
	//��̬��Ա����,�;�̬����һ�����Ա�޹�
	static void staticFunction(MyClass* obj) {//<--------------��̬�޷�ֱ�ӷ��ʷǾ�̬ , �ú����޷�ֱ�ӷ���a�����ʵĻ��޸�����  ���MyClass* obj
		cout << obj->a << endl;
		cout << "˽�г�Աvp=" << obj->vp << endl;
	}
	//�ɽ���̬����������Ԫ������������Ϊ����Է������е�˽�г�Ա

	void set_vp(int v) {
		vp = v;
	}

private:
	int vp;
};

int MyClass::b = 70;//��b�ĳ�ʼ��
	
int main() {
	cout <<"global:"<< global1 << endl;
	cout << "add���global:" <<get_global()<< endl;
	frequency();
	frequency();
	frequency();
	frequency();

	MyClass jack, alice;
	cout << "��ʼ����b:" << MyClass::b<<endl;
	alice.a = 10;
	jack.a = 20;
	jack.b = 30;
	cout << "jack��a:" << jack.a << "\t alice��a" << alice.a << endl;
	cout << "jack��b:" << jack.b << "\t alice��b" << alice.b << endl;
	alice.a = 40;
	alice.b = 60;
	cout << "jack��a:" << jack.a << "\t alice��a" << alice.a << endl;
	cout << "jack��b:" << jack.b << "\t alice��b" << alice.b << endl;
	//��̬�ֲ������ͳ�Ա�޹�
	MyClass::b = 100;
	cout << "MyClass::b��" << MyClass::b << endl;

	jack.staticFunction(&jack);//<--------��̬����ͨ�����������ʷǾ�̬����
	MyClass::staticFunction(&jack);//��---------������

	jack.set_vp(100);

}