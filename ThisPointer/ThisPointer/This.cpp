#include <iostream>

using namespace std;


//this
//	����һ��ָ��
//	�������ڳ�Ա����(�Ǿ�̬)��
//	ָ��ǰ�Ķ���
class	Human {
public :
	void set_age(int age) {
		this->age_ = age;
	}
	int get_age() {
		return this->age_;
	}

	Human* get_this() {//����һ��Human���͵�ָ��
		return this;
	}
	//��̬��Ա����û��thisָ��

	//����Ҫʹ�ã������һ��ָ�����������ָ�������ʷǾ�̬��Ա
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

	cout << "&jack" << &jack << "\tthis:" << jack.get_this() << endl;//˵��thisָ����ָ��������
	cout << "&alice" << &alice << "\tthis:" << alice.get_this() << endl;

	cout << "jack's age: " << jack.static_get_this(&jack) << endl;
}