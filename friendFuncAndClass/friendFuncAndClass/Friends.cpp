#include <iostream>
#include <string>
using namespace std;

//��Ԫ�������ɱ��̳кʹ���
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

	//get/setҲ������Ϊ�ӿڷ��ʳ�Ա������private
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
	
	//��Ԫ����������
	//friend void Marry(Human human1, Human human2);
	friend class minzhengju;//��Ԫ��  ��minzhengju�ɷ���private��name_

private:
	string name_;	
protected:
	int age_;
	bool gender_;
};

//����Marry��װ��minzhengju�����
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
	lihua.set_name("�");
	han.set_name("÷");
	minzhengju mzj;
	mzj.Marry(lihua, han);
	return 0;
}