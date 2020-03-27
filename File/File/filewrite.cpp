#include<iostream>
#include<fstream>

using namespace std;



void test01() {
	//1、包含头文件fstream
	//2、创建流对象
	ofstream ofs1;
	//3、指定打开方式
	ofs1.open("test.txt", ios::out);
	//4、写内容
	ofs1 << "姓名：张三" << endl;
	ofs1 << "性别：男" << endl;
	ofs1 << "年龄：18" << endl;
	//5、关闭文件
	ofs1.close();
}

//读文件
void test02() {
	//1、包含头文件fstream
	//2、创建流对象
	ifstream ifs1;
	ifs1.open("test.txt", ios::in);
	if (! ifs1.is_open()) {
		cout << "文件打开失败" << endl;
		return;
	}

	//第一种:
	char buf[1024];
	while (ifs1 >> buf) {
		
		cout << buf << endl;
	}
	
	//第二种:
/*	char buf[1024] = { 0 };
	while (ifs1.getline(buf, sizeof(buf))) {
		cout << buf << endl;

		//getline (char* s, streamsize n, char delim );
		//getline (char* s, streamsize n );
		//从istream中读取至多n个字符(包含结束标记符)保存在s对应的数组中。即使还没读够n个字符，如果遇到delim 或 字数达到限制，则读取终止
		
	}*/

	//第四种
/*	char c;//不推荐第四种 因为char类型一个个读很慢
	while ((c = ifs1.get()) != EOF) {
		cout << c ;
		//EOF   end of file
	}*/



	ifs1.close();
}


int main() {
	//test01();
	test02();
	system("pause");
	return 0;
}