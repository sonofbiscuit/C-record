//explicit=>
//implicit calls to constructors are prohibited


class Test1{
private:
	int num;
public:
	Test1(int n){
		num = n;
	}
};

class Test2{
private:
	int num;
public:
	explicit Test2(int n){
		num = n;
	} //explicit 显式构造函数
};

int main(){
	Test1 t1=10; //true
	Test2 t2=10; // <==error, implicit calls to constructor is prohibited
	Test2 t2(10); //true
}
