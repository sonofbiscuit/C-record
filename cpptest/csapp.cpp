#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <typeinfo>
using namespace std;


//��λ��&  ��� ^    �� |
//�߼���&&   �� ||

int isTmax(int x) {  //  ! ���κη�0��ֵ��Ϊ0��0��Ϊ1
	int i = x + 1;
	//xΪmaximum, ��i=1000 0000 ... 0000 ��Ϊminimum
	x = x + i; // two's integer maximum + minimum = -1 => 1111...1111
	x = ~x; //1111 1111...1111 => 0000 0000...0000
	i = !i; //iΪminimum��-2^32, !(-2^32)=0
	x = x + i; // xΪmax, ~(x+i)+!i=0+1=1
	cout << !x << endl;
	return !x;
}

int allOddBits(int x) {  // 1010 1010 1010 1010...1010
	int y = 0xAAAA;
	y = y + (y << 16);
	cout << !((y&x) ^ y) << endl;
	return !((y&x) ^ y);
}

int negate(int x) {
	cout << ~x + 1 << endl;
	return ~x + 1;
}

int isAsciiDigit(int x) { // 0x30 ~ 0x39 => ascii 0 ~ 9
	int min = x + (~0x30 + 1);//����
	//int max = ;
	return 0;
}

/*
int main() {
	//isTmax(4294967295);
	//cout << !(-4294967296) << endl;
	//allOddBits(0xFFFFFFFD);
	//allOddBits(0xAAAAAAAA);
	//negate(0x00000000000000000000000000000001);
	int a = 0x30;
	cout << a << endl;
	return 0;
}
*/



