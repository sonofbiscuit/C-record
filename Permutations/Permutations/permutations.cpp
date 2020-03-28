#include <iostream>
#include <vector>

using namespace std;

void swap(int a, int b, vector<char> &vec) {
	//cout << vec[a] <<"  "<< vec[b] << endl;
	char temp = vec[a];
	vec[a] = vec[b];
	vec[b] = temp;
	//cout << vec[a] << " " << vec[b] << endl;
}

void permutate(int begin, int end, vector<char> vec) {
	if (begin == end) {
		for (int i = 0; i < vec.size();i++) {
			cout << vec[i];
		}
		cout << endl;
	}

	for (int j = begin; j < end; j++) {
		swap(begin, j, vec);
		permutate(begin+1, end, vec);
		swap(j, begin, vec);
	}
}

int main() {
	vector<char> vec{ '(','(','(',')',')',')' };
	int n = vec.size();
	permutate(0, n, vec);
	return 0;
}