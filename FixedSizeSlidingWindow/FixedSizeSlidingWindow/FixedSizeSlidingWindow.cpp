#include <iostream>
#include <stack>
#include <string>
#include <queue>
#include <algorithm>

using namespace std;


/*
ÿ���ж�������Դ��еĿ�ͷ����ĩβ��һ�ſ��ƣ���������������� k �ſ��ơ�
��ĵ����������õ����е����п��Ƶĵ���֮��
���룺cardPoints = [1,2,3,4,5,6,1], k = 3
�����12
���ͣ���һ���ж��������������ƣ���ĵ������� 1 ��
���ǣ��������ұߵĿ��ƽ��������Ŀɻ�õ��������Ų��������ұߵ������ƣ����յ���Ϊ 1 + 6 + 5 = 12 ��

*/

//forѭ����Ƕ�׻ᵼ��ʱ�临�Ӷȵ�����
/*
int maxScore(vector<int>& cardPoints, int k) {
	int n = cardPoints.size();
	int windowsize = n - k;
	int ans = INT_MAX;
	int sum = 0;
	for (int left = 0, right = left + windowsize - 1; left <= n - windowsize && right < n; left++, right++) {
		int temp = 0;
		int l = left, r = right;
		for (l; l <= r; l++) {
			temp += cardPoints[l];
			cout << temp << endl;
		}
		ans = min(ans, temp);
		temp = 0;
	}
	for (auto c : cardPoints) {
		sum += c;
	}
	cout << endl;
	cout << sum - ans;
	return sum - ans;
}
*/

int maxScore(vector<int>& cardPoints, int k) {
	int n = cardPoints.size();
	int m = 0;
	for (auto a : cardPoints) {
		m += a;
	}
	int windowsize = n - k;
	int ans = INT_MAX;
	int sum = 0;
	int temp=0;
	for (int i = 0; i < windowsize; i++) {
		sum += cardPoints[i];//���һ�����ڵĺ�
	}
	temp = sum;
	for (int i = 1; i < n - windowsize + 1; i++) {
		sum = sum + cardPoints[windowsize - 1 + i] - cardPoints[i-1];//������������ƶ���ÿ�μӺ���һ������ȥǰ��һ��
		cout << cardPoints[windowsize - 1 + i] << "   " << cardPoints[i - 1] << endl;
		temp = min(temp, sum);//�洢�ù����е���Сֵ
	}
	return m - temp;
}



int main() {
	vector<int> card = { 1,79,80,1,1,1,200,1 };
	maxScore(card, 3);
}
