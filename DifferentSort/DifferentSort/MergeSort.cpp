#include <iostream>
#include "Sort.h"
using namespace std;

/*
�鲢����
	�鲢����
	��������
*/

//�鲢������ڷ���˼��
void Sort::MergeSort(int arr[], int low, int high) {
	if (low >= high) { return; } // ��ֹ�ݹ�������������г���Ϊ1
	int mid = (low + high) / 2;  // ȡ�������м��Ԫ��
	MergeSort(arr, low, mid);  // �����ߵݹ�
	MergeSort(arr, mid + 1, high);  // ���Ұ�ߵݹ�
	Merge(arr, low, mid, high);  // �ϲ�
}

void Sort::Merge(int arr[], int low, int mid, int high) {
	//lowΪ��1�������ĵ�1��Ԫ�أ�iָ���1��Ԫ��, midΪ��1�����������1��Ԫ��
	int i = low, j = mid + 1, k = 0;  //mid+1Ϊ��2��������1��Ԫ�أ�jָ���1��Ԫ��
	int *temp = new int[high - low + 1]; //temp�����ݴ�ϲ�����������
	while (i <= mid && j <= high) {
		if (arr[i] <= arr[j]) //��С���ȴ���temp��
			temp[k++] = arr[i++];
		else
			temp[k++] = arr[j++];
	}
	while (i <= mid)//���Ƚ���֮�󣬵�һ������������ʣ�࣬��ֱ�Ӹ��Ƶ�t������
		temp[k++] = arr[i++];
	while (j <= high)//ͬ��
		temp[k++] = arr[j++];
	for (i = low, k = 0; i <= high; i++, k++)//���ź���Ĵ��arr��low��high������
		arr[i] = temp[k];
	delete[]temp;//�ͷ��ڴ棬����ָ��������飬������delete []
}


//<------------------------------------->
//��������

//�����ݵ����λ��
int Sort::maxBit(int arr[], int n) {
	int d = 1;//�������λ��
	int p = 10;
	for (int i = 0; i < n; i++) {
		while (arr[i] > p) {
			d++;
			p *= 10;
		}
	}
	return d;
}

void Sort::RadixSort(int arr[], int n) {
	int d = maxBit(arr, n);
	int radix = 1;
	int *temp = new int[n];//��ʱ����
	int count[10];//�������
	int i, j, k;
	for (i = 1; i <= d; i++) {
		for (j = 0; j < 10; j++) {
			count[j] = 0;//��ձ������
		}
		for (j = 0; j < n; j++) {
			k = (arr[j] / radix) % 10;//ȡλ
			count[k]++;//��¼ÿ��λ�ε����ָ���
		}
		for (j = 1; j < 10; j++)
			count[j] = count[j - 1] + count[j]; //��ÿ��������λ��
		for (j = n - 1; j >= 0; j--) //������Ͱ�м�¼�����ռ���tmp��
		{
			k = (arr[j] / radix) % 10;
			temp[count[k] - 1] = arr[j];
			count[k]--;
		}
		for (j = 0; j < n; j++) //����ʱ��������ݸ��Ƶ�data��
			arr[j] = temp[j];
		radix = radix * 10;//ȡ��һλ
	}	
}




/*
arr:73 22 93 43 55 14 28 65 39 81
//ȡ��λ;
					count[10]       count[i]=count[i]+count[i-1]///i=1~d
0                      0                       0
1  81                  1                       1
2  22                  1                       2
3  73 93 43            3                       5
4  14                  1                       6
5  55 65               2                       8
6                      0                       8
7                      0                       8
8  28                  1                       9
9  39                  1                       10

					  temp[n]����
0                      81
1                      22
2                      73
3                      93
4                      43
5                      14
6                      55
7                      65
8                      28
9                      39

��temp���ݴ���arr
arr:81 22 73 93 43 14 55 65 28 39
//ȡʮλ;
					 count[10]       count[i]=count[i]+count[i-1]
0  0                    0                     0
1  14                   1                     1
2  22 28                2                     3
3  39                   1                     4
4  43                   1                     5
5  55                   1                     6
6  65                   1                     7
7  73                   1                     8
8  81                   1                     9
9  93                   1                     10


					  temp[n]����
0                       14       count[1]=1-1=0
1                       22       count[2]=3-1-1
2                       28       count[2]=3-1
3                       39       count[3]=4-1
4                       43       count[4]=5-1=4
5                       55       count[5]=6-1=5
6                       65       count[6]=7-1=6
7                       73       count[7]=8-1=7
8                       81       count[8]=9-1=8
9                       93       count[9]=10-1=9

*/