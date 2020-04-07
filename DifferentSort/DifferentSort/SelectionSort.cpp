#include <iostream>
#include "Sort.h"
using namespace std;

/*
ѡ������
	��ѡ������
	������
*/

void Sort::SelectSort(int arr[], int n) {
	for (int i = 0; i < n; i++) {
		int temp = i;
		for (int j = i + 1; j < n; j++) {
			if (arr[j] < arr[temp]) {
				temp = j;
			}
		}
		if (temp != i) {
			swap(arr[i], arr[temp]);
		}
	}
}



void Sort::HeapSort(int arr[], int len) {//���������������
	BuildMaxHeap(arr, len);
	cout << "��Ϊ��";
	for (int i = 0; i < len; i++) {
		cout << arr[i] << " ";//�ѵ���״
	}
	cout << endl << "�󶥶����: ";
	for (int i = len - 1; i > 0; i--) {
		Swap(arr, 0, i);//��������͵����һ������
		AdjustDown(arr, 0, i-1);
		cout << arr[i] << " ";
	}
}

void Sort::BuildMaxHeap(int arr[], int len) {//���������,len��Ϊn
	for (int i = len / 2 - 1; i >= 0; i--) {//��len/2��1������������
		AdjustDown(arr, i, len);
	}
}


//����󶥶�ʱ��ɾ������Ԫ�أ������һ��Ԫ�ؽ���
void Sort::AdjustDown(int arr[], int k, int len) {//�����Ϊk��Ԫ�����µ���
	int temp = arr[k];//arr[0]�ݴ�
	for (int i = 2 * k + 1; i < len; i = i * 2 + 1) {//��k�ķ�֧���������Ϊ�±��0��ʼ����λ���Ҫ*2+1
		if (i < len&&arr[i] < arr[i + 1]) {
			i++;//ȡֵ�ϴ�Ľ����±�
		}
		if (temp >= arr[i]) {
			break;
		}
		else {
			arr[k] = arr[i];//���ֵ���ƣ���A[i]������˫�׽��
			k = i;//��׼������һ����֧
		}
	}//for
	arr[k] = temp;
}

//����Ѳ���Ԫ��ʱ���Ƚ�������Ԫ�ط���ѵ�ĩβ���������ϵ���
void Sort::AdjustUp(int arr[], int k) {//���Ϊk���������ϵ���
	int temp = arr[k];//�ݴ�
	for (int i = (k - 1) / 2; i > 0; i = (i - 1) / 2) {
		if (i > 0 && arr[i] < temp) {
			arr[k] = arr[i];//���ڵ�����
			k = i;
		}
	}
	arr[k] = temp;
}

void Sort::Swap(int arr[], int x, int y) {
	int temp;
	temp = arr[x];
	arr[x] = arr[y];
	arr[y] = temp;
}