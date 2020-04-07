#include <iostream>
#include "Sort.h"
using namespace std;

/*
��������
	ֱ�Ӳ�������
	�۰��������
	ϣ������
*/

void Sort::DInsertSort(int arr[], int n) {
	cout << "ԭ����Ϊ: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;

	for (int i = 1; i < n; i++) {
		if (arr[i - 1] > arr[i]) {//��������
			int temp = arr[i];
			int j = 0;
			for (j = i - 1; j >= 0 && arr[j] > temp; j--) {
				arr[j + 1] = arr[j];
			}
			arr[j + 1] = temp;
		}
	}
	cout << "ֱ�Ӳ�������: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;
}

void Sort::HInsertSort(int arr[], int n) {//����
	for (int i = 1; i < n - 1; i++) {
		int left = i, right = n - 1;
		int temp = arr[i];
		while (left <= right) {
			int mid = (left + right) / 2;
			if (arr[mid] > temp) {
				right = mid - 1;
			}
			else {
				left = mid + 1;
			}
		}
		for (int j = right + 1; j < i; j++) {
			arr[j + 1] = arr[j];
		}
		arr[right] = temp;
	}
	cout << "�۰��������: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;
}

void Sort::SHellSort(int arr[], int n) {
	for (int dk = n / 2; dk >= 1; dk = dk / 2) {//�����仯
		for (int i = dk; i < n; i++) {
			int temp = arr[i];
			int j = 0;
			for (j = i - dk; j >= 0 && temp < arr[j]; j -= dk) {//����  tempΪarr[i]  arr[j]Ϊarr[i-dk]
				arr[j + dk] = arr[j];//����
			}
			arr[j + dk] = temp;
		}
	}
	cout << "ϣ������: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;
}

