#include <iostream>
#include "Sort.h"
using namespace std;

/*
插入排序
	直接插入排序
	折半插入排序
	希尔排序
*/

void Sort::DInsertSort(int arr[], int n) {
	cout << "原数组为: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;

	for (int i = 1; i < n; i++) {
		if (arr[i - 1] > arr[i]) {//升序排列
			int temp = arr[i];
			int j = 0;
			for (j = i - 1; j >= 0 && arr[j] > temp; j--) {
				arr[j + 1] = arr[j];
			}
			arr[j + 1] = temp;
		}
	}
	cout << "直接插入排序: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;
}

void Sort::HInsertSort(int arr[], int n) {//升序
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
	cout << "折半插入排序: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;
}

void Sort::SHellSort(int arr[], int n) {
	for (int dk = n / 2; dk >= 1; dk = dk / 2) {//步长变化
		for (int i = dk; i < n; i++) {
			int temp = arr[i];
			int j = 0;
			for (j = i - dk; j >= 0 && temp < arr[j]; j -= dk) {//升序  temp为arr[i]  arr[j]为arr[i-dk]
				arr[j + dk] = arr[j];//后移
			}
			arr[j + dk] = temp;
		}
	}
	cout << "希尔排序: ";
	for (int i = 0; i < n; i++) {
		cout << arr[i];
	}
	cout << endl;
}

