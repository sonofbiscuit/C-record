#include <iostream>
#include "Sort.h"
using namespace std;
/*
交换排序
	冒泡排序
	快速排序
*/

void Sort::BubbleSort(int arr[], int n) {
	bool flag;
	for (int i = 0; i < n-1; i++) {
		flag = false;
		for (int j = 0; j < n - i - 1; j++) {
			if (arr[j] > arr[j + 1]) {
				swap(arr[j], arr[j + 1]);
				flag = true;
			}
		}
		if (flag = false) {
			return;
		}
	}
}


int Sort::Partion(int arr[], int low, int high) {
	int pivot = arr[low];
	int i = low, j = high;
	while (i < j) {
		while (arr[j] >= pivot&&i<j) {//从右找第一个比pivot小的
			j--;
		}
		while (arr[i] <= pivot&&i<j) {//从左找第一个比pivot大的
			i++;	
		}
		if (i < j) {
			swap(arr[i], arr[j]);
		}
	}
	swap(arr[j], arr[low]);
	return j;
}

void Sort::QuickSort(int arr[], int low, int high) {
	if (low < high) {
		int pivot = Partion(arr, low, high);
		QuickSort(arr, low, pivot-1);
		QuickSort(arr, pivot + 1, high);
	}
}

/*
快排从右往左找
*/