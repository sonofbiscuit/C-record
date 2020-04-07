#include <iostream>
#include "Sort.h"
using namespace std;

/*
选择排序
	简单选择排序
	堆排序
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



void Sort::HeapSort(int arr[], int len) {//输出堆排序后的序列
	BuildMaxHeap(arr, len);
	cout << "堆为：";
	for (int i = 0; i < len; i++) {
		cout << arr[i] << " ";//堆的形状
	}
	cout << endl << "大顶堆输出: ";
	for (int i = len - 1; i > 0; i--) {
		Swap(arr, 0, i);//交换顶点和第最后一个数据
		AdjustDown(arr, 0, i-1);
		cout << arr[i] << " ";
	}
}

void Sort::BuildMaxHeap(int arr[], int len) {//建立大根堆,len即为n
	for (int i = len / 2 - 1; i >= 0; i--) {//从len/2到1，反复调整堆
		AdjustDown(arr, i, len);
	}
}


//输出大顶堆时逐步删除顶端元素，与最后一个元素交换
void Sort::AdjustDown(int arr[], int k, int len) {//将序号为k的元素向下调整
	int temp = arr[k];//arr[0]暂存
	for (int i = 2 * k + 1; i < len; i = i * 2 + 1) {//找k的分支，找最大。因为下标从0开始，逐次换行要*2+1
		if (i < len&&arr[i] < arr[i + 1]) {
			i++;//取值较大的结点的下标
		}
		if (temp >= arr[i]) {
			break;
		}
		else {
			arr[k] = arr[i];//大的值上移，将A[i]调整至双亲结点
			k = i;//在准备找下一个分支
		}
	}//for
	arr[k] = temp;
}

//大根堆插入元素时，先将待插入元素放入堆的末尾，再逐渐向上调整
void Sort::AdjustUp(int arr[], int k) {//序号为k的数字向上调整
	int temp = arr[k];//暂存
	for (int i = (k - 1) / 2; i > 0; i = (i - 1) / 2) {
		if (i > 0 && arr[i] < temp) {
			arr[k] = arr[i];//父节点下移
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