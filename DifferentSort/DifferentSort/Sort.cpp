#include <iostream>
#include "time.h"
#include "Sort.h"
using namespace std;

#define CLOCKS_PER_SEC  ((clock_t)1000)

int main() {
	clock_t start, finish;
	Sort sort;
	int a[] = { 4,9,1,2,4,8,0 };
	int n = sort.getLength(a);
	//int loop = 1000;
	
	
	start = clock();
	sort.DInsertSort(a, n);
	finish = clock();
	double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "直接插入排序时间: " << duration << endl;

	
	sort.HInsertSort(a, n);	
	sort.SHellSort(a, n);
	
	cout << "归并排序:";
	sort.MergeSort(a, 0, 6);
	for (int a : a) {
		cout << a << " ";
	}
	cout << endl;

	int a2[] = { 73, 22, 93, 43, 55, 14, 28, 65, 39, 81 };
	cout << "基数排序之前： ";
	for (int c : a2) {
		cout << c << " ";
	}
	cout << endl;
	cout << "基数排序之后： ";
	sort.RadixSort(a2, 10);
	for (int c : a2) {
		cout << c << " ";
	}
	cout << endl;

	cout << "快速排序之前： ";
	int a3[] = { 4,9,1,2,4,8,0,5,1,24,6,8,9 };
	for (int c : a3) {
		cout << c << " ";
	}
	cout << endl;
	int n1 = sort.getLength(a3);
	sort.QuickSort(a3, 0, n1-1);
	cout << "快速排序之后： ";
	for (int c : a3) {
		cout << c << " ";
	}
	cout << endl;

	cout << "冒泡排序之前： ";
	int a4[] = { 4,9,1,2,4,8,0,5,1,24,6,8,9 };
	for (int c : a4) {
		cout << c << " ";
	}
	cout << endl;
	int n2 = sort.getLength(a4);
	sort.BubbleSort(a4, n2);
	cout << "冒泡排序之后： ";
	for (int c : a4) {
		cout << c << " ";
	}
	cout << endl;

	cout << "简单选择排序之前： ";
	int a5[] = { 4,9,1,2,4,8,0,5,1,24,6,8,9 };
	for (int c : a5) {
		cout << c << " ";
	}
	cout << endl;
	int n3 = sort.getLength(a5);
	sort.SelectSort(a5, n3);
	cout << "简单选择排序之后： ";
	for (int c : a5) {
		cout << c << " ";
	}
	cout << endl;

	cout << "堆排序之前： ";
	int a6[] = { 4,9,1,2,4,8,0,5,1,24,6,8,9 };
	for (int c : a6) {
		cout << c << " ";
	}
	cout << endl;
	int n4 = sort.getLength(a6);
	sort.HeapSort(a6, n4);
}