#pragma once


class Sort {
public:
	template <typename T>
	int getLength(T& array);
	void DInsertSort(int arr[], int n);
	void HInsertSort(int arr[], int n);
	void SHellSort(int arr[], int n);
	
	void MergeSort(int arr[], int low, int high);
	void Merge(int arr[], int low, int mid, int high);

	int maxBit(int arr[], int n);
	void RadixSort(int arr[], int n);

	void BubbleSort(int arr[], int n);
	int Partion(int arr[], int low, int high);
	void QuickSort(int arr[], int low, int high);

	void SelectSort(int arr[], int n);

	void HeapSort(int arr[], int len);
	void BuildMaxHeap(int arr[], int len);
	void AdjustUp(int arr[], int k);
	void AdjustDown(int arr[], int l, int len);
	void Swap(int arr[], int x, int y);
};

template <typename T>
int Sort::getLength(T& array) {
	return (sizeof(array) / sizeof(array[0]));
}

