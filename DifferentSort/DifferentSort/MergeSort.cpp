#include <iostream>
#include "Sort.h"
using namespace std;

/*
归并排序
	归并排序
	基数排序
*/

//归并排序基于分治思想
void Sort::MergeSort(int arr[], int low, int high) {
	if (low >= high) { return; } // 终止递归的条件，子序列长度为1
	int mid = (low + high) / 2;  // 取得序列中间的元素
	MergeSort(arr, low, mid);  // 对左半边递归
	MergeSort(arr, mid + 1, high);  // 对右半边递归
	Merge(arr, low, mid, high);  // 合并
}

void Sort::Merge(int arr[], int low, int mid, int high) {
	//low为第1有序区的第1个元素，i指向第1个元素, mid为第1有序区的最后1个元素
	int i = low, j = mid + 1, k = 0;  //mid+1为第2有序区第1个元素，j指向第1个元素
	int *temp = new int[high - low + 1]; //temp数组暂存合并的有序序列
	while (i <= mid && j <= high) {
		if (arr[i] <= arr[j]) //较小的先存入temp中
			temp[k++] = arr[i++];
		else
			temp[k++] = arr[j++];
	}
	while (i <= mid)//若比较完之后，第一个有序区仍有剩余，则直接复制到t数组中
		temp[k++] = arr[i++];
	while (j <= high)//同上
		temp[k++] = arr[j++];
	for (i = low, k = 0; i <= high; i++, k++)//将排好序的存回arr中low到high这区间
		arr[i] = temp[k];
	delete[]temp;//释放内存，由于指向的是数组，必须用delete []
}


//<------------------------------------->
//基数排序

//求数据的最大位数
int Sort::maxBit(int arr[], int n) {
	int d = 1;//保存最大位数
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
	int *temp = new int[n];//临时数组
	int count[10];//标记数组
	int i, j, k;
	for (i = 1; i <= d; i++) {
		for (j = 0; j < 10; j++) {
			count[j] = 0;//清空标记数组
		}
		for (j = 0; j < n; j++) {
			k = (arr[j] / radix) % 10;//取位
			count[k]++;//记录每个位次的数字个数
		}
		for (j = 1; j < 10; j++)
			count[j] = count[j - 1] + count[j]; //给每个书留足位置
		for (j = n - 1; j >= 0; j--) //将所有桶中记录依次收集到tmp中
		{
			k = (arr[j] / radix) % 10;
			temp[count[k] - 1] = arr[j];
			count[k]--;
		}
		for (j = 0; j < n; j++) //将临时数组的内容复制到data中
			arr[j] = temp[j];
		radix = radix * 10;//取高一位
	}	
}




/*
arr:73 22 93 43 55 14 28 65 39 81
//取个位;
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

					  temp[n]存入
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

将temp内容存入arr
arr:81 22 73 93 43 14 55 65 28 39
//取十位;
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


					  temp[n]存入
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