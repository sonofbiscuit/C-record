#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include<vector>
#include <fstream>

using namespace std;

const int maxnum = 100;
const int maxint = 99999;

int n;//结点数
int line;//边数
int dist[maxnum];//表示当前点到源点的长度
int prevs[maxnum];//当前结点的前一个结点
int c[maxnum][maxnum];//记录图的两点间的长度


//n  <----------- n nodes
//v  <----------- the source code
//dist[]  <------ the distance from the ith node to the source node
//prevs[]  <----- the previous node od the ith node
//c[][]   <------ every two node's distance
void dijkstra(int n, int v, int* dist, int* prevs, int c[maxnum][maxnum]) {
	bool s[maxnum];//判断是否已存入该点到集合
	//初始化
	for (int i = 1; i <= n; i++) {
		dist[i] = c[v][i];//初始化路径
		s[i] = 0;//初始状态 点都未用过
		if (dist[i] == maxint)//源点到此无路径
			prevs[i] = 0;
		else
			prevs[i] = v;//源点到此有路径，该点前驱为源点
	}
	dist[v] = 0;//源点自身距离为0
	s[v] = 1;//源点用过


	//在未放入s集合的结点中，取dist[]最小的结点，放入s
	//一旦s包含了所有v中结点，dist就记录了从源点到其他所有顶点之间的最短路径长度
	//从第二个结点开始
	for (int i = 2; i <= n; i++) {
		int temp = maxint;
		int u = v;
		//找出当前未使用的点j的最短dist     <--------------------dist每次更新后依然选最小
		for (int j = 1; j < n; j++) {
			if (!s[j] && dist[j] < temp) {
				u = j;//u保存当前点
				temp = dist[j];
			}
		}
		s[u] = 1;//找到后，将u存入s
		//更新dist
		for (int j = 1; j <= n; j++) {
			if (!s[j] && c[u][j] < temp) {//没用过并且有路径
				int newdist = dist[u] + c[u][j];
				if (newdist < dist[j]) {
					dist[j] = newdist;
					prevs[j] = u;
				}
			}
		}
		for (int j = 1; j <= n; j++)
			cout << dist[j] << endl;
	}
}


//根据dist查找从源点v到u的路径,并且输出
void searchPath(int* prevs , int v , int u) {
	int que[maxnum];
	int tot = 1;
	que[tot] = u;
	tot++;

	int temp = prevs[u];
	//cout << endl << "------------------------------" << prevs[u] << endl;
	while (temp != v) {//找源结点
		que[tot] = temp;
		tot++;
		temp = prevs[temp];
	}
	que[tot] = v;
	for (int i = tot; i >= 1; i--) {
		if (i != 1)
			cout << que[i] << "->";
		else
			cout << que[i];
	}
}



int main() {
	freopen("text.txt", "r" ,stdin);
	cin >> n;
	cin >> line;
	int p, q, len;//p,q两点及其路径长度

	//初始化c[][]为maxint
	for (int i = 0; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			c[i][j] = maxint;
		}
	}

	for (int i = 1; i <= line; i++) {
		cin >> p >> q >> len; 
		if (len < c[p][q]) {
			c[p][q] = len;//无向图
			c[q][p] = len;
		}
	}

	//初始化dist
	for (int i = 1; i <= n; i++) {
		dist[i] = maxint;
	}
	//输出c
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++)
			printf("%2d \t", c[i][j]);
		printf("\n");
	}
	dijkstra(n, 1, dist, prevs, c);
	// 最短路径长度
	cout << "源点到最后一个顶点的最短路径长度: " << dist[n] << endl;

	// 路径
	cout << "源点到最后一个顶点的路径为: ";
	searchPath(prevs, 1, n);
}