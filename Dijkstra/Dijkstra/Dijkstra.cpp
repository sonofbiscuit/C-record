#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include<vector>
#include <fstream>

using namespace std;

const int maxnum = 100;
const int maxint = 99999;

int n;//�����
int line;//����
int dist[maxnum];//��ʾ��ǰ�㵽Դ��ĳ���
int prevs[maxnum];//��ǰ����ǰһ�����
int c[maxnum][maxnum];//��¼ͼ�������ĳ���


//n  <----------- n nodes
//v  <----------- the source code
//dist[]  <------ the distance from the ith node to the source node
//prevs[]  <----- the previous node od the ith node
//c[][]   <------ every two node's distance
void dijkstra(int n, int v, int* dist, int* prevs, int c[maxnum][maxnum]) {
	bool s[maxnum];//�ж��Ƿ��Ѵ���õ㵽����
	//��ʼ��
	for (int i = 1; i <= n; i++) {
		dist[i] = c[v][i];//��ʼ��·��
		s[i] = 0;//��ʼ״̬ �㶼δ�ù�
		if (dist[i] == maxint)//Դ�㵽����·��
			prevs[i] = 0;
		else
			prevs[i] = v;//Դ�㵽����·�����õ�ǰ��ΪԴ��
	}
	dist[v] = 0;//Դ���������Ϊ0
	s[v] = 1;//Դ���ù�


	//��δ����s���ϵĽ���У�ȡdist[]��С�Ľ�㣬����s
	//һ��s����������v�н�㣬dist�ͼ�¼�˴�Դ�㵽�������ж���֮������·������
	//�ӵڶ�����㿪ʼ
	for (int i = 2; i <= n; i++) {
		int temp = maxint;
		int u = v;
		//�ҳ���ǰδʹ�õĵ�j�����dist     <--------------------distÿ�θ��º���Ȼѡ��С
		for (int j = 1; j < n; j++) {
			if (!s[j] && dist[j] < temp) {
				u = j;//u���浱ǰ��
				temp = dist[j];
			}
		}
		s[u] = 1;//�ҵ��󣬽�u����s
		//����dist
		for (int j = 1; j <= n; j++) {
			if (!s[j] && c[u][j] < temp) {//û�ù�������·��
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


//����dist���Ҵ�Դ��v��u��·��,�������
void searchPath(int* prevs , int v , int u) {
	int que[maxnum];
	int tot = 1;
	que[tot] = u;
	tot++;

	int temp = prevs[u];
	//cout << endl << "------------------------------" << prevs[u] << endl;
	while (temp != v) {//��Դ���
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
	int p, q, len;//p,q���㼰��·������

	//��ʼ��c[][]Ϊmaxint
	for (int i = 0; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			c[i][j] = maxint;
		}
	}

	for (int i = 1; i <= line; i++) {
		cin >> p >> q >> len; 
		if (len < c[p][q]) {
			c[p][q] = len;//����ͼ
			c[q][p] = len;
		}
	}

	//��ʼ��dist
	for (int i = 1; i <= n; i++) {
		dist[i] = maxint;
	}
	//���c
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++)
			printf("%2d \t", c[i][j]);
		printf("\n");
	}
	dijkstra(n, 1, dist, prevs, c);
	// ���·������
	cout << "Դ�㵽���һ����������·������: " << dist[n] << endl;

	// ·��
	cout << "Դ�㵽���һ�������·��Ϊ: ";
	searchPath(prevs, 1, n);
}