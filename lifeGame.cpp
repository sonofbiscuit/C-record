/*
给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：

如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
根据当前状态，写一个函数来计算面板上所有细胞的下一个（一次更新后的）状态。下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。

*/

class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        int rows=board.size();
        int cols=board[0].size();
        int neighbors[]={-1,0,1};
        vector<vector<int>> copyBoard(rows,vector<int>(cols,0));
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                copyBoard[row][col] = board[row][col];//赋值一份相同的数组
            }
        }

        for(int row=0;row<rows;row++){
            for(int col=0;col<cols;col++){
                int livecell=0;
                for(int i=0;i<3;i++){
                    for(int j=0;j<3;j++){
                        if (!(neighbors[i] == 0 && neighbors[j] == 0)) {
                        int r=row+neighbors[i];
                        int c=col+neighbors[j];
                        if((r>=0&&r<rows)&&(c>=0&&c<cols)&&(copyBoard[r][c]==1)){
                            livecell+=1;//统计周围八个格子的活细胞数
                         }
                        }
                    }
                }
                //规则1/3，此时细胞死亡
                if ((copyBoard[row][col] == 1) && (livecell < 2 || livecell > 3)) {
                    board[row][col] = 0;
                }
                //规则2/4,细胞存活和复活
                if (copyBoard[row][col] == 0 && livecell == 3) {
                    board[row][col] = 1;
                }
            }
        }
    }
};
