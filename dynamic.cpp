class Solution {
public:
    string longestPalindrome(string s) {
        int len=s.size();
		if(len==0||len==1)
			return s;
		vector<vector<int>> dp(len,vector<int>(len));
		for(int i=0;i<len;i++)
			dp[i][i]=1;
        int maxlen=1;
		int index=0;
		for(int j=1;j<len;j++){
			for(int i=0;i<j;i++){
				if(s[i]=s[j]){//相同
                    if(j-i<3){//若子串长度是2或者1
					    dp[i][j]=1;//那么i到j处为回文
                    }
				    else{//若长度大于2，则看上一级子串是否回文，若回文，则加上s[i]和s[j]后依然回文
				    	dp[i][j]=dp[i+1][j-1];
				    }
                }else{//否则的话  i到j中不回文
                    dp[i][j]=0;
                }
				if(dp[i][j]==1){//是回文  求长度
					int slength=j-i+1;
					if(slength>maxlen){
						maxlen=slength;
						index=i;
					}
				}//if
			}//for
		}//for
		return s.substr(index,index+maxlen);
    }//end
};

//dp[i][i]=1   dp[i][i+1]=1 if s[i]=s[i+1]


/*

1 2 3 3 2 1

dp[1][1]
dp[2][2]
dp[3][3]
dp[4][4]
dp[5][5]
dp[6][6]
//初始化

maxlen=1
index=0

j=1;j<s.size();j++
i=0;i<j;i++

1 2 3 3 2 1
  j
i

s[i]=s[j]
=> 
if(j-i<3)
	dp[i][j]=1  
else
	dp[i][j]=dp[i-1][j-1]

if(dp[i][j]==1)
{
	int slength=j-i+1;
	if(slength>maxlength){
		maxlength=slength;
		index=i;
	}
}

return s[index][index+maxlenght]
*/



//暴力
class Solution {
public:
    string longestPalindrome(string s) {
        int len=s.length();
        string res="";//结果
        string tem="";//子串
        for(int i=0;i<len;i++){
            for(int j=i;j<len;j++){
                tem=tem+s[j];
                string temp=tem;//temp为反转
                std::reverse(temp.begin(),temp.end());
                if(temp==tem)
                    res=res.length()>tem.length()?res:tem;
            }
            tem="";
        }
        return res;
    }
};