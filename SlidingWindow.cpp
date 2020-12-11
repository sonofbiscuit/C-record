class Solution(){//找无重复最大子串
public:
	int lengthOfLongestSubstring(string s){
		if(s=="")
			return 0;
		unordered_set<char> lookup;
		int maxLength=0;
		int left=0;
		for(int i=0;i<s.size();i++){
			while(lookup.find(s[i])!=lookup.end()){//找到重复
				lookup.erase(s[left]);//删左
				left++;
			}
			maxLength=max(maxLength,lookup.size);
			lookup.insert(s[i]);
		}
		return maxLength;
	}
}

//暴力法,利用unique唯一性，比较长度
//unique因为是去除相邻的重复元素，因此通常使用前容器应该要是有序的。unique会自动返回去重后的尾迭代器
//对332211567进行sort后为112233567  unique后变为123567567 
//去除的数字数目n会使用后n位填补



class Solution {
public:
    int counts=0;
    int lengthOfLongestSubstring(string s) {
        for(int j=0;j<=s.length();j++){
        	for(int i=0;i<=j;i++){
            	string temp=s.substr(i,j-i);
            	sort(temp.begin(),temp.end());//先排序
            	string::iterator iterEnd=unique(temp.begin(),temp.end());//再unique
            	temp.erase(iterEnd,temp.end());
            	if(temp.length()!=s.substr(i,j-i).length()){
                	continue;
           		}else{
                	if(temp.length()>=counts)
                    counts=temp.length();
            	}
        	}
    	}
        return counts;
    }
};
