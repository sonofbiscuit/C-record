#include<string>
#include<vector>
#include<cstddef>
#include<algorithm>
#include<iostream>
using namespace std;

namespace strtool{//命名空间strtool,防止冲突
string trim(const string& str){
	string::size_type pos=str.find_first_not_of(' ');//找第一个空格
	if(pos == string::npos){//find_first_not_of未找到返回string::npos,说明一个空格都没有
		return str;
	}
	string::size_type pos1=str.find_last_not_of(' ');//第二个空格
	if(pos1!=string::npos){//找到空格
		return str.substr(pos,pos1-pos+1);
	}
	return str.substr(pos);//find_last_not_of没有空格,从pos开始的整个str
}


int split(const string& str, vector<string>& ret_, string spl){//分割一个字符,如13221,123,213
	if(str.empty()){
		return 0;
	}
	string temp;
	string::size_type com_pos=0;
	string::size_type pos_begin=str.find_first_not_of(spl);//找到第一个非spl的位置

	while(pos_begin!=string::npos){//存在除了spl之外的字符
		com_pos=str.find(spl ,pos_begin);//str中从pos_begin开始查找spl
		if(com_pos!=string::npos){//在pos_begin之后还有spl
			temp=str.substr(pos_begin,com_pos-pos_begin);
			pos_begin=com_pos+spl.length();//pos_begin后移到下一个字符
		}else{//pos_begin之后没有spl
			temp=str.substr(pos_begin);
			pos_begin=com_pos;
		}

		if(!temp.empty()){
			ret_.push_back(temp);
			temp.clear();
		}
	}
}


string replace(const string& str, const string& old_, const string& new_){
	string temp="";
	string::size_type pos_begin=0;
	string::size_type pos_find=str.find(old_,pos_begin);

	while(pos_find!=string::npos){//有old_
		temp.append(str.data()+pos_begin, pos_find-pos_begin);
		temp+=new_;
		pos_begin=pos_find+1;
		pos_find=str.find(old_,pos_begin);
	}
	//已经没了,pos_begin未到底
	if(pos_begin<str.length()){
		temp.append(str.begin()+pos_begin,str.end());
	}
	return temp;
}

}//#namespace strtool

int main(){
	cout<<strtool::trim(" adadq ")<<endl;
	cout<<"----------------"<<endl;
	vector<string> vec;
	strtool::split(",,a,f,rf,,gq,g,",vec,",");
	for(int i=0;i<vec.size();i++){
		cout<<vec[i]<<endl;
	}
	cout<<"----------------"<<endl;
	string tem=strtool::replace("12123a13a1312", "a", "b");
	cout<<tem<<endl;
}