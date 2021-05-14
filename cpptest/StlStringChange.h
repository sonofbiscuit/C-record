#include<string>
#include<vector>
#include<cstddef>
#include<algorithm>
#include<iostream>
using namespace std;

#pragma once
namespace strtool {
	string trim(const string& str);
	int split(const string& str, vector<string>& ret_, string spl);
	string replace(const string& str, const string& old_, const string& new_);
}