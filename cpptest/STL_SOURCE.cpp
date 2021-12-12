
#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <typeinfo>
#include <stack>
#include <queue>
#include <array>
#include <deque>
#include <regex>
#include <time.h>
#include <numeric>
#include <functional>



using namespace std;
/*
	STL从来不是以面向对象的思想去编写的，而是以泛型编程的思维去编写的
*/


//  ######################################################  traits   ###############################################

/* STL源码设计了获取迭代器value_type, iterator_category, distance_type等的全局函数*/
template<typename Iterator>
inline typename iterator_traits<Iterator>::value_type*
value_type(const Iterator&) {
	// 根据Iterator是class type还是 T*或是const T* 原生指针， 调用泛化或特化的iterator_traits
	return static_cast<typename iterator_traits<Iterator>::value_type*>(0);
}

//iterator_category
//distance_type


// 利用class类的声明内嵌型别设计的萃取机
// 也是STL萃取器的泛化版本
template<typename Iterator>
struct iterator_traits {
	typedef typename Iterator::iterator_category iterator_category;   // 迭代器类型
	typedef typename Iterator::value_type        value_type;          // 迭代器所指对象的类型
	typedef typename Iterator::difference_type   difference_type;     // 用来指两个迭代器之间的距离
	typedef typename Iterator::pointer           pointer;             // represents a pointer-to-value_type. 
	typedef typename Iterator::reference         reference;
	// reference “迭代器所指之物的内容能否改变”。 
	// 不允许改变的称为 constant iterator, 如 const int* p
	// 可以改变的称为 mutable iterator, 如 int* p
};

// 原生指针不是 class type ，无法定义内嵌型别
// STL利用C++偏特化（Partial Specialization）的方式为泛化设计提供一个特化版本，即将泛化版本中的某些template参数赋予明确的指定
template<typename T>
class C {...};  // 这个泛化版本可以接受任意型别的 T

template<typename T>
class C<T*> {...};   // 这个特化版本仅适用于 “T为原生指针”的情况。
// T 为原生指针， 即是“T为任何类别”的一个特化版本

// STL对原生指针T*设计的特化版本的萃取机
template<typename T>
class iterator_traits<T*> {
	typedef random_access_iterator_tag      iterator_category;
	typedef T                               value_type;
	typedef ptrdiff_t                       difference_type;
	typedef T*                              pointer;
	typedef T&                              reference;
};










// ######################################################  algorithm   ##########################################

// =======================================================lower_bound  upper_bound=======================================
/*
	lower_bound  在不破坏排序状态的原则下，可插入value的第一个位置
	upper_bound  在不破坏排序状态的原则下，可插入value的最后一个合适位置
*/


template<typename ForwardIterator, typename T>
inline ForwardIterator lower_bound(ForwardIterator first, ForwardIterator last, const T& value) {
	return __lower_bound(first, last, value, distance_type(first), iterator_type(first));
}

template<typename ForwardIterator, typename T, typename Distance>
// 模板函数需要发现与其迭代器参数最具体的类，以便在编译时能够最高效的使用算法
// 因此， 对于Iterator类型，必须定义每个迭代器的   iterator_traits<iterator>::iterator_category  是描述迭代器行为最具体的类标记
ForwardIterator __lower_bound(ForwardIterator first, ForwardIterator last, const T& value, Distance*, forward_iterator_tag) {
	Distance len = 0;
	distance(first, last, len);  // Return distance between iterators
	Distance half;
	ForwardIterator mid;

	while (len > 0)
	{
		half = len >> 1;
		advance(mid, half);
		if (*mid < value) {
			first = mid;
			++first;
			len = len - half - 1;
		}
		else {  //<=
			len = half;
		}
	}
	return first;
}

// 使用random_access_iterator
template<typename RandomAccessIterator, typename T, typename Distance>
RandomAccessIterator __lower_bound(RandomAccessIterator first, RandomAccessIterator last, const T& value, Distance*, random_access_iterator_tag) {
	Distance len = last - first + 1;
	Distance half;
	RandomAccessIterator mid;

	while (len > 0) {
		half = len >> 1;
		mid = first + half;
		if (*mid < value) {
			first = mid;
			len = len - half - 1;
		}
		else {
			len = half;
		}
	}
	return first;
}


template<typename ForwardIterator, typename T>
ForwardIterator upper_bound(ForwardIterator first, ForwardIterator last, const T& value) {
	return __upper_bound(first, last, value, distance_type(first), iterator_category(first));
}
template<typename ForwardIterator, typename T, typename Distance>
ForwardIterator __upper_bound(ForwardIterator first, ForwardIterator last, const T& value, Distance*, forward_iterator_tag) {
	Distance len = 0;
	distance(first, last, len);  // 整个区间的长度
	Distance half;
	ForwardIterator mid;

	while (len) {
		half = len >> 1;
		mid = first;
		advance(mid, half);  //将mid移到中间位置
		if (value < *mid) {  // 在mid左侧
			len = half;
		}
		else {  //在 mid右侧
			first = mid;
			++first;
			len = len - half - 1;
		}
	}
	return first;
}


// [first, last)
template<typename RandomIterator, typename T>
RandomIterator upper_bound(RandomIterator first, RandomIterator last, T value) {
	return __upper_bound(first, last, value, distance_type(first), iterator_category(first));
}

template<typename RandomIterator, typename T, typename Distance>
RandomIterator __upper_bound(RandomIterator first, RandomIterator last, const T& value, Distance*, random_access_iterator_tag) {
	Distance len;
	len = last - first;
	Distance half;
	RandomIterator mid;

	while (len > 1) {
		half = len >> 1;
		mid = first + half;
		if (value < *mid) {
			len = half;
		}
		else {  //   *mid >=value
			first = mid + 1;
			len = len - half - 1;
		}
	}
	return first;
}

