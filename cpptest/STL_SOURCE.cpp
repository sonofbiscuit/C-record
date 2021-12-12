
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
	STL������������������˼��ȥ��д�ģ������Է��ͱ�̵�˼άȥ��д��
*/


//  ######################################################  traits   ###############################################

/* STLԴ������˻�ȡ������value_type, iterator_category, distance_type�ȵ�ȫ�ֺ���*/
template<typename Iterator>
inline typename iterator_traits<Iterator>::value_type*
value_type(const Iterator&) {
	// ����Iterator��class type���� T*����const T* ԭ��ָ�룬 ���÷������ػ���iterator_traits
	return static_cast<typename iterator_traits<Iterator>::value_type*>(0);
}

//iterator_category
//distance_type


// ����class���������Ƕ�ͱ���Ƶ���ȡ��
// Ҳ��STL��ȡ���ķ����汾
template<typename Iterator>
struct iterator_traits {
	typedef typename Iterator::iterator_category iterator_category;   // ����������
	typedef typename Iterator::value_type        value_type;          // ��������ָ���������
	typedef typename Iterator::difference_type   difference_type;     // ����ָ����������֮��ľ���
	typedef typename Iterator::pointer           pointer;             // represents a pointer-to-value_type. 
	typedef typename Iterator::reference         reference;
	// reference ����������ָ֮��������ܷ�ı䡱�� 
	// ������ı�ĳ�Ϊ constant iterator, �� const int* p
	// ���Ըı�ĳ�Ϊ mutable iterator, �� int* p
};

// ԭ��ָ�벻�� class type ���޷�������Ƕ�ͱ�
// STL����C++ƫ�ػ���Partial Specialization���ķ�ʽΪ��������ṩһ���ػ��汾�����������汾�е�ĳЩtemplate����������ȷ��ָ��
template<typename T>
class C {...};  // ��������汾���Խ��������ͱ�� T

template<typename T>
class C<T*> {...};   // ����ػ��汾�������� ��TΪԭ��ָ�롱�������
// T Ϊԭ��ָ�룬 ���ǡ�TΪ�κ���𡱵�һ���ػ��汾

// STL��ԭ��ָ��T*��Ƶ��ػ��汾����ȡ��
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
	lower_bound  �ڲ��ƻ�����״̬��ԭ���£��ɲ���value�ĵ�һ��λ��
	upper_bound  �ڲ��ƻ�����״̬��ԭ���£��ɲ���value�����һ������λ��
*/


template<typename ForwardIterator, typename T>
inline ForwardIterator lower_bound(ForwardIterator first, ForwardIterator last, const T& value) {
	return __lower_bound(first, last, value, distance_type(first), iterator_type(first));
}

template<typename ForwardIterator, typename T, typename Distance>
// ģ�庯����Ҫ������������������������࣬�Ա��ڱ���ʱ�ܹ����Ч��ʹ���㷨
// ��ˣ� ����Iterator���ͣ����붨��ÿ����������   iterator_traits<iterator>::iterator_category  ��������������Ϊ����������
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

// ʹ��random_access_iterator
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
	distance(first, last, len);  // ��������ĳ���
	Distance half;
	ForwardIterator mid;

	while (len) {
		half = len >> 1;
		mid = first;
		advance(mid, half);  //��mid�Ƶ��м�λ��
		if (value < *mid) {  // ��mid���
			len = half;
		}
		else {  //�� mid�Ҳ�
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

