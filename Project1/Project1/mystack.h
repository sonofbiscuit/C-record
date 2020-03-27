#pragma once

//�����������ʱ
template<typename T, int max_size>
class Stack {
public:
	Stack();

	void push(T item);
	void pop();

	T& top();

	T size();

private:
	int* data_[10];//max_size
	int count_;
};

template<typename T, int max_size>
Stack<T, max_size>::Stack()
	:data_{ nullptr }
	, count_(0){}

template<typename T, int max_size>
void Stack<T, max_size>::push(T item) {
	if (count_ >= 10) {//max_size
		return;
	}
	data_[count_] = new T;
	*data_[count_] = item;
	count_++;
}

template<typename T, int max_size>
void Stack<T, max_size>::pop() {
	if (count_ == 0) {
		return;
	}
	delete(data_[--count_]);
}

template<typename T, int max_size>
T& Stack<T, max_size>::top() {
	if (count_ == 0) {
		throw"Error!";
	}
	return *data_[count_ - 1];
}

template<typename T, int max_size>
T Stack<T, max_size>::size() {
	return count_;
}