
template <typename T>
int getLength(T& array) {
	return (sizeof(array) / sizeof(array[0]));
}