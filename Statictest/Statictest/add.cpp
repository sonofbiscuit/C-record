

int global = 10;	
//static int global = 10;   <-----�޷���extern

int add(int a, int b) {
	return a + b;
}

int get_global() {
	return global;
}