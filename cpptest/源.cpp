/*
map
//map<key, value>
//unique keys
//ordered

multimap
//Multiple elements in the container can have equivalent keys.
//ordered
//multimap<key,value>

unordered_map
unordered_multimap
//unordered
//...

//unordered_map的count和find
count返回找到的数量
find返回迭代器, auto a = xxx.find();  a->first, a->second...


map优点：

有序性，这是map结构最大的优点，其元素的有序性在很多应用中都会简化很多的操作
红黑树，内部实现一个红黑书使得map的很多操作在lgn的时间复杂度下就可以实现，因此效率非常的高
缺点： 空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间

适用处：对于那些有顺序要求的问题，用map会更高效一些

unordered_map：

优点： 因为内部实现了哈希表，因此其查找速度非常的快
缺点： 哈希表的建立比较耗费时间
适用处：对于查找问题，unordered_map会更加高效一些，因此遇到查找问题，常会考虑一下用unordered_map
=================================================================

set   //unique key
multiset //multi key
//The value of an element is also the key used to identify it.
//ordered


unordered_set
unordered_multiset  //multi
//unordered
//The value of an element is also the key used to identify it.

何时使用set

我们需要有序的数据。
我们将不得不打印/访问数据（按排序顺序）。
我们需要元素的前任/后继。
由于set是有序的，因此我们可以在set元素上使用binary_search（），lower_bound（）和upper_bound（）之类的函数。这些函数不能在unordered_set（）上使用。

何时使用unordered_set
我们需要保留一组不同的元素，并且不需要排序。
我们需要单元素访问，即无遍历。



*/