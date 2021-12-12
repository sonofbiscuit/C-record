#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <random>
#include <cstdlib>
#include <set>
#include <bit>
#pragma GCC optimize(2) //O2优化
using namespace std;

// objectPool
template<typename T>
class UserPool {
private:
    std::vector<std::unique_ptr<T>> user_pool;
public:
    //using DelType = std::function<void(T*)>;

    std::function<void(T*)> deleter = [](T* p) {
        delete p;
    };

    void add(std::unique_ptr<T> t) {
        this->user_pool.emplace_back(std::move(t));
    }

    //std::unique_ptr<T, DelType> get()
    std::unique_ptr<T, decltype(deleter)> get() {
        if (user_pool.empty()) {
            cout << "empty students!" << endl;
        }
        //bind a custom deleter for default unique_ptr
        std::unique_ptr<T, decltype(deleter)> ptr(user_pool.back().release(), [this](T* t) {
            user_pool.push_back(std::unique_ptr<T>(t));
        });
        user_pool.pop_back();
        return std::move(ptr);
    }

    [[nodiscard]] bool is_empty() const {
        return user_pool.empty();
    }

    [[nodiscard]] int pool_size() const {
        return user_pool.size();
    }

};


// student struct
struct Users {
    vector<int> activities;
    int index;

    //用户编号和随机生成一些其所拥有的活动
    Users(int index, int activity_num) {   // student index 用户编号,  activity count 活动数量
        this->index = index;
        static uniform_int_distribution<unsigned> u(0, 9);
        static default_random_engine e;
        // 当种子和随机数生成器在循环内定义的时候，随机数生成的值会相同。static解决该问题
        for (int i = 0; i < activity_num; ++i) {
            int temp = u(e);
            this->activities.emplace_back(temp);
        }
    }

    [[nodiscard]] int Get_Variety() const {
        cout << "用户 " << this->index << "的活动的个数为:  " << (int)this->activities.size() << endl << endl;
        return (int)this->activities.size();
    }

    ~Users() {
        cout << "delete " << this->index << endl;
    }

};


void Display(vector<Users*>& pool) {   //展示所有用户活动
    for (auto& a : pool) {
        cout << a->index << " : ";
        for (auto& b : a->activities) {
            cout << b << " ";
        }
        cout << endl;
    }
}

/*
int main() {
    UserPool<Users> pool;
    int user_num=3;
    int ac_num=5;
    //cout << "输入用户数:";
    //cin >> user_num;
    //cout << "输入每个用户的活动数: ";
    //cin >> ac_num;
    for (int i = 0; i < user_num; ++i) {
        cout << "create user: " << i << endl;
        pool.add(std::make_unique<Users>(i, ac_num));
    }

    {
        //在这个块里可以通过get()获取到对象池里的对象，每获取一次，会从对象池中删除
        cout << "pool size: " <<pool.pool_size() << endl;
        auto p1 = pool.get();
        cout << p1->index << endl;
        auto p2 = pool.get();
        cout << p2->index << endl;
        cout << "pool size: " << pool.pool_size() << endl;
    }
    // 到这里，所有对象还是在对象池里，不受前一块的get()的影响
    cout << "pool size: " << pool.pool_size() << endl;
    // 最后会自动调用析构函数删除所有对象。
    return 0;
}*/