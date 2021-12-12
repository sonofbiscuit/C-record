#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <random>
#include <cstdlib>
#include <set>
#include <bit>
#pragma GCC optimize(2) //O2�Ż�
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

    //�û���ź��������һЩ����ӵ�еĻ
    Users(int index, int activity_num) {   // student index �û����,  activity count �����
        this->index = index;
        static uniform_int_distribution<unsigned> u(0, 9);
        static default_random_engine e;
        // �����Ӻ��������������ѭ���ڶ����ʱ����������ɵ�ֵ����ͬ��static���������
        for (int i = 0; i < activity_num; ++i) {
            int temp = u(e);
            this->activities.emplace_back(temp);
        }
    }

    [[nodiscard]] int Get_Variety() const {
        cout << "�û� " << this->index << "�Ļ�ĸ���Ϊ:  " << (int)this->activities.size() << endl << endl;
        return (int)this->activities.size();
    }

    ~Users() {
        cout << "delete " << this->index << endl;
    }

};


void Display(vector<Users*>& pool) {   //չʾ�����û��
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
    //cout << "�����û���:";
    //cin >> user_num;
    //cout << "����ÿ���û��Ļ��: ";
    //cin >> ac_num;
    for (int i = 0; i < user_num; ++i) {
        cout << "create user: " << i << endl;
        pool.add(std::make_unique<Users>(i, ac_num));
    }

    {
        //������������ͨ��get()��ȡ���������Ķ���ÿ��ȡһ�Σ���Ӷ������ɾ��
        cout << "pool size: " <<pool.pool_size() << endl;
        auto p1 = pool.get();
        cout << p1->index << endl;
        auto p2 = pool.get();
        cout << p2->index << endl;
        cout << "pool size: " << pool.pool_size() << endl;
    }
    // ��������ж������ڶ���������ǰһ���get()��Ӱ��
    cout << "pool size: " << pool.pool_size() << endl;
    // �����Զ�������������ɾ�����ж���
    return 0;
}*/