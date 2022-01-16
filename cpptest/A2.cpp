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
class UserPool{
private:
    std::vector<std::unique_ptr<T>> user_pool;
public:
    //using DelType = std::function<void(T*)>;

    std::function<void(T*)> deleter = [](T* p){
        delete p;
    };

    void add(std::unique_ptr<T> t){
        this->user_pool.emplace_back(std::move(t));
    }

    //std::unique_ptr<T, DelType> get()
    std::unique_ptr<T, decltype(deleter)> get(){
        if(user_pool.empty()){
            cout<<"empty students!"<<endl;
        }
        //bind a custom deleter for default unique_ptr
        std::unique_ptr<T, decltype(deleter)> ptr(user_pool.back().release(),[this](T* t){
            user_pool.push_back(std::unique_ptr<T>(t));
        });
        user_pool.pop_back();
        return std::move(ptr);
    }

    [[nodiscard]] bool is_empty() const{
        return user_pool.empty();
    }

    [[nodiscard]] int pool_size() const{
        return user_pool.size();
    }

};


// student struct
struct Students{
    vector<int> activities;
    set<int> variety;
    int activity_digit;
    int index;

    Students(int index, int activity_num){   // student index 学生编号,  activity count 活动数量
        this->index = index;
        this->activity_digit=0;
        static uniform_int_distribution<unsigned> u(0,9);
        static default_random_engine e;
        // 当种子和随机数生成器在循环内定义的时候，随机数生成的值会相同。static解决该问题
        for(int i = 0;i<activity_num;++i){
            int temp = u(e);
            this->activities.emplace_back(temp);
            this->variety.insert(temp);
            activity_digit|=(1<<temp);
        }
    }

    [[nodiscard]] int Get_Variety() const{
        cout<<"用户 "<<this->index<<"的不同活动的个数为:  "<<(int)this->variety.size()<<endl<<endl;
        return (int)this->variety.size();
    }

    ~Students()=default;

};


// Variety() 获取一个用户不同活动的数量
// Difference() 该方法用来和另一个用户的活动比较,返回不一致的活动数量
// Display() 展示所有用户的活动
// MinimumVariety() 确定所有用户的活动数量的最小差异
// MinimumDifference() 确定所有用户两两之间的最小差异
int Variety(Students* stu){
    return stu->Get_Variety();
}

int Difference(Students* t1, Students* t2){   //和另一个用户比较，输出不一致的活动数量
    int interval = t1->activity_digit^t2->activity_digit;
    int dif = 0;
    for(int k=0;k<t1->activities.size();++k){
        int temp = (t1->activities[k])==(t2->activities[k])?0:1;
        dif+=temp;
        //cout<<temp<<" ";
    }
    //cout<<endl;
    cout<< t1->index<<" and "<<t2->index <<" 之间的差异性是： "<<dif<<endl<<endl;
    return dif;
}

void Display(vector<Students*>& pool){   //展示所有用户活动
    for(auto &a:pool){
        cout<<a->index<<" : ";
        for(auto &b:a->activities){
            cout<<b<<" ";
        }
        cout<<endl;
    }
}

void MinimumVariety(vector<Students*>& pool){   //最小差异值
    int ans = INT_MAX;
    for(auto &a:pool){
        ans = min(ans,(int)a->variety.size());
    }
    cout<<"所有用户的活动数量的最小差异是: "<<ans/2<<endl<<endl<<endl;
}

void MinimumDifference(vector<Students*>& pool){ //所有用户之间两两最小差异
    int ans = INT_MAX;
    int stu_num = (int)pool.size();
    for(int i = 0;i<stu_num;++i){
        for(int j = i+1;j<stu_num;++j){
            cout<<"第"<<i<<"个用户和第"<<j<<"个用户差异如下:"<<endl;
            for(auto &a:pool[i]->activities)
                cout<<a<<" ";
            cout<<endl;
            for(auto &a:pool[j]->activities)
                cout<<a<<" ";
            cout<<endl;
            int dif = 0;
            for(int k=0;k<pool[i]->activities.size();++k){
                int temp = (pool[i]->activities[k])==(pool[j]->activities[k])?0:1;
                dif+=temp;
                cout<<temp<<" ";
            }
            cout<<endl;
            cout<<"二者差异性为: "<<dif<<endl<<endl;
            ans = min(ans, dif);
        }
    }
    cout<< "所有用户中两两用户最小差异为: "<<ans<<endl;
}

/*
int main() {
    UserPool<Students> pool;
    int user_num;
    int ac_num;
    cout<<"输入用户数:";
    cin>>user_num;
    cout<<"输入每个用户的活动数: ";
    cin>>ac_num;
    for(int i =0;i<user_num;++i){
        pool.add(std::make_unique<Students>(i,ac_num));
    }
    {
        int n = pool.pool_size();
        vector<Students*> users;
        while(n){
            auto p = pool.get();
            users.insert(users.begin(),p.get());
            p.release();
            --n;
        }
        cout<<"用户总数为: "<<users.size()<<endl;
        cout<<"用户的编号范围在 [ "<<0<<" , "<<(int)users.size()<<" ]"<<endl;

        Display(users);    //展示所有用户的活动

        int user_index;
        cout<<"输入想获取活动量的用户的编号: ";
        cin>>user_index;
        Variety(users[user_index]);   //获取一个用户不同活动的数量

        int user1,user2;
        cout<< "输入想比较的两个用户编号: "<<endl<<"user1: ";
        cin>>user1;
        cout<<"user2: ";
        cin>>user2;
        Difference(users[user1],users[user2]); // 该方法用来和另一个用户的活动比较,返回不一致的活动数量

        MinimumVariety(users);  // 确定所有用户的活动数量的最小差异
        MinimumDifference(users);  // 确定所有用户两两之间的最小差异

    }
    return 0;
}
*/