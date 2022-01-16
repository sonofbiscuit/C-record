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

    Students(int index, int activity_num){   // student index ѧ�����,  activity count �����
        this->index = index;
        this->activity_digit=0;
        static uniform_int_distribution<unsigned> u(0,9);
        static default_random_engine e;
        // �����Ӻ��������������ѭ���ڶ����ʱ����������ɵ�ֵ����ͬ��static���������
        for(int i = 0;i<activity_num;++i){
            int temp = u(e);
            this->activities.emplace_back(temp);
            this->variety.insert(temp);
            activity_digit|=(1<<temp);
        }
    }

    [[nodiscard]] int Get_Variety() const{
        cout<<"�û� "<<this->index<<"�Ĳ�ͬ��ĸ���Ϊ:  "<<(int)this->variety.size()<<endl<<endl;
        return (int)this->variety.size();
    }

    ~Students()=default;

};


// Variety() ��ȡһ���û���ͬ�������
// Difference() �÷�����������һ���û��Ļ�Ƚ�,���ز�һ�µĻ����
// Display() չʾ�����û��Ļ
// MinimumVariety() ȷ�������û��Ļ��������С����
// MinimumDifference() ȷ�������û�����֮�����С����
int Variety(Students* stu){
    return stu->Get_Variety();
}

int Difference(Students* t1, Students* t2){   //����һ���û��Ƚϣ������һ�µĻ����
    int interval = t1->activity_digit^t2->activity_digit;
    int dif = 0;
    for(int k=0;k<t1->activities.size();++k){
        int temp = (t1->activities[k])==(t2->activities[k])?0:1;
        dif+=temp;
        //cout<<temp<<" ";
    }
    //cout<<endl;
    cout<< t1->index<<" and "<<t2->index <<" ֮��Ĳ������ǣ� "<<dif<<endl<<endl;
    return dif;
}

void Display(vector<Students*>& pool){   //չʾ�����û��
    for(auto &a:pool){
        cout<<a->index<<" : ";
        for(auto &b:a->activities){
            cout<<b<<" ";
        }
        cout<<endl;
    }
}

void MinimumVariety(vector<Students*>& pool){   //��С����ֵ
    int ans = INT_MAX;
    for(auto &a:pool){
        ans = min(ans,(int)a->variety.size());
    }
    cout<<"�����û��Ļ��������С������: "<<ans/2<<endl<<endl<<endl;
}

void MinimumDifference(vector<Students*>& pool){ //�����û�֮��������С����
    int ans = INT_MAX;
    int stu_num = (int)pool.size();
    for(int i = 0;i<stu_num;++i){
        for(int j = i+1;j<stu_num;++j){
            cout<<"��"<<i<<"���û��͵�"<<j<<"���û���������:"<<endl;
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
            cout<<"���߲�����Ϊ: "<<dif<<endl<<endl;
            ans = min(ans, dif);
        }
    }
    cout<< "�����û��������û���С����Ϊ: "<<ans<<endl;
}

/*
int main() {
    UserPool<Students> pool;
    int user_num;
    int ac_num;
    cout<<"�����û���:";
    cin>>user_num;
    cout<<"����ÿ���û��Ļ��: ";
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
        cout<<"�û�����Ϊ: "<<users.size()<<endl;
        cout<<"�û��ı�ŷ�Χ�� [ "<<0<<" , "<<(int)users.size()<<" ]"<<endl;

        Display(users);    //չʾ�����û��Ļ

        int user_index;
        cout<<"�������ȡ������û��ı��: ";
        cin>>user_index;
        Variety(users[user_index]);   //��ȡһ���û���ͬ�������

        int user1,user2;
        cout<< "������Ƚϵ������û����: "<<endl<<"user1: ";
        cin>>user1;
        cout<<"user2: ";
        cin>>user2;
        Difference(users[user1],users[user2]); // �÷�����������һ���û��Ļ�Ƚ�,���ز�һ�µĻ����

        MinimumVariety(users);  // ȷ�������û��Ļ��������С����
        MinimumDifference(users);  // ȷ�������û�����֮�����С����

    }
    return 0;
}
*/