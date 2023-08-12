#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <bits/stdc++.h>
#include <unistd.h>
#include <pthread.h>

#include "sudoku.h"
using namespace std;

const int max_size=500000;
const int cpu_num=12;//cpu数量
int total=0;//读入的总数独
int boards[max_size][81];//162MB
bool done[max_size];//这题是否已经解出来，未解出来的题不能被打印，初始化为0
bool print[max_size];//这题是否已经打印，已打印的题才能被覆盖，初始化为1
bool game_end;//读取已经结束的标志
const int batch=100;//每次分题目给每个线程分多少题


int64_t now()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

struct node{
	int index;//下标
	int len;//往后计算多少个值
	node(int a,int b){//构造函数 
		index=a;
		len=b;
	}
};

struct my_queue{
	queue<node> q;//存储下标
	pthread_mutex_t lock;//访问队列的锁
	pthread_cond_t cond;//信号量，用于唤醒线程起来做题目
	
	my_queue(){//构造函数 
		pthread_mutex_init(&lock,NULL);
		pthread_cond_init(&cond,NULL);//初始化 
	}
	
	int len(){//队列中待计算的数量 
		return q.size();
	}
	
	void push(int index,int len){//把工作放到队列里面（由输入线程调用 
		pthread_mutex_lock(&lock);
		node tmp=node(index,len);
		q.push(tmp);
		pthread_cond_signal(&cond);//把这个线程叫起来工作
		pthread_mutex_unlock(&lock); 
	}
	
	node pop(){//从队列里面取出工作（由计算线程调用
		pthread_mutex_lock(&lock);
		while(q.size()==0)
			pthread_cond_wait(&cond,&lock);//没有活干了，睡觉 
		node tmp=q.front();
		q.pop(); 
		pthread_mutex_unlock(&lock);
		return tmp;
	}
	
};

my_queue queues[cpu_num];//有多少个cpu就开多少个核干活 

int in_sleep=0;//用于评估程序瓶颈
int out_sleep=0;

void* cin_thread(void *arg){
  char chs[100];//最长文件名为100字符
  int curr=0;//当前已派发下去的索引 
  while(cin.getline(chs,100)){

    FILE* fp = fopen(chs, "r");
    char puzzle[128];

    while (fgets(puzzle, sizeof puzzle, fp) != NULL) {	//每次读取一行数独
      if (strlen(puzzle) >= N) {  //成功读取一行
        //cout<<"读取一行"<<endl;
        while(print[total%max_size]==0){
          usleep(10000);//如果读得太快了，该块内存还没被打印，睡一下10ms
          in_sleep++;
        }
        input(puzzle,boards[total%max_size]);//将这一行读入boards中
        done[total%max_size]=print[total%max_size]=0;
        ++total;
        //cout<<total<<endl;
      }
      if(total-curr==batch){//攒满一个batch了
         my_queue * it=&queues[0];
         int Min=queues[0].len();
	  	 for(int i=1;i<cpu_num;i++){//检查哪个队列最空 
		    if(queues[i].len()<Min){
		    	Min=queues[i].len();
		    	it=&queues[i];
			}
		 }
		 it->push(curr,batch);//将这个批分配给这个线程
		 curr=total;
	  }
    }
    if(total!=curr){//最后还剩下一点没分配完的工作
      queues[0].push(curr,total-curr);//直接分配给第一个线程 
      curr=total;
      }
    fclose(fp);//关闭文件
  }

    //cout<<"分给第一个线程"<<endl;
  game_end=1;
  return NULL;
}


void *cout_thread(void *){
  int curr=0;//当前要输出的下标
  while(1){
    while(done[curr%max_size]==0){
      usleep(10000);//如果打印得太快了，CPU还没算出它的答案，睡10ms
      out_sleep++;
      if(game_end==1 && (curr==total))//如果睡了10ms后，已经结束了，退出线程
        {return NULL;}
    }
          for(int j=0;j<81;j++)
      cout<<boards[curr%max_size][j];//<<flush;
    cout<<endl;
    //if(done[curr%max_size]!=1 or print[curr%max_size]!=0) {cout<<"line134 error"<<endl;return NULL;}
    done[curr%max_size]=0;
    print[curr%max_size]=1;
    
    curr++;
  }
}

void* compute_thread(void* arg){//参数用于表示这个线程的编号，它将使用特定编号的队列 
  long long int id=(long long int) arg;
  while(1){
  	node tmp=queues[id].pop();//获得一份工作 
    //cout<<"获得工作"<<endl;
  	for(int i=tmp.index;i<tmp.index+tmp.len;i++){
  		solve_sudoku_dancing_links(boards[i%max_size]);//解决这个任务
  		done[i%max_size]=1;//标记已经解决 
      //cout<<"解决工作"<<endl;
	  }
  }
}



int main(int argc, char* argv[])
{
  //ios::sync_with_stdio(false);
  memset(done,0,max_size); //初始化
  memset(print,1,max_size);
  game_end=0;
  
  pthread_t p_in,p_out;
  pthread_create(&p_in,NULL,cin_thread,NULL);
  //cout<<"创建了输入线程\n";
  pthread_create(&p_out,NULL,cout_thread,NULL);
  //cout<<"创建了输出线程"<<endl;
  pthread_t p[cpu_num];
  for(long long int i=0;i<cpu_num;i++){
    //cout<<"i="<<i<<'\n';
	pthread_create(&p[i],NULL,compute_thread, (void *)i);
  }
  //cout<<"创建了计算线程"<<endl;
  pthread_join(p_out,NULL);//当p_out结束以后，说明程序结束了 
  //cout<<"in_sleep="<<in_sleep<<"\nout_sleep="<<out_sleep<<endl;
  //cout<<"total="<<total<<endl;
  //int64_t end = now();
  //double sec = (end-start)/1000000.0;
  //std::cout<<sec<<" sec "<< 1000*sec/total<<"ms each "<<total<<"\n";
  //printf("%f sec %f ms each %d\n", sec, 1000*sec/total, total);

  return 0;
}

