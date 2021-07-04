#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<istream>
#include<ostream>
#include<algorithm>
#include<windows.h>
#include<stack>
#include<queue>
#include<map>

using namespace std;

class sys_cons{
	public:
		double ins;
		int g_max(int x,int y){return x>y?x:y;}
		int g_min(int x,int y){return x<y?x:y;}
		double g_max(double x,double y){return x>y?x:y;}
		double g_min(double x,double y){return x<y?x:y;}
		bool j_max(int x,int y){return x>y;}
		bool j_min(int x,int y){return x<y;}
		bool j_max(double x,double y){return x>y;}
		bool j_min(double x,double y){return x<y;}
		bool j_ica(int x,int y){return x==y;}
		bool j_ica(double x,double y){return fabs(x-y)<0;}
		sys_cons(){ins=0.00000001;}
		void sl(){Sleep(500);}
		void cl(){system("cls");}
		void en(){puts("");}
		void sswap(int &x,int &y){int p=x;x=y;y=p;}
		inline bool isdigit(char x){return (x>='0'&&x<='9');}
		inline int read(){
			int x=0,flag=1;char c=getchar();
			while(!isdigit(c)){if(c=='-')flag=-1;c=getchar();}
			while(isdigit(c)){x=(x<<3)+(x<<1)+c-'0';c=getchar();}
			return flag*x;
		}
}cons;

class math_fuc{
	public:
		int fac[20];
		double sqr(int a){
			return a*a;
		}
		bool isprime(int n){
		    if(n==2||n==3)return true;
		    if(n<=1||(n%6!=1&&n%6!=5))return false;
		    for(int i=5;i*i<=n;i+=6)
		    	if(n%i==0||n%(i+2)==0)
		    		return false;
		    return true;
		}
		bool prime[101];
		void make_prime(int n){
			prime[0]=prime[1]=0;
			for(int i=2;i*i<=n;i++)
				if(prime[i])
					for(int j=2*i;j<n;j+=i)
						prime[j]=false;
			return;
		}
		int gcd(int x,int y){
			if(x%y==0)return y;
			else return gcd(y,x%y);
		}
		void factor(){
			fac[0]=1;
			for(int i=1;i<20;i++){
				fac[i]=fac[i-1]*i;
			}
		}
		int qpow(int a,int n,int mod){
			int ans=1,base=a%mod;
			while(n){
				if(n&1)ans=(ans*base)%mod;
				base=base*base%mod;
				n>>=1;
			}
			return ans;
		}
		void cantor(){
			factor();
			int ans=1,can[21],n;
			printf("Cantor.n:");
			scanf("%d",&n);
			printf("Cantor.array[]:");
			for(int i=1;i<=20;i++)
				scanf("%d",&can[i]);
			for(int i=1;i<=20;i++){
				int k=0;
				for(int j=i+1;j<=n;j++)
					if(can[j]<can[i])
						k++;
				ans+=k*fac[n-i];
			}
			printf("Cantor.no:%d",ans);
			cons.en();
		}
		void decantor(){
			factor();
			bool vis[21]={false};
			int n,q,cnt,k,j,can[21];
			printf("Cantor.n:");
			scanf("%d",&n);
			printf("Cantor.no:");
			scanf("%d",&k);
			for(int i=1;i<=n;i++){
				q=k/fac[n-i];
				k=k%fac[n-i];
				cnt=0;
				for(j=1;;j++){
					if(!vis[j])cnt++;
					if(cnt>q)break;
				}
				can[i]=j;
				vis[j]=true;
			}
			printf("Cantor.array[]:");
			for(int i=1;i<=n;i++)
				printf("%d ",can[i]);
		}
		int Fibo(int n){
			int a[3]={2,1,1};
			for(int i=1;i<(n+2)/3;i++){
				a[1]=a[2]+a[0];
				a[2]=a[1]+a[0];
				a[0]=a[1]+a[2];
			}
			return a[n%3];
		}
		double dist(int x1,int y1,int x2,int y2){
			return sqrt(sqr(x1-x2)+sqr(y1-y2));
		}
}math;

class sorts{
	public:
		int a[2<<10],b[2<<10],n,ans;
		bool excute(){
			puts("input n,a[]");
			puts("n<=1024,a[]::int");
			printf("n:");scanf("%d",&n);
			printf("a[]:");
			for(int i=1;i<=n;i++)
				scanf("%d",&a[i]);
			exch();Mergesort();
			exch();Quicksort();
			exch();Composort();
			return true;
		}
	private:
		void exch(){
			cons.sl();
			cons.sl();
			cons.cl();
			outpt();
		}
		void outpt(){
			printf("%d\n",n);
			for(int i=1;i<=n;i++)
				printf("%d ",a[i]);
			printf("\n\n");
		}
		void Mergesort(){
			int ans=0;
			merger(1,n);
			printf("There is %d inversions\n",ans);
			cons.en();
		}
		void merger(int l,int r){
			if(l==r)return;
			int i=l,m=(l+r)/2,j=m+1,g=0;
			merger(l,m);merger(m+1,r);
			while(i<=m&&j<=r)
				if(a[i]<=a[j])
					b[++g]=a[i++];
				else
					b[++g]=a[j++],ans=ans+m-i+1;
			while(i<=m)b[++g]=a[i++];
			while(j<=r)b[++g]=a[j++];
			for(int i=1;i<=g;i++)a[i+l-1]=b[i];
		}
		void Quicksort(){
			quicker(1,n);
		}
		void quicker(int l,int r){
			int i=l,j=r,m=a[(l+r)/2];
			do{
				while(a[i]<m)i++;
				while(a[j]>m)j--;
				if(i<=j)
					cons.sswap(a[i++],a[j--]);
			}while(i<=j);
			if(l<j)quicker(l,j);
			if(i<r)quicker(i,r);
		}
		void Composort(){
			puts("look out sort.png");
			return;
		}
};

class bags{
	private:
		void D1(){
			puts("Now let's use Mixer to solve the bags.");
			scanf("%d%d",&m,&n);
			for(int i=1;i<=n;i++)
				scanf("%d%d%d",&w[i],&c[i],&p[i]);
			for(int i=1;i<=n;i++)
				if(p[i]==0)
					for(int j=w[i];j<=m;j++)
						f[j]=cons.g_max(f[j],f[j-w[i]]+c[i]);
				else
					for(int j=1;j<=p[i];j++)
						for(int k=m;k>=w[i];k--)
							f[k]=cons.g_max(f[k],f[k-w[i]]+c[i]);
			printf("%d",f[m]);
		}
		void D2(){
			scanf("%d%d%d",&s,&t,&u);
			for(int i=1;i<=s;i++)
				scanf("%d%d%d",&a[i],&q[i],&r[i]);
			for(int i=1;i<=s;i++)
				for(int j=t;j>=0;j--)
					for(int k=u;k>=0;k--){
						int t1=j+q[i],t2=k+r[i];
						t1=t1>t?t:t1;
						t2=t2>u?u:t2;
						if(d[t1][t2]>d[j][k]+a[i])
							d[t1][t2]=d[j][k]+a[i];
					}
			printf("%d",d[t][u]);
			return;
		}
	public:
		int f[201],p[31],w[31],c[31],n,m;//1D
		int d[201][201],a[31],q[31],r[31],s,t,u;//2D
		/***************
		notes:
		[1D]
		n :: the number of things
		m :: the allowed weight
		w :: weight
		c :: price
		p :: ==0,Infinite
			 !=0,it_Numbers
		[2D]
		s  :: the number of things
		t :: the allowed weight1
		u :: the allowed weight2
		a :: price
		q :: weight1
		r :: weight2
		***************/
};

class sstack{
	public:
		sstack(){
			ttop=0;
			memset(sst,0,sizeof(sst));
		}
		bool spop(){
			if(!ttop){
				puts("Failed(A1)");
				return 0;
			}
			else{
				printf("%d",sst[ttop--]);
				cons.en();
				return 1;
			}
		}
		bool spush(){
			if(!(ttop-30)){
				puts("Failed(A2)");
				return 0;
			}
			else{
				int d;
				scanf("%d",&d);
				sst[ttop++]=d;
				return 1;
			}
		}
		void slen(){
			printf("%d",ttop);
		}
		bool sget(){
			if(!ttop){
				puts("NULL(A1)");
				return 0;
			}
			else{
				printf("%d",sst[ttop]);
				return 1;
			}
		}
	private:
		int sst[31],ttop;
};

class qqueue{
	public:
		qqueue(){
			qhead=qtail=qlen=0;
			memset(qqu,0,sizeof(qqu));
		}
		bool qpop(){
			if(!(qhead-qtail)){
				puts("Failed(A3)");
				return 0;
			}
			else{
				printf("%d",qqu[qhead--]);
				cons.en();
				qlen--;
				return 1;
			}
		}
		bool qpush(){
			if(qlen==100){
				puts("Failed(A4)");
				return 0;
			}
			else{
				int d;
				scanf("%d",&d);
				qqu[(qtail+1)%100]=d;
				qlen++;
				return 1;
			}
		}
		void qget(){
			printf("%d",qlen);
			cons.en();
		}
		bool qlon(){
			if(!qlen){
				puts("NULL(A3)");
				return 0;
			}
			else{
				printf("%d",qqu[qhead]);
				return 1;
			}
		}
	private:
		int qqu[102],qhead,qtail,qlen;
};

class tree_array{
	private:
		int a[120],b[120],n,m;
		int lb(int x){return x&(-x);}
		void update(int x,int val){
			for(;x<=n;x+=lb(x))
				b[x]+=val;
		}
		void query_update(int x,int y,int val){
			update(x,val);
			update(y+1,-val);
		}
		int getsum(int x){
			int res=0;
			for(;x>=1;x-=lb(x))
				res+=b[x];
			return res;
		}
		int query_ask(int x, int y){
			return getsum(y)-getsum(x-1);
		}
	public:
		void run(){
			printf("input n:");
			scanf("%d",&n);
			printf("input m:");
			scanf("%d",&m);
			printf("input array:\n");
			for(int i=1;i<=n;i++)
				scanf("%d",&a[i]);
			for(int i=1;i<=n;i++)
				update(i,a[i]-a[i-1]);
			printf("\n");
			printf("operater: \n");
			printf("1 a b 		: point a , number +b\n");
			printf("2 a b c 	: from a to b , number +c\n");
			printf("3 a b 		: from a to b , output sum\n");
			for(int i=1;i<=m;i++){
				int o,p,q,r;
				scanf("%d",&o);
				if(o==1){
					scanf("%d%d",&p,&q);
					query_update(p,p,q);
				}
				if(o==2){
					scanf("%d%d%d",&p,&q,&r);
					query_update(p,q,r);
				}
				if(o==3){
					scanf("%d%d",&p,&q);
					printf("%d\n",query_ask(p,q));
				}
			}
		}
};

class line_array{
	private:
		int a[120],b[480],c[480],n,m,p,q,r,s;
		void build(int k,int l,int r){
			if(l==r){
				b[k]=a[l];
				return;
			}
			int mid=(l+r)/2;
			build(k*2,l,mid);
			build(k*2+1,mid+1,r);
			b[k]=b[k*2]+b[k*2+1];
		}
		void add(int k,int l,int r,int v){
			c[k]+=v;
			b[k]+=(long long)v*(r-l+1);
		}
		void pushdown(int k,int l,int r,int mid){
			if(c[k]==0)return;
			add(k*2,l,mid,c[k]);
			add(k*2+1,mid+1,r,c[k]);
			c[k]=0;
		}
		void modify(int k,int l,int r,int x,int y,int v){ 
			if(l>=x&&r<=y)
				return add(k,l,r,v);
			int mid=(l+r)/2;
			pushdown(k,l,r,mid);
			if(x<=mid)
				modify(k*2,l,mid,x,y,v);
			if(mid<y)
				modify(k*2+1,mid+1,r,x,y,v);
			b[k]=b[k*2]+b[k*2+1];
		}
		int query(int k,int l,int r,int x,int y){
			if(l>=x&&r<=y)
				return b[k];
			int mid=(l+r)/2;
			long long res=0;
			pushdown(k,l,r,mid);
			if(x<=mid)
				res+=query(k*2,l,mid,x,y);
			if(mid<y)
				res+=query(k*2+1,mid+1,r,x,y);
			return res;
		}
	public:
		void run(){
			printf("input n:");
			scanf("%d",&n);
			printf("input m:");
			scanf("%d",&m);
			for(int i=1;i<=n;i++)
				scanf("%d",&a[i]);
			build(1,1,n);
			while(m--){
				scanf("%d%d%d",&p,&q,&r);
				if(p==1){
					scanf("%d",&s);
					modify(1,1,n,q,r,s);
				}
				else{
					printf("%d",query(1,1,n,q,r));
				}
			}
		}
};

class heap{
	private:
		int hheap[120],n,m,p,q,r,maxn;
		void put(int x){
			int now,next;
			hheap[++n]=x;
			now=x;
			while(now>1){
				if(hheap[now]>=hheap[next])
					break;
				cons.sswap(hheap[now],hheap[next]);
				now=next;
			}
		}
		int get(){
			int now=1,next,res=hheap[now];
			res=hheap[1];
			hheap[1]=hheap[n--];
			while(now*2<=n){
				next=now*2;
				if(next<n&&hheap[next+1]<hheap[next])
					next++;
				if(hheap[now]<=hheap[next])
					break;
				cons.sswap(hheap[now],hheap[next]);
				now=next;
			}
			return res;
		}
	public:
		void run(){
			printf("input the maxsize of the heap:");
			scanf("%d",&maxn);
			printf("input the operates' number:");
			scanf("%d",&m);
			n=1;
			memset(hheap,0,sizeof(hheap));
			for(int i=1;i<=m;i++){
				scanf("%d",&p);
				if(p==1){
					scanf("%d",&q);
					put(q);
				}
				else printf("%d\n",get());
			}
		}
};

class high{
public:
	high& operator=(const char*);
	high& operator=(int);
	int num[10010],cmp0,carr=100000;

	high(){
		memset(num,0,sizeof(num));
		num[0]=1;cmp0=0;
	}
	high(int n){
		*this=n;
	}
	friend int operator>(const high &x,const high &y){
		if(x.num[0]!=y.num[0])return (x.num[0]>y.num[0]?1:-1);
		for(int i=x.num[0];i>=1;i--)
			if(x.num[i]!=y.num[i])
				return (x.num[i]>y.num[i]);
		return 0;
	}
	friend high operator+(const high &x,const high &y){
		high z;
		z.num[0]=cons.g_max(x.num[0],y.num[0]);
		for(int i=1;i<=z.num[0];i++){
			z.num[i]=x.num[i]+y.num[i];
			if(z.num[i]>x.carr){
				z.num[i]=z.num[i]-x.carr;
				z.num[i+1]++;
			}
		}
		if(z.num[z.num[0]+1]>0)z.num[0]++;
		return z;
	}
	friend high operator-(const high &x,const high &y){
		high z;
		if(y>x){
			z=y-x;
			z.cmp0=-1;
			return z;
		}
		z.num[0]=x.num[0];
		for(int i=1;i<=z.num[0];i++){
			z.num[i]=x.num[i]-y.num[i];
			if(z.num[i]<0){
				z.num[i]=z.num[i]+x.carr;
				z.num[i+1]--;
			}
		}
		while(z.num[z.num[0]]==0&&z.num[0]>1)z.num[0]--;
		return z;
	}
	friend high operator*(const high &x,const high &y){
		high z;
		z.num[0]=x.num[0]+y.num[0]+1;
		for(int i=1;i<=x.num[0];i++)
			for(int j=1;j<=y.num[0];j++){
				z.num[i+j-1]+=x.num[i]*y.num[j];
				z.num[i+j]+=z.num[i+j-1]/x.carr;
				z.num[i+j-1]%=x.carr;
			}
		while(z.num[z.num[0]]==0&&z.num[0]>1)
			z.num[0]--;
		return z;
	}
	friend high operator/(const high &x,const high &y){
		high z1,z2;
		z1.num[0]=x.num[0]+y.num[0]+1;
		z2.num[0]=0;
		for(int i=x.num[0];i>=1;i--){
			memmove(z2.num+2,z2.num+1,sizeof(z2.num)-sizeof(int)*2);
			z2.num[0]++;
			z2.num[1]=x.num[i];
			int l=0,r=x.carr-1,mid;
			while(l<r){
				mid=(l+r)/2;
				if(!(y*high(mid)>z2))l=mid+1;
				else r=mid;
			}
			z1.num[i]=r-1;
			high p(r-1);
			z2=z2-y*p;
		}
		while(z1.num[z1.num[0]]==0&&z1.num[0]>1)
			z1.num[0]--;
		return z1;
	}
	friend high operator%(const high &x,const high &y){
		high z1,z2;
		z1.num[0]=x.num[0]+y.num[0]+1;
		z2.num[0]=0;
		for(int i=x.num[0];i>=1;i--){
			memmove(z2.num+2,z2.num+1,sizeof(z2.num)-sizeof(int)*2);
			z2.num[0]++;
			z2.num[1]=x.num[i];
			int l=0,r=x.carr-1,mid;
			while(l<r){
				mid=(l+r)/2;
				if(!(y*high(mid)>z2))l=mid+1;
				else r=mid;
			}
			z1.num[i]=r-1;
			high p(r-1);
			z2=z2-y*p;
		}
		while(z1.num[z1.num[0]]==0&&z1.num[0]>1)
			z1.num[0]--;
		return z2;
	}
	high qpow(high a,high n,high mod){
		high ans=1,base=a%mod,p=n;
		while(p>0){
			if((p%2>1)==0)ans=(ans*base)%mod;
			base=base*base%mod;
			n=n/2;
		}
		return ans;
	}
};
high& high::operator=(const char*c){
	memset(num,0,sizeof(num));
	int n=strlen(c),p1=1,p2=1;
	for(int i=1;i<=n;i++){
		if(p2==carr)p1++,p2=1;
			num[p1]+=p2*(c[n-i]-'0');
		p2*=10;
	}
	num[0]=p2;cmp0=1;
	return *this;
}
high& high::operator=(int x){
	char s[100010];
	sprintf(s,"%d",x);
	return *this=s;
}

class frac{
public:
	int son,mot;
	frac(){
		this->son=1;
		this->mot=1;
	}
	frac(int x,int y){
		this->son=x;
		this->mot=y;
	}
	friend frac simp(frac x){
		int gc=math.gcd(x.son,x.mot);
		return frac(x.son/gc,x.mot/gc);
	}
	friend frac operator-(const frac& x,const frac& y){
		return simp(frac(x.son*y.mot-y.son*x.mot,x.mot*y.mot));
	}
	friend bool operator<(const frac& x,const frac& y){
		frac z=x-y;
		return z.son<0;
	}
	friend frac absf(const frac& x){
		frac z=simp(x);
		z.son=abs(z.son);
		return z;
	}
};

class BinarySearch{
public:
	void excute(){
		input();
		se_x();
		se_lb();
		se_ub();
	}
private:
	int a[110],n,x;
	void input(){
		printf("n=");scanf("%d",&n);
		printf("x=");scanf("%d",&x);
		printf("A=\n");
		for(int i=1;i<=n;i++)
			scanf("%d",&a[i]);
	}
	//Search for x
	void se_x(){
		int l=0,r=n,mid=0;
		while(l<=r){
			mid=(l+r)/2;
			if(a[mid]==x){
				printf("it's the number %d\n",mid);
				return;
			}
			else if(a[mid]>x)r=mid-1;
			else l=mid+1;
		}
		printf("No x\n");
	}
	//Search for the smallest number bigger than x
	void se_ub(){
		int l=0,r=n,mid=0;
		while(l<=r){
			mid=(l+r)/2;
			if(a[mid]>x)r=mid-1;
			else l=mid+1;
		}
		printf("the smallest number bigger than x is the number %d\n",l);
	}
	//Search for the biggest number smaller than x
	void se_lb(){
		int l=0,r=n,mid=0;
		while(l<=r){
			mid=(l+r)/2;
			if(a[mid]<x)l=mid+1;
			else r=mid-1;
		}
		printf("the biggest number smaller than x is the number %d\n",r);
	}
};

int main(){
	return 0;
}