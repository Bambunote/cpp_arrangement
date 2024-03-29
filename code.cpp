#include <map>
#include <cmath>
#include <ctime>
#include <stack>
#include <queue>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>
#include <iostream>
#include <algorithm>
#include <windows.h>

#pragma GCC optimize(2)

//using namespace std;

const double eps=0.000001;
int gmax(int x,int y){return x>y?x:y;}
int gmin(int x,int y){return x<y?x:y;}
double gmax(double x,double y){return x>y?x:y;}
double gmin(double x,double y){return x<y?x:y;}
bool jmax(int x,int y){return x>y;}
bool jmin(int x,int y){return x<y;}
bool jmax(double x,double y){return x>y;}
bool jmin(double x,double y){return x<y;}
bool jieq(double x,double y){return fabs(x-y)<eps;}
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

class math_fuc{
public:
	const static int Math=120,Mod=1000000;
	int fac[Math],inv[Math],ifac[Math],C[Math][Math],can[Math];
	double sqr(int a){return a*a;}
	bool isprime(int n){
	    if(n==2||n==3)return true;
	    if(n<=1||(n%6!=1&&n%6!=5))return false;
	    for(int i=5;i*i<=n;i+=6)
	    	if(n%i==0||n%(i+2)==0)
	    		return false;
	    return true;
	}
	bool prime[Math],visit[Math];
    void make_prime(int n){
        int primesize=0;
        memset(prime,1,sizeof(prime));
        prime[1]=false;
        for(int i=2;i<=n;i++){
            if(prime[i])
                visit[++primesize]=i;
            for(int j=1;j<=primesize&&i*visit[j]<=n;j++){
                prime[i*visit[j]]=false;
                if(i%visit[j]==0)break;
            }
        }
    }
	int gcd(int x,int y){
		if(x%y==0)return y;
		else return gcd(y,x%y);
	}
	void factor(int mod,int n){
		fac[0]=1;
		for(int i=1;i<n;i++)
			fac[i]=fac[i-1]*i%mod;
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
		factor(Mod,Math);
		int ans=1,n;
		printf("Cantor.n:");
		scanf("%d",&n);
		printf("Cantor.array[]:");
		for(int i=1;i<=Math;i++)
			scanf("%d",&can[i]);
		for(int i=1;i<=Math;i++){
			int k=0;
			for(int j=i+1;j<=n;j++)
				if(can[j]<can[i])
					k++;
			ans+=k*fac[n-i];
		}
		printf("Cantor.no:%d",ans);
		en();
	}
	void decantor(){
		factor(Mod,Math);
		bool vis[Math]={false};
		int n,q,cnt,k,j,can[Math];
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
	void toBin(int n){
		int a[Math],i;
		while(n){
			a[i]=n&1;
			n=n>>1;i++;
		}
		for(i--;i>=0;i--)
			printf("%d",a[i]);
	}
	//ax+by=c
	int exgcd(int &x,int &y,int a,int b){
		if(!b){x=0;y=0;return a;}
		int r=exgcd(x,y,b,a%b);
		int t=y;y=x*(a/b);x=t;
		return r;
	}
	int exgcd_leinteger(int &x,int &y,int a,int b,int mod){
		int t=mod/gcd(mod,a);
		exgcd(x,y,a,b);
		x=(x%t+t)%t;
		return x;
	}
	void finv(int mod,int n){
		inv[1]=1;
		for(int i=2;i<n;i++)
			inv[i]=(mod-mod/i)*inv[mod%i]%mod;
	}
	void fifac(int mod,int n){
		finv(mod,n);
		ifac[0]=1;
		for(int i=1;i<=n;i++)
			ifac[i]=ifac[i-1]*inv[i]%mod;
	}
	int Combination(int n,int k,int mod){
		factor(mod,n);fifac(mod,n);
		return fac[n]*ifac[k]%mod*ifac[n-k]%mod;
	}
    int Catalan(int n,int mod){
        return Combination(2*n,n,mod)*inv[n+1]%mod;
    }
}math;

class range{
public:
	static const int N=2010;
	range(){n=0;memset(a,0,sizeof(a));}
	void merger(int l,int r,int *b,int& inv){
		if(l==r)return;
		int m=(l+r)/2,i=l,j=m+1,g=0;
		merger(l,m,b,inv);merger(m+1,r,b,inv);
		while(i<=m&&j<=r)
			if(a[i]<=a[j])b[++g]=a[i++];
			else b[++g]=a[j++],inv=inv+m-i+1;
		while(i<=m)b[++g]=a[i++];
		while(j<=r)b[++g]=a[j++];
		for(int i=1;i<=g;i++)a[i+l-1]=b[i];
	}
private:
	int a[N],n;
};

class bags{
private:
    void bag01(){
        scanf("%d%d",&m,&n);
    	for(int i=1;i<=n;i++)
    		scanf("%d%d",&w[i],&c[i]);
    	for(int i=1;i<=n;i++)
    		for(int j=m;j>=w[i];j--)
    			if(f[j-w[i]]+c[i]>f[j])
    				f[j]=f[j-w[i]]+c[i];
    	printf("%d",f[m]);
    }
    void bagfull(){
        scanf("%d%d",&m,&n);
        for(int i=1;i<=n;i++)
            scanf("%d%d",&w[i],&c[i]);
        for(int i=1;i<=n;i++)
            for(int j=w[i];j<=m;j++)
                f[j]=f[j-w[i]]+c[i]>f[j]?f[j-w[i]]+c[i]:f[j];
        printf("%d",f[m]);
    }
	void mix(){
		scanf("%d%d",&m,&n);
		for(int i=1;i<=n;i++)
			scanf("%d%d%d",&w[i],&c[i],&p[i]);
		for(int i=1;i<=n;i++)
			if(p[i]==0)
				for(int j=w[i];j<=m;j++)
					f[j]=gmax(f[j],f[j-w[i]]+c[i]);
			else
				for(int j=1;j<=p[i];j++)
					for(int k=m;k>=w[i];k--)
						f[k]=gmax(f[k],f[k-w[i]]+c[i]);
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

class ttree{
	static const int N=1200,Maxdepth=210;
private:
	std::vector<int> e[N];
	int n,t,tot,v[N],fir[N*2],nxt[N*2],to[N*2],dep[N],f[N][Maxdepth],maxdep,L[N],R[N];
	void init(int u,int fa){
		dep[u]=dep[fa]+1;
		maxdep=gmin(maxdep,dep[u]);
		for(int i=0;i<=Maxdepth;i++)
			f[u][i+1]=f[f[u][i]][i];
		for(int e=fir[u];e;e=nxt[e]){
			int v=to[e];
			if(v==fa)continue;
			f[v][0]=u;
			init(v,u);
		}
	}
public:
	ttree(){}
	ttree(std::vector<int> vec){
		t=tot=0;n=vec.size();
		maxdep=Maxdepth;
		memset(fir,0,sizeof(fir));
		memset(nxt,0,sizeof(nxt));
		memset(to,0,sizeof(to));
		memset(dep,0,sizeof(dep));
		memset(f,0,sizeof(f));
		for(int i=2;i<n;i++){
			int v=vec[i];
			tot++;
			nxt[tot]=fir[i];
			fir[i]=tot;
			to[tot]=v;
			tot++;
			nxt[tot]=fir[v];
			fir[v]=tot;
			to[tot]=i;
			e[i].push_back(v);
			e[v].push_back(i);
		}
		init(1,0);
	}
	int lca(int x,int y){
		if(dep[x]<dep[y])
			sswap(x,y);
		for(int i=maxdep;i>=0;i--){
			if(dep[f[x][i]]>=dep[y])
				x=f[x][i];
			if(x==y)
				return x;
		}
		for(int i=maxdep;i>=0;i--)
			if(f[x][i]!=f[y][i])
				x=f[x][i],y=f[y][i];
		return f[x][0];
	}
	int dis(int x,int y){
		return dep[x]+dep[y]-2*dep[lca(x,y)];
	}
	void dfs(int x,int pre,int d){
		L[x]=++tot;//dep[x]=d;
		int en=e[x].size();
		for(int i=0;i<en;i++){
			int y=e[x][i];
			if(y==pre)continue;
			dfs(y,x,d+1);
		}
		R[x]=tot;
	}
}tree;

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
		printf("operator: \n");
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

class segment_tree{
private:
	int a[120],b[480],sum[480],n,m,p,q,r,s;
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
		sum[k]+=v;
		b[k]+=(long long)v*(r-l+1);
	}
	void pushdown(int k,int l,int r,int mid){
		if(sum[k]==0)return;
		add(k*2,l,mid,sum[k]);
		add(k*2+1,mid+1,r,sum[k]);
		sum[k]=0;
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

class ddca{//double-direction connected array
private:
	struct node{
		int val,pre,nex;
	}node[121];
	int head,tail,tot,q;
	void ins(int p,int v){
		q=(++tot);
		node[q].val=v;
		node[node[p].nex].pre=q;
		node[q].nex=node[p].nex;
		node[p].nex=q;
		node[q].pre=p;
	}
	void del(int p){
		node[node[p].pre].nex=node[p].nex;
		node[node[p].nex].pre=node[p].pre;
		node[p].val=0;
	}
	void clr(){
		memset(node,0,sizeof(node));
		head=tail=1;tot=0;
	}
public:
	ddca(){
		memset(node,0,sizeof(node));
		tot=0;head=1;tail=1;
		node[head].nex=node[head].pre=tail;
		node[tail].nex=node[tail].pre=head;
	}
	void excute(){
		bool fl=true;
		while(fl){
			//1 v p  :: insert v after number p
			//2		 :: get the array
			//3 p    :: delete the node p
			//4		 :: clear the array
			//others :: exit
			char jud;scanf("%c",&jud);
			switch(jud){
				case '1':{
					int p,v;
					scanf("%d%d",&p,&v);
					ins(p,v);
					break;
				}
				case '2':{
					printf(">>>>\n");
					int in=head;
					do{
						printf("%d ",node[in].val);
						in=node[in].nex;
					}while(in!=head);
					printf("\n<<<<\n");
					break;
				}
				case '3':{
					int x;
					scanf("%d",&x);
					del(x);
					break;
				}
				case '4':{
					clr();
					break;
				}
				case '5':{
					fl=0;
					break;
				}
			}
		}
	}
};

class high{
public:
	high& operator=(const char*);
	high& operator=(int);
	int num[10010],cmp0;
	const static int carr=100000;

	high(){
		memset(num,0,sizeof(num));
		num[0]=1;cmp0=0;
	}
	high(int n){
		*this=n;
	}
    void outpt(){
        for(int i=this->num[0];i>=1;i--)
            printf("%d",this->num[i]);
        printf("\n");
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
		z.num[0]=gmax(x.num[0],y.num[0]);
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
    void input(){
        printf("n=");scanf("%d",&n);
        printf("x=");scanf("%d",&x);
        printf("A=\n");
        for(int i=1;i<=n;i++)
            scanf("%d",&a[i]);
        std::sort(a+1,a+1+n);
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
private:
	int a[1200],n,x;
};

class rrand{
public:
	int generate_short(){return rand();}//=[short]
	int generate_zone(int L,int R){return rand()%(R-L+1)+L;}//=[L,R]
	double generate_double(int eps){return rand()%eps/eps;}//=[0,1],Δ=1/eps
	int generate_int(){return (rand()<<15)+rand();}
	std::vector<int> generate_tree_fa(int n){
		std::vector<int> vec;vec[0]=n;vec[1]=-1;
		for(int i=2;i<=n;i++)
			vec.push_back(rand()%(i-1)+1);
		return vec;
	}
	struct edge{
		int x,y,val;
		friend bool operator< (edge p,edge q){
			return p.x==q.x?p.x<q.x:p.y<q.y;
		}
		edge(int qx,int qy,int qval){
			x=qx;y=qy;val=qval;
		}
	};
	std::vector<edge> generate_tree_side(int n){
		std::vector<int> tre=generate_tree_fa(n);
		std::vector<edge> vec;
		for(int i=1;i<n;i++){
			edge tmp(i+1,tre[i+1],rand());
			vec.push_back(tmp);
		}
		return vec;
	}
	std::vector<edge> generate_graph(const int n,int m){
		std::vector<edge> tre=generate_tree_side(n);
		bool vis[n][n]={0};
		for(int i=1;i<n;i++)
			vis[tre[i].x][tre[i].y]=true;
		for(int i=n;i<=m;i++){
			int u=rand()%n+1,v=rand()%n+1,val=rand();
			if(!vis[u][v]&&u!=v){
				edge tmp(u,v,val);
				tre.push_back(tmp);
			}
			else i--;
		}
		return tre;
	}
	std::vector<int> generate_shuffule(int n){
		std::vector<int> vec;
		for(int i=1;i<n;i++)
			vec.push_back(i);
		random_shuffle(vec.begin(),vec.end());
		return vec;
	}
	void checkanswer(){
		for(int i=1;i<=10000;i++){
			system("datamaker.exe");
			system("std.exe");
			system("mine.exe");
			if(system("fc std.out mine.out"))
				break;
		}
	}
};

class Hash{
public:
	const static int N=120,p1=1000000007,p2=99999989;
	int h1[N],h2[N],ft1[N],ft2[N];
	void init(std::string str,int mod){
		int n=str.length();
		for(int i=1;i<=n;i++){
			h1[i]=(h1[i-1]*p1+str[i]-'a')%mod;
			h2[i]=(h2[i-1]*p2+str[i]-'a')%mod;
		}
		ft1[0]=1;ft2[0]=1;
		for(int i=1;i<=n;i++){
			ft1[i]=ft1[i-1]*p1%mod;
			ft2[i]=ft2[i-1]*p2%mod;
		}
	}
	int calc1(int l,int r,int mod){return ((h1[r]-h1[l]*ft1[r-l+1])%mod+mod)%mod;}
	int calc2(int l,int r,int mod){return ((h2[r]-h2[l]*ft2[r-l+1])%mod+mod)%mod;}
	bool check(int l1,int r1,int l2,int r2,int mod){return calc1(l1,r1,mod)==calc1(l2,r2,mod)&&calc2(l1,r1,mod==calc2(l2,r2,mod));}
};

class Trie{
private:
	const static int N=120,M=27;//N stands for total , M stands for |c|
	int nxt[N][M],tot,rt;
public:
	Trie(){tot=0;rt=tot+1;memset(nxt,0,sizeof(nxt));}
	void insert(std::string s){
		int now=rt,n=s.length();
		for(int i=0;i<n;i++){
			int x=s[i]-'a'+1;
			if(!nxt[now][x]){
				nxt[now][x]=++tot;
				x=nxt[now][x];
			}
		}
	}
	bool query(std::string s){
		int now=rt,n=s.length();
		for(int i=0;i<n;i++){
			int x=s[i]-'a'+1;
			if(!nxt[now][x])return false;
			x=nxt[now][x];
		}
		return true;
	}
};

class unionset{
private:
    int n,fa[120];
    int find(int x){
        if(find(fa[x]!=x))fa[x]=find(fa[x]);
        return fa[x];
    }
    void uni(int x,int y){
        if(find(x)!=find(y))fa[y]=x;
    }
public:
    void init(){
        scanf("%d",&n);
        for(int i=1;i<=n;i++)
            fa[i]=i;
        int m;scanf("%d",&m);
        while(m--){
            int x,y;scanf("%d%d",&x,&y);
            uni(x,y);
        }
    }
};//union-find-disjoint set

class graph{
private:
	const static int N=120,M=1200,inf=100010;
	int n,m,g[N][N];
	struct node{
		int index,dist;
		bool operator<(const node &x)const{return dist>x.dist;}
	};
	struct edge{int u,v,nxt,w;}edg[M];
	int head[N],dis[N],cnt;
	bool vis[N];
public:
	void add(int u,int v,int w){
		edg[++cnt].u=u;
		edg[cnt].v=v;
		edg[cnt].w=w;
		edg[cnt].nxt=head[u];
		head[u]=cnt;
		return;
	}
	void init_matrix(){
		memset(g,0x7f,sizeof(g));
		n=read();m=read();
		bool both=true;
		for(int i=1;i<=n;i++)
			g[i][i]=0;
		for(int i=1;i<=m;i++){
			int u=read(),v=read(),w=read();
			if(both){g[u][v]=g[v][u]=w;}
			else g[u][v]=w;
		}
	}
	void init_forward_star(){
		memset(head,-1,sizeof(head));
		n=read();m=read();
		for(int i=1;i<=n;i++){
			int u=read(),v=read(),w=read();
			add(u,v,w);
		}
	}
	void floyd(){
		for(int k=1;k<=n;k++)
			for(int i=1;i<=n;i++)
				for(int j=1;j<=n;j++)
					if(g[i][k]+g[k][j]<g[i][j])
						g[i][j]=g[i][k]+g[k][j];
	}
	void dijkstra_heap(){
		std::priority_queue<node> q;
		memset(vis,0,sizeof(vis));
		memset(dis,0x7f,sizeof(dis));
		int s=read();
		dis[s]=0;node t={s,0};q.push(t);
		while(!q.empty()){
			node x=q.top();q.pop();
			int u=x.index;
			if(vis[u])continue;
			vis[u]=1;
			for(int i=head[u];i;i=edg[i].nxt)
				if(dis[edg[i].v]>dis[u]+edg[i].w){
					dis[edg[i].v]=dis[u]+edg[i].w;
					node t={edg[i].v,dis[edg[i].v]};
					q.push(t);
				}
		}
	}
	void Bellman_ford(){
		int s=read();
		memset(dis,0x3f,sizeof(dis));
		dis[s]=0;
		for(int k=1;k<=n-1;k++)
			for(int i=1;i<cnt;i++)
				if(dis[edg[i].v]>dis[edg[i].u]+edg[i].w)
					dis[edg[i].v]=dis[edg[i].u]+edg[i].w;
	}
	void SPFA(){
		memset(dis,0x3f,sizeof(dis));
		memset(vis,0,sizeof(vis));
		int s=read();dis[s]=0;vis[s]=1;
		std::queue<int> q;q.push(s);
		while(!q.empty()){
			int k=head[q.front()];
			while(k!=-1){
				if(dis[edg[k].v]>dis[edg[k].u]+edg[k].w){
					dis[edg[k].v]=dis[edg[k].u]+edg[k].w;
					if(!vis[edg[k].v]){
						q.push(edg[k].v);
						vis[edg[k].v]=true;
					}
				}
				k=edg[k].nxt;
			}
			q.pop();
		}
	}
	int minor_circle_floyd(){
		int ans=inf;
		for(int k=1;k<=n;k++)
			for(int i=1;i<k;i++)
				for(int j=i+1;j<k;j++)
					ans=gmin(ans,g[i][k]+g[k][j]+g[i][j]);
		return ans;
	}
};

class Linear_Basis{
public:
	const static int N=31;
	int b[N],nb[N],tot;
	Linear_Basis(){
		tot=0;
		memset(b,0,sizeof(b));
		memset(nb,0,sizeof(nb));
	}
	bool ins(int x){
		for(int i=N;i>=0;i--)
		if(x&(1<<i)){
			if(!b[i]){b[i]=x;break;}
			x^=b[i];
		}
		return x>0;
	}
	int Max(int x){
		int res=x;
		for(int i=62;i>=0;i--)
		res=gmax(res,res^b[i]);
		return res;
	}
	int Min(int x){
		int res=x;
		for(int i=0;i<=N;i++)
		if(b[i])res^=b[i];
		return res;
	}
	void rebuild(){
		for(int i=N;i>=0;i--)
			for(int j=i-1;j>=0;j--)
				if(b[i]&(1<<j))b[i]^=b[j];
			for(int i=0;i<=N;i++)
				if(b[i])nb[tot++]=b[i];
	}
	int Kth_Max(int k){
		int res=0;
		for(int i=N;i>=0;i--)
		if (k&(1<<i))res^=nb[i];
		return res;
	}
};

int main(){
	srand(time(0));//for rrand
	return 0;
}
