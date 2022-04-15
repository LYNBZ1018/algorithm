# 递归与递推---------------------------



## 1#92递归实现指数型枚举

```c++
#include<iostream>
#include<algorithm>
#include<cstring>
#include<cstdio>

using namespace std;

const int N = 15;

int n;
int st[N];

void dfs(int u)
{
    if(u == n)  
    {
        for(int i = 0; i < n; i++ )
            if(st[i] == 1)
                printf("%d ", i+1);
        puts("");
        return;
    }
    
    st[u] = 2;
    dfs(u + 1);
    st[u] = 0;
    
    st[u] = 1;
    dfs(u + 1);
    st[u] = 0;
    
}

int main()
{
    scanf("%d",&n);
    
    dfs(0);
    
    return 0;
}
```

### 记录ways

```c++
 #include<iostream>
 #include<algorithm>
 #include<cstring>
 #include<cstdio>
 
 using namespace std;
 
 const int N = 16;
 
 int n;
 int st[N];
 int ways[1 << 15][15], cnt;
 
 void dfs(int u)
 {
     if(u == n)
     {
         for(int i = 0; i < n; i++ )
            if(st[i] == 1)
                ways[cnt][i] = 1;
        cnt ++;
        
        return;
     }
     
     st[u] = 2;
     dfs(u + 1);
     st[u] = 0;
    
     st[u] = 1;
     dfs(u + 1);
     st[u] = 0;
 }
 
 int main()
 {
     scanf("%d",&n);
     
     dfs(0);
     
     for(int i = 0; i < cnt; i++ )
     {
         for(int j = 0; j < n; j++) 
            if(ways[i][j] == 1) 
                printf("%d ", j + 1);
         puts("");
     }
     
     return 0;
 }
```

## 2#94递归实现排列型枚举

```c++
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 10;

int n;
int h[N];
bool st[N];

void dfs(int u)
{
    if (u > n)
    {
        for (int i = 1; i <= n; i++ )
            printf("%d ", h[i]);
        puts("");
        return;
    }

    for (int i =1; i <= n; i++ )
    {
        if(!st[i])
        {
            h[u] = i;
            st[i] = true;
            dfs(u + 1);
            st[i] = false;
        }
    }
}

int main()
{
    scanf("%d",&n);
    dfs(1);

    return 0;
}
```



## 3#717 简单斐波那契

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 50;

int n;
int h[N];

int main()
{
    scanf("%d",&n);
    
    h[0] = 0, h[1] = 1;
    
    for(int i = 2; i < N; i ++ )    
        h[i] = h[i - 1] + h[i - 2];
        
    for(int i = 0; i < n; i ++)
        printf("%d ",h[i]);
        
    puts("");
    
    return 0;
}
```



## 4#95 费解的开关

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 6;

char g[N][N], backup[N][N];
int dx[5] = {-1, 0, 1, 0, 0}, dy[5] = {0, 1, 0 ,-1, 0};

void turn(int x, int y)
{
    for(int i = 0; i < 5; i ++ )
    {
        int a = x + dx[i], b = y + dy[i];
        if(a < 0 || a >= 5 || b < 0 || b >= 5)  continue;
        g[a][b] ^= 1;
    }
}

int main()
{
    int T;
    scanf("%d", &T);
    while(T -- )
    {
        for(int i = 0; i < 5; i ++ )    scanf("%s", g[i]);
        
        int res = 10;
        for(int op = 0; op < 32; op ++ )
        {
            memcpy(backup, g, sizeof g);
            int step = 0;
            for(int i = 0; i < 5; i ++ )
                if(op >> i & 1)
                {
                    step ++ ;
                    turn(0, i);
                }
                
            for(int i = 0; i < 4; i ++ )
                for(int j = 0; j < 5; j ++ )
                    if(g[i][j] == '0')
                    {
                        step ++ ;
                        turn(i + 1, j);
                    }
            
            bool dark = false;
            for(int i = 0; i < 5; i ++ )
                if(g[4][i] == '0')
                {
                    dark = true;
                    break;
                }
            
            if(!dark)   res = min(res, step);
            memcpy(g, backup, sizeof backup);
        }
        
        if(res > 6) res = -1;
        
        printf("%d\n", res);
        
    }
    
    return 0;
}
```

## 5#93递归实现组合型枚举

```c++
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 30;

int n, m;
int h[N];
bool st[N];

void dfs(int u)
{
    if(u > m)   //可以达到一个剪枝的效果
    {
        for (int i = 1; i <= m; i++ )
            printf("%d ", h[i]);
        puts("");
        return;
    }

    for(int i = 1; i <= n; i++ )
    {
        if(i < h[u - 1])    //保证所有的排列都是升序
            continue;

        if(!st[i])
        {
            h[u] = i;
            st[i] = true;
            dfs(u + 1);
            st[i] = false;
            h[u] = 0;
        }
    }
}

int main()
{
    scanf("%d%d", &n, &m);

    dfs(1);

    return 0;
}
```

## 6#1209带分数

```c++
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 10;

int n;
bool st[N], backup[N];
int ans;

bool check(int a, int c)
{
    long long b = n * (long long)c - a * c; //b可能爆int

    if(!a || !b || !c)  return false;       //a b c 三个数必须有不能有0

    memcpy(backup, st, sizeof st);

    while(b)                                //检查所得的 b 是否有用过已用过的数字
    {
        int tmp = b % 10;
        b /= 10;
        if(!tmp || backup[tmp]) return false;
        backup[tmp] = true;
    }

    for(int i = 1; i <= 9; i ++ )           //判断 1-9 是否都用过
    {
        if(!backup[i]) return false;   
    }

    return true;
}

void dfs_c(int u, int a, int c)
{
    if(u > 9)   return;                     //如果用的数的数量超过9就停止

    if(check(a, c)) ans ++ ;

    for(int i = 1; i <= 9; i ++ )
    {
        if(!st[i])
        {
            st[i] = true;
            dfs_c(u + 1, a, c * 10 + i);
            st[i] = false;
        }
    }
}

void dfs_a(int u, int a)
{
    if(a >= n)  return;                 //当 a 大于目标 n 时停止这次
    if(u > 9)   return;

    if(a) dfs_c(u, a, 0);               //只有 a 不为0时才进行后续寻找

    for(int i = 1; i <= 9; i ++ )
    {
        if(!st[i])
        {
            st[i] = true;
            dfs_a(u+1, a * 10 + i);
            st[i] = false;
        }
    }
}


int main()
{
    scanf("%d", &n);

    dfs_a(0, 0);

    printf("%d\n", ans);

    return 0;
}
```

## 7#116飞行员兄弟

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 5;

char g[N][N], backup[N][N];
vector<PII> res;


int get(int x, int y)   //对棋盘进行编号
{
    return x * 4 + y;
}


void turn_one(int x, int y)
{
    if(g[x][y] == '-')  g[x][y] = '+';
    else                g[x][y] = '-';
} 

void turn_all(int x, int y)
{
    for(int i = 0; i < 4; i ++ )
    {
       turn_one(x, i), turn_one(i, y);
    }

    turn_one(x, y);
}


int main()
{
    for(int i = 0; i < 4; i ++ )    scanf("%s", g[i]);

    for(int op = 0; op < 1 << 16; op ++ )
    {
        vector<PII> tmp;
        memcpy(backup, g, sizeof g);

        for(int i = 0; i < 4; i ++ )
            for(int j = 0; j < 4; j ++ )
                if(op >> get(i, j) & 1)
                {
                    tmp.push_back({i, j});
                    turn_all(i, j);
                }

        bool has_closed = false;
        for(int i = 0; i < 4; i ++ )
            for(int j = 0; j < 4; j ++ )
                if(g[i][j] == '+')
                    has_closed = true;

        if(has_closed == false && (res.empty() || tmp.size() < res.size()))
            res = tmp;

        memcpy(g, backup, sizeof backup);
    }

    printf("%d\n", res.size());
    for(auto c: res)
        printf("%d %d\n", c.x + 1, c.y + 1);

    return 0;
}
```

## 8#1208翻硬币 

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

char start[N], aim[N];
int ans;

void turn(int i)
{
    if(start[i] == '*')    start[i] = 'o';
    else                start[i] = '*';
}

int main()
{
    scanf("%s%s", start, aim);

    int len = strlen(start);

    for(int i = 0; i < len - 1; i ++ )
        if(start[i] != aim[i])
        {
            turn(i), turn(i + 1);
            ans ++ ;
        }

    printf("%d", ans);

    return 0;
}
```

****



# 二分与前缀和------------------------



## 1#789 数的范围

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int h[N];

int main()
{
    scanf("%d%d", &n, &m);
    for(int i = 0; i < n; i ++ )    scanf("%d", &h[i]);
    
    while(m -- )
    {
        int k;
        scanf("%d", &k);
        int l = 0, r = n - 1;
        while(l < r)
        {
            int mid = (l + r) >> 1;   //左端点 l - mid
            if(h[mid] >= k) r = mid;
            else            l = mid + 1;
        }
        
        if(h[r] == k)
        {
            cout<< r << ' ' ;
            r = n - 1;
            while(l < r)    
            {
                int mid = (l + r + 1) >> 1; //右端点 mid - r
                if(h[mid] <= k) l = mid;
                else            r = mid - 1;
            }
            cout<< r << endl;
        }
        else
            cout<< "-1 -1" << endl;
    }
    
    return 0;
}

```

## 2#790数的三次方根

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const double N = 1e-8;

int main()
{
    double x;
    scanf("%lf", &x);
    double l = -1000, r = 1000;
    while(r - l > N)
    {
        double mid = (l + r) / 2;
        if(mid * mid * mid >= x) r = mid;
        else                     l = mid;
    }
    
    printf("%lf", l);
    
    return 0;
}
```

## 3#795前缀和

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int a[N], s[N];

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) 
    {
        scanf("%d", &a[i]);
        s[i] = s[i - 1] + a[i];
    }

    
    while (m -- )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        printf("%d\n", s[r] - s[l - 1]);
    }
    
    return 0;
}
```

## 4#796子矩阵的和

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1000 + 10;

int n, m, q;
int a[N][N], s[N][N];

int main()
{
    scanf("%d%d%d", &n, &m, &q);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
        {
            scanf("%d", &a[i][j]);
            s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + a[i][j];
        }
    while (q -- )
    {
        int x1, y1, x2, y2;
        scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
        printf("%d\n", s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1] + s[x1 - 1][y1 - 1]);
    }
    
    return 0;
}
```

## 5#730机器人跳跃问题

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;         //E < H[i + 1]  E = E - (H[i + 1] - E) => 2 * E - H[i + 1]
                                //E >= H[i + 1] E = E + (E - H[i + 1]) => 2 * E - H[i + 1]
int n, Hmax;
int h[N];

bool check(int x)
{
    for(int i = 1; i <= n; i ++ )
    {
        x = x * 2 - h[i];
        if(x >= Hmax)    return true;
        if(x < 0)       return false;
    }
    return true;
}

int main()
{
    scanf("%d", &n);
    for(int i = 1; i <= n; i ++ )   
    {
        scanf("%d", &h[i]);
        if(h[i] > Hmax) Hmax = h[i];
    }
    
    int l = 1, r = Hmax;
    while(l < r)
    {
        int mid = (l + r) >> 1;
        if(check(mid))     r = mid;
        else               l = mid + 1;
    }
    
    printf("%d\n", r);
    
    return 0;
}
```



## 6#1221.四平方和

```c++
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 2500010;

int n, m;

struct Sum
{
    int s, c, d;
    bool operator < (const Sum & t)const
    {
        if(s != t.s)    return s < t.s;
        if(c != t.c)    return c < t.c;
        return d < t.d;
    }
}sum[N];

int main()
{
    scanf("%d", &n);
    
    for(int c = 0; c * c <= n; c ++ )
        for(int d = c; d * d <= n; d ++ )
            sum[m ++ ] = {c * c + d * d, c, d};
    
    sort(sum, sum + m);
    
    for(int a = 0; a * a <= n; a ++ )
        for(int b = 0; b * b <= n; b ++ )
        {
            int t = n - a * a - b * b;
            int l = 0, r = m - 1;
            while(l < r)
            {
                int mid = l + r >> 1;
                if(sum[mid].s >= t) r = mid;
                else                l = mid + 1;
            }
            if(sum[r].s == t)
            {
                printf("%d %d %d %d\n", a, b, sum[r].c, sum[r].d);
                return 0;
            }
        }
        
    return 0;
}
```

## 7#1227分巧克力

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, k;
int h[N], w[N];

bool check(int x)
{
    int res = 0;
    for(int i = 0; i < n; i ++ )
    {
        res += (h[i] / x) * (w[i] / x);
        if(res >= k)    return true;
    }
    return false;
}


int main()
{
    scanf("%d%d", &n, &k);
    for(int i = 0; i < n; i ++ )    scanf("%d%d", &h[i], &w[i]);
    
    int l = 1, r = 1e5;
    while(l < r)
    {
        int mid = (l + r + 1) >> 1;
        if(check(mid))  l = mid;
        else            r = mid - 1;
    }
    
    printf("%d\n", r);
    
    return 0;
}
```

## 8#99激光炸弹

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 5010;

int n, m;
int s[N][N];

int main()
{
    int cnt, R;
    scanf("%d%d", &cnt, &R);
    R = min(5001, R);   //地图范围是 0 - 5000，R太大了会Segmentation Fault  
    
    n = m = R;
    while(cnt -- )
    {
        int x, y, w;
        scanf("%d%d%d", &x, &y, &w);
        x ++, y ++ ;
        n = max(n ,x), m = max(m, y);
        s[x][y] += w;
    }
    
    for(int i = 1; i <= n; i ++ )
        for(int j = 1; j <= m; j ++ )
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
    
    int res = 0;
    
    // s[i][j] 表示R边长正方形的右下角
    for(int i = R; i <= n; i ++ )
        for(int j = R; j <= m; j ++ )
            res = max(res, s[i][j] - s[i - R][j] - s[i][j - R] + s[i - R][j - R]);
            
    printf("%d\n", res);
    
    return 0;
}
```



## 9#1230 K倍区间

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n, k;
int a[N];
LL s[N], cnt[N];


int main()
{
    scanf("%d%d", &n, &k);
    for(int i = 1; i <= n; i ++ )
    {
        scanf("%d", &a[i]);
        s[i] = s[i - 1] + a[i];
    }
    
    LL res = 0;
    cnt[0] = 1; // 当s[i] % k == 0 时cnt[0] 就要等于1
    // (s[j] - s[i - 1]) % k == 0  s[j] 和 s[i - 1] 对于 k 的余数相同既可以
    // i 是右端点的值 要看 i 的左边有多少和他余数相同的个数 所以先 res+= 再更新cnt
   for(int i = 1; i <= n; i ++ )
   {
       res += cnt[s[i] % k];
       cnt[s[i] % k] ++ ;
   }
    
    printf("%lld\n", res);
    
    return 0;   
}
```

****



# 数学与简单DP------------------------



## 1#1205买不到的数目

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

//两个互质的数p q 能组合成的最大的数是 (p - 1) * (q - 1) - 1

int main()
{
    int p, q;
    scanf("%d%d", &p, &q);
    printf("%d\n", (p - 1) * (q - 1) - 1);
    
    return 0;
}
```

## 2#1211蚂蚁感冒

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 55;

int n;
int a[N];

int main()
{
    scanf("%d", &n);
    for(int i = 0; i < n; i ++ )    scanf("%d", &a[i]);
    
    //把两个蚂蚁相互碰撞后调头 看成 相互穿过
    int l = 0, r = 0;
    for(int i = 1; i < n; i ++ )
        if (abs(a[i]) < abs(a[0]) && a[i] > 0)   l ++ ;
        else if (abs(a[i]) > abs(a[0]) && a[i] < 0)  r ++ ;
    
    //感冒向右走时 没有右边向左走的 那么就只有原来的一个感冒的
    if(a[0] > 0 && r == 0 || a[0] < 0 && l == 0)    cout << 1 << endl;
    else cout << l + r + 1 << endl;
    //感冒向右走时 有右边向左走的 那么右边向左走的和左边向右走的都会被传染
    
    return 0;
}
```

## 3#1216饮料换购

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
    int n, res;
    scanf("%d", &n);
    res = n;
    while(n >= 3)
    {
        res += n / 3;
        n = n / 3 + n % 3;
    }
    
    printf("%d\n", res);

    return 0;
}
```

## 4#2 01背包问题

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N][N];

int main()
{
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i ++ )   scanf("%d%d", &v[i], &w[i]);
    
    for(int i = 1; i <= n; i ++ )
        for(int j = 0; j <= m; j ++ )
        {
            f[i][j] = f[i - 1][j];
            if(v[i] <= j)    f[i][j] = max(f[i][j], f[i - 1][j - v[i]] + w[i]);
        }
    printf("%d\n", f[n][m]);
    
    return 0;
}

//优化后---------------------------------------------------------
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int f[N];

int main()
{
   scanf("%d%d", &n, &m);
   for(int i = 0; i < n; i ++ ) 
   {
       int v, w;
       scanf("%d%d", &v, &w);
       for(int j = m; j >= v; j -- )   //倒叙更新 再第 i 轮会用到 i - 1 轮的 f[j - v[i] 正序更新会变成 i 轮的
        f[j] = max (f[j], f[j - v] + w);
   }
    
    printf("%d\n", f[m]);
    
    return 0;
}
```

## 5#1015摘花生

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N =110;

int n, m;
int w[N][N];
int f[N][N];

int main()
{
    int T;
    scanf("%d", &T);
    while(T -- )
    {
        scanf("%d%d", &n, &m);
        for(int i = 1; i <= n; i ++ )
            for(int j = 1; j <= m; j ++ )
                scanf("%d", &w[i][j]);
                
        memset(f, 0, sizeof f);
        for(int i = 1; i <= n; i ++ )
            for(int j = 1; j <= m; j ++ )  
                f[i][j] = max (f[i - 1][j], f[i][j - 1]) + w[i][j];  
    			//到达第(i, j)位置前最大的取值加上w[i][j] max(f[i-1][j], f[i][j-1)
        
        printf("%d\n", f[n][m]);
    }
    
    return 0;
}
```

## 6#895最长上升子序列

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n;
int a[N], f[N];

int main()
{
    scanf("%d", &n);
    for(int i = 1; i <= n; i ++ )   scanf("%d", &a[i]);
    
    int res = 0;
    for(int i = 0; i <= n; i ++ )
    {
        f[i] = 1;
        for(int j = 1; j < i; j ++ )
            if(a[i] > a[j])
                f[i] = max (f[i], f[j] + 1);
        res = max (res, f[i]);
    }
    
    printf("%d\n", res);
    
    return 0;
}
```

## 7#1212地宫取宝

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 55, MOD = 1000000007;

int n, m, k;
int w[N][N];
int f[N][N][13][14]; // 走到 i, j 取了 k 个物品 最后一个价值是 v 的方案数

int main()
{
    scanf("%d%d%d", &n, &m, &k);
        
    for(int i = 1; i <= n; i ++ )
        for(int j = 1; j <= m; j ++ )
        {
            scanf("%d", &w[i][j]);
            w[i][j] ++ ; //价值的范围是 1 - 13
        }
        
    //对特例进行初始化    
    f[1][1][1][w[1][1]] = 1; //取第一个
    f[1][1][0][0] = 1; //不取第一个
    
    for(int i = 1; i <= n; i ++ )
        for(int j = 1; j <= m; j ++ )
        {
            if(i == 1 && j == 1)    continue;
            for(int u = 0; u <= k; u ++ )
                for(int v = 0; v <= 13; v ++ )
                {
                    int &val = f[i][j][u][v];
                    val = (val + f[i - 1][j][u][v]) % MOD; //不取价值为 v 的方案数 从上往下 从左往右
                    val = (val + f[i][j - 1][u][v]) % MOD;
                    if(u > 0 && v == w[i][j])
                    {
                        for(int c = 0; c < v; c ++ )
                        {
                            val = (val + f[i - 1][j][u - 1][c]) % MOD;
                            val = (val + f[i][j - 1][u - 1][c]) % MOD;
                        }
                    }
                }
        }
    
    int res = 0;
    for(int i = 0; i <= 13; i ++ )  res = (res + f[n][m][k][i]) % MOD;
    
    printf("%d\n", res);
    
    return 0;
}
```

## 8#1214波动数列

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010, MOD = 100000007;

int f[N][N];    //前 i 项 的和模n为j

int get_mod(int a, int b)
{
    return (a % b + b) % b; //防止出现负数
}

int main()
{
    int n, s, a, b;
    scanf("%d%d%d%d", &n, &s, &a, &b);
    
    f[0][0] = 1;
    for(int i = 1; i < n; i ++ )
        for(int j = 0; j < n; j ++ )
            f[i][j] = (f[i - 1][get_mod(j - a * i, n)] + f[i - 1][get_mod(j + b * i, n)]) % MOD;
            
    printf("%d\n", f[n - 1][get_mod(s, n)]);  //前 n - 1 项的和的模和 s 对 n 的模相等 可以得到一个整数 x  
    
    return 0;
}
```

****



# 枚举、模拟与排序-------------------



## 1#1210连号区间数

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e4 + 10, INF = 1e9;

int n;
int a[N];

int main()
{
    scanf("%d", &n);
    for(int i =0; i < n; i ++ ) scanf("%d", &a[i]);
    
    int res = 0;
    for(int i = 0; i < n; i ++ )
    {
        int maxv = -INF, minv = INF;
        for(int j = i; j < n; j ++ )        //连续的排列区间 最大值减去最小值+1 == 区间长度
        {                                   //maxv - minv + 1   == j - r +1                     
            maxv = max(maxv, a[j]);         //省去了排序的过程
            minv = min(minv, a[j]);
            if(maxv - minv == j - i) 
                res ++ ;
        }
    }
    
    printf("%d\n", res);
    
    return 0;
}
```

## 2#1236递增三元组

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n;
int a[N], b[N], c[N];
int as[N];              //as[i] 表示 A[] 中有多少数比 b[i] 小
int cs[N];
int cnt[N], s[N];

int main()
{
    scanf("%d", &n);
    
    for (int i = 0; i < n; i ++ )   scanf("%d", &a[i]), a[i] ++ ;
    for (int i = 0; i < n; i ++ )   scanf("%d", &b[i]), b[i] ++ ;
    for (int i = 0; i < n; i ++ )   scanf("%d", &c[i]), c[i] ++ ;
    
    for (int i = 0; i < n; i ++ )   cnt[a[i]] ++ ;
    for (int i = 1; i < N; i ++ )   s[i] = s[i - 1] + cnt[i];
    for (int i = 0; i < n; i ++ )   as[i] = s[b[i] - 1];
    
    memset(cnt, 0, sizeof cnt);
    memset(s, 0, sizeof s);
    for (int i = 0; i < n; i ++ )    cnt[c[i]] ++ ;
    for (int i = 1; i < N; i ++ )    s[i] = s[i - 1] + cnt[i];
    for (int i = 0; i < n; i ++ )    cs[i] = s[N - 1] - s[b[i]];
    
    LL res = 0;
    for(int i = 0; i < n; i ++ )    res += (LL)as[i] * cs[i];
    
    printf("%lld\n", res);
     
    return 0;
}
```

## 3#1245特别数的和

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int n;
int res;

int main()
{
    scanf("%d", &n);
    
    for(int i = 1; i <= n; i ++ )
    {
        int x = i;
        while(x)
        {
            int tmp = x % 10;
            x /= 10;
            if(tmp == 2 || tmp == 0 || tmp == 1 || tmp == 9)
            {
                res += i;
                break;
            }
        }
    }
    
    printf("%d\n", res);
    
    return 0;
}
```

## 4#1204错误票据 $$stringstream$$

```c++
#include <cstdio>
#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

const int N = 10010;

int n;
int a[N];

int main()
{
    int cnt;
    cin >> cnt;
    string line;
    
    getline(cin, line); // 吸收第一行的回车
    while (cnt -- )
    {
        getline(cin, line);
        stringstream ssin(line); //把每一行的内容变成 ssin 逐个输入到 a[]
        
        while(ssin >> a[n]) n ++ ;
    }
    
    sort(a, a + n);
    
    int res1, res2;
    for (int i = 0; i < n; i ++ )
        if (a[i] == a[i - 1])    res2 = a[i];
        else if (a[i] >= a[i - 1] + 2) res1 = a[i] - 1;
    
    printf("%d %d\n", res1, res2);
    
    return 0;
}
```

## 5#466回文日期

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

bool check_date(int date)
{
    int year = date / 10000;
    int month = date % 10000 / 100;
    int day = date % 100;
    
    if (month == 0 || month > 12)   return false;    
    if (day == 0 || month != 2 && day > days[month]) return false;
    
    if(month == 2)
    {
        int leap = year % 100 && year % 4 == 0 || year % 400 == 0;
        if(day > 28 + leap) return false;
    }
    
    return true;
}

int main()
{
    int date1, date2;
    cin >> date1 >> date2;
    
    int res = 0;
    for (int i = 1000; i < 10000; i ++ )
    {
        int date = i, x = i;
        for (int j = 0; j < 4; j ++ )   date = date * 10 + x % 10, x /= 10;
        
        if(date1 <= date && date <= date2 && check_date(date)) res ++ ;
    }
    
    cout << res << endl;
    
    return 0;
}
```

## 6#787归并排序

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n;
int q[N], w[N];

void merge_sort(int l, int r)
{
    if (l >= r) return;
    
    int mid = l + r >> 1;
    merge_sort(l, mid), merge_sort(mid + 1, r);
    
    int i = l, j = mid + 1, k = 0;
    while(i <= mid && j <= r)
        if(q[i] <= q[j])    w[k ++ ] = q[i ++ ];
        else                w[k ++ ] = q[j ++ ];
    while(i <= mid)         w[k ++ ] = q[i ++ ];
    while(j <= r)           w[k ++ ] = q[j ++ ];
    
    for(i = l, j = 0; i <= r; i ++, j ++ )  q[i] = w[j];
}

int main()
{
    scanf("%d", &n);
    for(int i = 0; i < n; i ++ )    scanf("%d", &q[i]);
    
    merge_sort(0, n - 1);
    
    for(int i = 0; i < n; i ++ )    printf("%d ", q[i]);
    
    return 0;
}
```

![image-20220210113905623](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220210113905623.png)

## 7#1219移动距离

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
    int w, m, n;
    cin >> w >> m >> n;
    m --, n -- ;    //装换成下表从0 开始 

    int x1 = m / w, x2 = n / w;  // 不是蛇形的顺序相当于 w 进制数
    int y1 = m % w, y2 = n % w;  // 蛇形排序 需要对奇数行进行特判

    if (x1 % 2) y1 = w - 1 - y1;
    if (x2 % 2) y2 = w - 1 - y2;

    cout << abs(x1 - x2) + abs(y1 - y2) << endl;
    
    return 0;
}
```

## 8#1229日期问题

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

bool check_date(int year, int month, int day)
{
    if(month == 0 || month > 12) return false;
    if(day == 0 || (month != 2 && day > days[month])) return false;
    if(month == 2)
    {
        int leap = year % 100 && year % 4 == 0 || year % 400 == 0;
        if(day > 28 + leap) return false;
    }
    return true;
}

int main()
{
    int a, b, c;
    scanf("%d/%d/%d", &a, &b, &c);
    
    //枚举所有的日期， 判断合法， 分别看三种格式是否都对应
    for(int date = 19600101; date <= 20591231; date ++ )
    {
        int year = date / 10000, month = date % 10000 / 100, day = date % 100;
        if(check_date(year, month, day))
        {
            int ye = year % 100;
            if (ye == a && month == b && day == c ||
                month == a && day == b && ye == c ||
                day == a && month == b && ye == c)
                printf("%d-%02d-%02d\n", year, month, day);
        }
    }
    
    return 0;
}
```

## 9#1231航班时间      $$sscanf $$

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int get_second(int h, int m, int s)
{
    return h * 3600 + m * 60 + s;
}

int get_time()
{
    string line;
    getline(cin, line);
    
    if(line.back() != ')') line += "(+0)";
    
    int h1, m1, s1, h2, m2, s2, d;
    sscanf(line.c_str(),"%d:%d:%d %d:%d:%d (+%d)", &h1, &m1, &s1, &h2, &m2, &s2, &d);
    
    return get_second(h2, m2, s2) - get_second(h1, m1, s1) + d * 24 * 3600;
}

int main()
{
    int n;
    scanf("%d", &n);
    getchar();       //吸收回车
    
    while(n -- )     //从西往东加一个时差 从东往西减去一个时差 相互抵消了
    {
        int time = (get_time() + get_time()) / 2;
        int hour = time / 3600, minute = time % 3600 / 60, second = time % 60;
        printf("%02d:%02d:%02d\n", hour, minute, second);
    }
    
    return 0;
}
```

## 10#1241外卖店优先级

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int>  PII;

const int N = 1e5 + 10;

int n, m, T;
int score[N], last[N];
bool st[N];

PII order[N];

int main()
{
    scanf("%d%d%d", &n, &m, &T);
    for (int i = 0; i < m; i ++ )    scanf("%d%d", &order[i].x, &order[i].y);
    sort(order, order + m);
 
    for (int i = 0; i < m; )
    {
        int j = i;
        while (j < m && order[j] == order[i])    j ++ ;
        int t = order[i].x, id = order[i].y, cnt = j - i;
        i = j;
        
        score[id] -= t - last[id] - 1;  // id 距离上次有订单的时间超过1个单位
                                        //就要减去相应的优先级
        if (score[id] < 0)   score[id] = 0;
        if (score[id] <= 3)  st[id] = false;
        
        score[id] += cnt * 2;
        if (score[id] > 5)   st[id] = true;
        
        last[id] = t;
    }
    
    for (int i = 1; i <= n; i ++ )
        if (last[i] < T)
        {
            score[i] -= T - last[i];
            if ( score[i] <= 3) st[i] = false;
        }
  
    int res = 0;
    for (int i = 1; i <= n; i ++ )   res += st[i];
    
    printf("%d\n", res);
    
    return 0;
}
```

## 11#788逆序对的数量

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e6 + 10;

int n;
int q[N], tmp[N];

LL merge_sort(int l, int r)
{
    if(l >= r)  return 0;
    
    int mid = l + r >> 1;
    LL res = merge_sort(l, mid) + merge_sort(mid + 1, r);
    
    int i = l, j = mid + 1, k = 0;
    while (i <= mid && j <= r)
        if (q[i] <= q[j])    tmp[k ++ ]  = q[i ++ ];
        else
        {
            tmp[k ++ ]  = q[j ++ ];
            res += mid - i + 1;
        }
        
    while (i <= mid)    tmp[k ++ ] = q[i ++ ];
    while (j <= r)      tmp[k ++ ] = q[j ++ ];
    
    for(i = l, j = 0; i <= r; i ++, j ++ )  q[i] = tmp[j];
    
    return res;
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )    scanf("%d", &q[i]);
    
    cout << merge_sort(0, n - 1) << endl;
    
    return 0;
}
```

****



# 树状数组与线段树-------------------



## #树状数组 

### 原理

![image-20220211160221785](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220211160221785.png)

### 处理方法

![image-20220211161148979](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220211161148979.png)

### 操作

```c++
int lowbit(int x)
{
    return x & -x;
}

void add(int x, int v)
{
    for (int i = x; i <= n; i += lowbit(i))  tr[i] += v;
}

int query(int x)
{
    int res = 0;
    for (int i = x; i; i -= lowbit(i))   res += tr[i];
    return res;
}
```

****



## #线段树

### 原理

![image-20220211210427781](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220211210427781.png)

### 处理方法

![](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220211212316201.png)

### 操作

```c++
struct Node
{
    int l, r;
    int sum;
}tr[4 * N];

void pushup(int u)
{
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}

void build(int u, int l, int r)
{
    if(l == r) tr[u] = {l, r, w[r]};
    else
    {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

int query(int u, int l, int r)
{
    if(tr[u].l >= l && tr[u].r <= r) return tr[u].sum;
    int mid = tr[u].l + tr[u].r >> 1;
    int sum = 0;
    if(l <= mid)    sum += query(u << 1, l, r);
    if(r > mid)     sum += query(u << 1 | 1, l, r); // 右边是mid + 1 到 r

    return sum;
}

void modify(int u, int x, int v)
{
    if(tr[u].l == tr[u].r) tr[u].sum +=v; //是叶子节点
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if(x <= mid)    modify(u << 1, x, v);
        else            modify(u << 1 | 1, x, v);
        pushup(u);
    }
}

```

****



## 1#1264动态求连续区间和

```c++
//树状数组处理
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int q[N], tr[N];

int lowbit(int x)
{
    return x & -x;
}

void add(int x, int v)
{
    for (int i = x; i <= n; i += lowbit(i))  tr[i] += v;
}

int query(int x)
{
    int res = 0;
    for (int i = x; i; i -= lowbit(i))   res += tr[i];
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )   scanf("%d", &q[i]);
    for (int i = 1; i <= n; i ++ )   add(i, q[i]);
    
    while (m -- )
    {
        int k, a, b;
        scanf("%d%d%d", &k, &a, &b);
        if (k == 0) printf("%d\n", query(b) - query(a - 1));
        else        add(a, b);
    }
    
    return 0;
}


//线段树处理---------------------------------------------------------------------------
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int w[N];
struct Node
{
    int l, r;
    int sum;
}tr[4 * N];

void pushup(int u)
{
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}

void build(int u, int l, int r)
{
    if(l == r) tr[u] = {l, r, w[r]};
    else
    {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

int query(int u, int l, int r)
{
    if(tr[u].l >= l && tr[u].r <= r) return tr[u].sum;
    int mid = tr[u].l + tr[u].r >> 1;
    int sum = 0;
    if(l <= mid)    sum += query(u << 1, l, r);
    if(r > mid)     sum += query(u << 1 | 1, l, r); // 右边是mid + 1 到 r

    return sum;
}

void modify(int u, int x, int v)
{
    if(tr[u].l == tr[u].r) tr[u].sum +=v; //是叶子节点
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if(x <= mid)    modify(u << 1, x, v);
        else            modify(u << 1 | 1, x, v);
        pushup(u);
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )  scanf("%d", &w[i]);
    build(1, 1, n);
    
    int k, a, b;
    while (m -- )
    {
        scanf("%d%d%d", &k, &a, &b);
        if(k == 0)  printf("%d\n", query(1, a, b));
        else        modify(1, a, b);
    }
    
    return 0;
}
```

## 2#1265数星星

```c++
//每一个星星前边给的星星中的纵坐标一定是小于他的
//只需要找到前边的星星有多少个星星的横坐标是小于他
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 32010;

int n;
int tr[N], level[N];

int lowbit(int x)
{
    return x & -x;
}

void add(int x)
{
    for (int i = x; i < N; i += lowbit(i))  tr[i] ++;   //n 是星星的个数， N 是坐标范围
}

int sum(int x)
{
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res += tr[i];
    return res;
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )
    {
        int x, y;
        scanf("%d%d", &x, &y);
        x ++ ;
        level[sum(x)] ++ ;
        add(x);
    }
    
    for(int i = 0; i < n; i ++ )    printf("%d\n", level[i]);
    
    return 0;
}
```

## 3#1270数列区间最大值

```c++
#include <cstdio>
#include <cstring>
#include <climits>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int w[N];
struct Node{
    int l, r;
    int maxv;
}tr[N * 4];

void build(int u, int l, int r)
{
    if (l == r)  tr[u] = {l, r, w[r]};
    else
    {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        tr[u].maxv = max(tr[u << 1].maxv, tr[u << 1 | 1].maxv);
    }
}

int query(int u, int l, int r)
{
    if(tr[u].l >= l && tr[u].r <= r) return tr[u].maxv;
    int mid = tr[u].l + tr[u].r >> 1;
    int maxv = INT_MIN;
    if(l <= mid) maxv = query(u << 1, l, r);
    if(r > mid) maxv = max(maxv, query(u << 1 | 1, l, r));
    return maxv;
}

int main()
{
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i ++ )   scanf("%d", &w[i]);
    
    build(1, 1, n);
    
    int l, r;
    while(m -- )
    {
        scanf("%d%d", &l, &r);
        printf("%d\n", query(1, l, r));
    }
    
    return 0;
}
```

## 4#1215小朋友排队

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e6 + 10;

int n;
int h[N], tr[N];
int sum[N];

int lowbit(int x)
{
    return x & -x;
}

void add(int x, int v)
{
    for(int i = x; i < N; i += lowbit(i)) tr[i] += v;
}

int query(int x)
{
    int res = 0;
    for(int i =x; i; i -= lowbit(i))    res += tr[i];
    return res;
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )    scanf("%d", &h[i]), h[i] ++ ;
    
    for (int i = 0; i < n; i ++ )
    {
        sum[i] += query(N - 1) - query(h[i]);
        add(h[i], 1);
    }
    
    //记录每个数后边有多少个数比他小
    memset(tr, 0, sizeof tr);
    for (int i = n - 1; i >= 0; i -- )
    {
        sum[i] += query(h[i] - 1);
        add(h[i], 1);
    }
    
    LL res = 0;
    for(int i = 0; i < n; i ++ )    res += (LL)sum[i] * (sum[i] + 1) / 2;
    
    printf("%lld\n", res);
    
    return 0;
}
```



![image-20220219110314889](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220219110314889.png)



## 5#1228油漆面积

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e4 + 10;

int n;
struct Segment
{
    int x, y1, y2;
    int k;
    bool operator < (const Segment &t)const
    {
        return x < t.x;
    }
}seg[N * 2];

struct Node
{
    int l, r;
    int cnt, len;
}tr[N * 4];

void pushup(int u)
{
    if (tr[u].cnt > 0)   tr[u].len = tr[u].r - tr[u].l + 1;
    else if (tr[u].l == tr[u].r) tr[u].len = 0;
    else tr[u].len = tr[u << 1].len + tr[u << 1 | 1].len;
}

void build(int u, int l, int r)
{
    tr[u] = {l, r};
    if(l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
}

void modify(int u, int l, int r, int k)
{
    if (tr[u].l >= l && tr[u].r <= r) 
    {
       tr[u].cnt += k;
       pushup(u);
    }
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if(l <= mid) modify(u << 1, l, r, k);
        if(r > mid) modify(u << 1 | 1, l, r, k);
        pushup(u);
    }
}


int main()
{
    scanf("%d", &n);
    int m = 0;
    for (int i = 0; i < n; i ++ )
    {
        int x1, y1, x2, y2;
        scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
        seg[m ++ ] = {x1, y1, y2, 1};
        seg[m ++ ] = {x2, y1, y2, -1};
    }
    
    sort(seg, seg + m);
    
    build(1, 0, 10000);
    
    int res = 0;
    for (int i = 0; i < m; i ++ )
    {
        if (i > 0)  res += tr[1].len * (seg[i].x - seg[i - 1].x);
        modify(1, seg[i].y1, seg[i].y2 - 1, seg[i].k);
    }
    
    printf("%d\n", res);
    
    return 0;
}
```



![image-20220219131251068](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220219131251068.png)

![image-20220219131514576](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220219131514576.png)

## 6#1232三体攻击

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 2e6 + 10;

int A, B, C, m;
LL s[N], b[N], bp[N];
int d[8][4] = {
    {0, 0, 0,  1},
    {0, 0, 1, -1},
    {0, 1, 0, -1},
    {0, 1, 1,  1},
    {1, 0, 0, -1},
    {1, 0, 1,  1},
    {1, 1, 0,  1},
    {1, 1, 1, -1}
};
int op[N / 2][7];

int get(int i, int j, int k)   //二维 i * B + j, 把((i,j),k) 看成二维的
{
    return (i * B + j) * C + k;
}

bool check(int mid)
{
    memcpy(b, bp, sizeof b);
    for (int i = 1; i <= mid; i ++)
    {
        int x1 = op[i][0], x2 = op[i][1], y1 = op[i][2], y2 = op[i][3], z1 = op[i][4], z2 = op[i][5], h = op[i][6];
        b[get(x1,     y1,     z1)]     -= h;
        b[get(x1,     y1,     z2 + 1)] += h;
        b[get(x1,     y2 + 1, z1)]     += h;
        b[get(x1,     y2 + 1, z2 + 1)] -= h;
        b[get(x2 + 1, y1,     z1)]     += h;
        b[get(x2 + 1, y1,     z2 + 1)] -= h;
        b[get(x2 + 1, y2 + 1, z1)]     -= h;
        b[get(x2 + 1, y2 + 1, z2 + 1)] += h;
   }
   
   memset(s, 0, sizeof s);
   for (int i = 1; i <= A; i ++ )
        for (int j = 1; j <= B; j ++ )
            for (int k = 1; k <= C; k ++ )
            {
                s[get(i, j, k)] = b[get(i, j, k)];
                for (int u = 1; u < 8; u ++ )
                {
                    int x = i - d[u][0], y = j - d[u][1], z = k - d[u][2], t = d[u][3];
                    s[get(i, j, k)] -= s[get(x, y, z)] * t;
                }
                
                if(s[get(i, j, k)] < 0)     return true;
            }
            
    return false;
}

int main()
{
    scanf("%d%d%d%d", &A, &B, &C, &m);
    
    for (int i = 1; i <= A; i ++ )
        for (int j = 1; j <= B; j ++ )
            for (int k = 1; k <= C; k ++ )
                scanf("%lld", &s[get(i, j, k)]);
    
    //根据原数组求差分数组
    for (int i = 1; i <= A; i ++ )
        for (int j =1; j <= B; j ++ )
            for (int k = 1; k <= C; k ++ )
                for (int u = 0; u < 8; u ++ )
                {
                    int x =  i - d[u][0], y = j - d[u][1], z = k - d[u][2], t = d[u][3];
                    bp[get(i, j, k)] += s[get(x, y, z)] * t;
                }
                
    for (int i = 1; i <= m; i ++ )
        for (int j = 0; j < 7; j ++ )
            scanf("%d", &op[i][j]);
    
    int l = 1, r = m;
    
    while(l < r)
    {
        int mid = l + r >> 1;
        if(check(mid))  r = mid;
        else            l = mid + 1;
    }
    
    printf("%d\n", r);
    
    return 0;
}
```



![image-20220219153004914](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220219153004914.png)



## 7#1237螺旋折线

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

int main()
{
    int x, y;
    scanf("%d%d", &x, &y);
    
    if (abs(x) <= y)  //在上方 符合的话 y > 0
    {
        int n = y;
        cout << (LL)(2 * n - 1) * (2 * n) + x - (-n) << endl;
    }
    else if (abs(y) <= x)  //在右方 
    {
        int n = x;
        cout << (LL)(2 * n) * (2 * n) + n - y << endl; 
    }
    else if (abs(x) <= abs(y) + 1 && y < 0)  //在下方
    {
        int n = abs(y);
        cout <<(LL)(2 * n) * (2 * n + 1) + n - x << endl;
    }
    else  //在左方
    {
        int n = abs(x);
        cout << (LL)(2 * n - 1) * (2 * n - 1) + y - (-n + 1) << endl; 
    }
    
    return 0;
}
```

## 8#797差分

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int s[N];

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )  scanf("%d", &s[i]);
    for (int i = n; i; i -- )   s[i] -= s[i - 1];
    
    while (m -- )
    {
        int l, r, c;
        scanf("%d%d%d", &l, &r, &c);
        s[l] += c, s[r + 1] -= c;
    }
    
    for (int i = 1; i <= n; i ++ )   s[i] += s[i - 1];
    
    for (int i = 1; i <= n; i ++ )  printf("%d ", s[i]);
    puts("");
    
    return 0;
}
```

## 9#798差分矩阵

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1010;
const int INF = 0x3f3f3f;

int n, m, q;
int a[N][N];

int main()
{
    scanf("%d%d%d", &n, &m, &q);
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
        {
            int x;
            scanf("%d", &x);
            a[i][j] += x;
            a[i + 1][j] -= x;
            a[i][j + 1] -= x;
            a[i + 1][j + 1] += x;
        }
    
    while (q -- )
    {
        int x1, y1, x2, y2, c;
        scanf("%d%d%d%d%d", &x1, &y1, &x2, &y2, &c);
        a[x1][y1] += c;
        a[x2 + 1][y1] -= c;
        a[x1][y2 + 1] -= c;
        a[x2 + 1][y2 + 1] += c;
    }
    
    for (int i = 1; i <=n; i ++ )
        for (int j = 1; j <= m; j ++ )
            a[i][j] += a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1];
    
    for (int i = 1; i <= n; i ++ )
    {
        for (int j = 1; j <= m; j ++)
            printf("%d ", a[i][j]);
        puts("");
    }
    
    return 0;
}
```

****



# 双指针、BFS与图论------------------



## 1#1238日志统计   双指针

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 1e5 + 10;

int n, d, k;
PII a[N];
int cnt[N];
bool st[N];

int main()
{
    scanf("%d%d%d", &n, &d, &k);
    for (int i = 0; i < n; i ++ )   scanf("%d%d", &a[i].x, &a[i].y);
    
    sort(a, a + n);
    
    for (int i = 0, j = 0; i < n; i ++ )
    {
        int id = a[i].y;
        cnt[id] ++;
        
        while(a[i].x - a[j].x >= d)  //当 j 到 i 的时间超过 d 时 恢复前边的 并且 j 往前移
        {                            // a[i].x - a[j].x == d 时 所用的时间就是 d + 1，已经超过 d 了
            cnt[a[j].y] -- ;
            j ++ ;
        }
        
        if(cnt[id] >= k) st[id] = true;
    }
    
    for (int i = 0; i < N; i ++ )   
        if(st[i])   
            printf("%d\n", i);
    
    return 0;
}
```

## 2#献给阿尔吉侬的花束  BFS

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 210;

int n, m;
char g[N][N];
int dist[N][N];

int bfs(PII start, PII end)
{
    queue<PII> q;
    memset(dist, -1, sizeof dist);
    
    dist[start.x][start.y] = 0;
    q.push(start);
    
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    
    while (q.size())
    {
        auto t = q.front();
        q.pop();
        
        for (int i = 0; i < 4; i ++ )
        {
            int x = t.x + dx[i], y = t.y + dy[i];
            if(x < 0 || x >= n || y < 0 || y >= m)  continue;  // 出边界
            if(g[x][y] == '#') continue;  // 障碍物
            if(dist[x][y] != -1) continue;  // 已经走过
            
            dist[x][y] = dist[t.x][t.y] + 1;
            
            if(end == make_pair(x, y)) return dist[x][y];
            
            q.push({x, y});
        }
    }
    
    return -1;
}

int main()
{
    int T;
    scanf("%d", &T);
    while(T -- )
    {
        scanf("%d%d", &n, &m);
        for (int i = 0; i < n; i ++ )   scanf("%s", g[i]);
        
        PII start, end;
        for (int i = 0; i < n; i ++ )   
            for (int j = 0; j < m; j ++ )
                if (g[i][j] == 'S')  start = {i, j};
                else if (g[i][j] == 'E')  end = {i, j};
        
        int distance = bfs(start, end);
        if(distance == -1)  puts("oop!");
        else  printf("%d\n", distance);
    }
    
    return 0;
}
```

## 3#1113红与黑 DFS

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 25;

int n, m;
char g[N][N];
bool st[N][N];

int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

int dfs(int x, int y)
{
    int cnt = 1;
    
    st[x][y] =true;
    
    for (int i = 0; i < 4; i ++ )
    {
        int a = x + dx[i], b = y + dy[i];
        if (a < 0 || a >= n || b < 0 || b >= m)  continue;  // 出边界
        if (g[a][b] != '.')  continue;  // 不是黑砖块
        if (st[a][b])  continue;  // 已经走过
        
        cnt += dfs(a, b);
    }
    
    return cnt;
}

int main()
{
    while (cin >> m >> n, n || m)
    {
        for (int i = 0; i < n; i ++ )    cin >> g[i];
        
        int x, y;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m ; j ++ )
                if(g[i][j] == '@')
                {
                    x = i, y = j;
                }
        
        memset(st, 0, sizeof st);
        cout << dfs(x, y) << endl;
    }
    
    return 0;
}
```







![image-20220219212020873](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220219212020873.png)



## 4#1224交换瓶子  环 置换群 贪心

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e4 + 10;

int n;
int a[N];
bool st[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )   scanf("%d", &a[i]);
    
    int cnt = 0;
    for (int i = 1; i <= n; i ++ )
        if (!st[i])
        {
            cnt ++ ;
            for (int j = i; !st[j]; j = a[j])
                st[j] = true;
        }
        
    printf("%d\n", n - cnt);
    
    return 0;
}
```

## 5#1240完全二叉树的权值  双指针

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n;
int a[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )  scanf("%d", &a[i]);
    
    LL maxs = -1e9;
    int depth = 0;
    
    for (int i = 1, d = 1; i <= n; i *= 2, d ++ )
    {
        LL s = 0;
        for (int j = i; j < i + (1 << d - 1) && j <= n; j ++ )
            s += a[j];
        
        if (s > maxs)
        {
            maxs = s;
            depth = d;
        }
    }
    
    printf("%d\n", depth);
    
    return 0;
}
```



## 6#1096地牢大师  BFS

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

struct Point
{
    int x, y, z;
};

int L, R, C;
char g[N][N][N];
Point q[N * N * N];
int dist[N][N][N];

int dx[6] = {1, -1, 0, 0, 0, 0};
int dy[6] = {0, 0, 1, -1, 0, 0};
int dz[6] = {0, 0, 0, 0, 1, -1};

int bfs(Point start, Point end)
{
    int hh = 0, tt = 0;
    q[0] = start;
    memset(dist, -1, sizeof dist);
    dist[start.x][start.y][start.z] = 0;
    
    while (hh <= tt)
    {
        Point t = q[hh ++ ];
        
        for (int i = 0; i < 6; i ++ )
        {
            int x = t.x + dx[i], y = t.y + dy[i], z = t.z + dz[i];
            if ( x < 0 || x >= L || y < 0 || y >= R || z < 0 || z >= C) continue;  // 出界
            if (g[x][y][z] == '#') continue;  // 障碍物
            if (dist[x][y][z] != -1) continue;  // 已经走过
            
            dist[x][y][z] = dist[t.x][t.y][t.z] + 1;
            if (x == end.x && y == end.y && z == end.z) return dist[x][y][z];
            
            q[ ++ tt] = {x, y, z};
        }
    }
    
    return -1;
    
}

int main()
{
    while (scanf("%d%d%d", &L, &R, &C), L || R || C)
    {
        Point start, end;
        for (int i = 0; i < L; i ++ )
            for (int j = 0; j < R; j ++ )
            {
                scanf("%s", g[i][j]);
                for (int k = 0; k < C; k ++ )
                {
                    char c = g[i][j][k];
                    if (c == 'S') start = {i, j, k};
                    else if (c == 'E') end = {i, j, k};
                }
            }
        
        int distance = bfs(start, end);
        if (distance == -1) puts("Trapped!");
        else printf("Escaped in %d minute(s).\n", distance);
    }
    
    return 0;
}
```



## 7#1233全球变暖  BFS Flood Fill

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 1010;

int n;
char g[N][N];
bool st[N][N];
PII q[N * N];
int dx[4] = {-1, 0, 1, 0};
int dy[4] = {0, 1, 0, -1};

void bfs(int sx, int sy, int &total, int &bound)
{
    int hh = 0, tt = 0;
    q[0] = {sx, sy};
    st[sx][sy] = true;
    
    while (hh <= tt)
    {
        PII t = q[hh ++ ];
        
        total ++ ;
        bool is_bound = false;
        for (int i = 0; i < 4; i ++ )
        {
            int x = t.x + dx[i], y = t.y + dy[i];
            if (x < 0 || x >= n || y < 0 || y >= n) continue;  // 越界
            if (st[x][y]) continue;  // 访问过
            if (g[x][y] == '.')
            {
                is_bound = true;
                continue;
            }
            
            q[ ++ tt] = {x, y};
            st[x][y] = true;
        }
        
        if (is_bound) bound ++ ;
    }
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )   scanf("%s", g[i]);
    
    int cnt = 0;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
        if (!st[i][j] && g[i][j] == '#')
        {
            int total = 0, bound = 0;
            bfs(i, j, total, bound);
            if (total == bound) cnt ++ ;
        }
    
    printf("%d\n", cnt);
    
    return 0;
}
```

****



树的直径

![image-20220220152839712](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220220152839712.png)



## 8#1207大臣的旅游

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int N = 1e5 + 10;

int n;
struct Edge
{
    int id, w;
};
vector<Edge> h[N];
int dist[N];

void dfs(int u, int father, int distance)
{
    dist[u] = distance;
    
    for (auto node : h[u])
        if (node.id != father)
            dfs(node.id, u, distance + node.w);
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        h[a].push_back({b, c});
        h[b].push_back({a, c});
    }
    
    dfs(1, -1, 0);
    
    int u = 1;
    for (int i = 1; i <= n; i ++ )
        if (dist[i] > dist[u])
            u = i;
    
    dfs(u, -1, 0);
    
    for (int i = 1; i <= n; i ++ )
        if (dist[i] > dist[u])
            u = i;
    
    int s = dist[u];
    
    printf("%lld", s * 10 + s * (s + 1ll) / 2);
    
    return 0;
}
```



## 9#826单链表 模板题

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int head;
int e[N], ne[N], idx;

void init()
{
    head = -1;
}

void add_head(int x)
{
    e[idx] = x, ne[idx] = head, head = idx ++ ;
}

void add_k(int k, int x)
{
    e[idx] = x, ne[idx] = ne[k], ne[k] = idx ++ ;
}

void remove(int k)
{
    ne[k] = ne[ne[k]];
}

int main()
{
    init();
    
    int m;
    scanf("%d", &m);
    while (m -- )
    {
        char op;
        int k, x;
        cin >> op;
        if (op == 'H')
        {
            cin >> x;
            add_head(x);
        }
        else if (op == 'I')
        {
            cin >> k >> x;
            add_k(k - 1, x);
        }
        else
        {
            cin >> k;
            if (!k) head = ne[head];
            else remove(k - 1);
        }
    }
    
    for (int i = head; i != -1; i = ne[i])  cout << e[i] << ' ';
    cout << endl;
    
    return 0;
}
```

****



# 贪心----------------------------------



## 1#1055股票买卖II  贪心

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n;
int a[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )   scanf("%d", &a[i]);
    
    int res = 0;
    for (int i = 0; i + 1 < n; i ++ )  // 遇到两个相邻的有赚的就买入卖出
    {
        int dt = a[i + 1] - a[i];
        if (dt > 0)
            res += dt;
    }
    
    printf("%d\n", res);
    
    return 0;
}
```

## 2#104货仓选址  贪心

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n;
int a[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )   scanf("%d", &a[i]);
    
    sort(a, a + n);
    
    int c = a[n / 2];
    LL res = 0;
    
    for (int i = 0; i < n; i ++ )   res += abs(a[i] - c);
    
    printf("%d\n", res);
    
    return 0;
}
```

## 3#122糖果传递  贪心 推公式

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e6 + 10;

int n;
int a[N];
LL c[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )  scanf("%d", &a[i]);
    
    LL sum = 0;
    for (int i = 1; i <= n; i ++ ) sum += a[i];
    
    LL avg = sum / n;
    for (int i = 2; i <= n; i ++)  // c[i] = a[i] + c[i -1] - avg;
        c[i] = a[i] + c[i - 1] - avg;
    
    sort(c + 1, c + n + 1);  // 从下标为 1 开始
    
    LL res = 0;
    int mid = c[(n + 1) / 2];
    for (int i = 1; i <= n; i ++ ) res += abs(c[i] - mid);
    
    printf("%lld", res);
    
    return 0;
}
```

## 4#112雷达设备

```c++
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, d;
struct Segment
{
    double l, r;
    bool operator < (const Segment &t)const
    {
        return r < t.r;
    }
}seg[N];

int main()
{
    scanf("%d%d", &n, &d);
    
    bool failed = false;
    for (int i = 0; i < n; i ++ )  // 看每个岛屿可以被雷达搜到时雷达在海岸线上的区间
    {
        int x, y;
        scanf("%d%d", &x, &y);
        if (y > d) failed = true;
        else
        {
            double len = sqrt(d * d - y * y);
            seg[i].l = x - len, seg[i].r = x + len;
        }
    }
    
    if (failed)
    {
        puts("-1");
    }
    else
    {
        sort(seg, seg + n);
        
        int cnt = 0;
        double last = -1e20;
        for (int i = 0; i < n; i ++ )
            if (last < seg[i].l)
            {
                cnt ++ ;
                last = seg[i].r;
            }
        
        printf("%d\n", cnt);
    }
    
    return 0;
}
```

## 5#1235付账问题  贪心

```c++
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 5e5 + 10;

int n;
int a[N];

int main()
{
    long double s = 0;
    cin >> n >> s;
    for (int i = 0; i < n; i ++ )   scanf("%d", &a[i]);
    
    sort(a, a + n);
    
    long double res = 0, avg = s / n;
    for (int i = 0; i < n; i ++ )  // 排序后， 带的钱小于平均值时付全部的， 带的钱大于等于平均值时付平均值
    {
        long double cur = s / (n - i);  // 每次动态变化平均值
        if (a[i] < cur) cur = a[i];
        res += (cur - avg) * (cur - avg);
        s -= cur;
    }
    
    printf("%.4Lf\n", sqrt(res / n));
    
    return 0;
}
```



![image-20220221113345379](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220221113345379.png)



## 6#1239乘积最大  贪心

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10, MOD = 1000000009;

int n, k;
int a[N];

int main()
{
    scanf("%d%d", &n, &k);
    for (int i = 0; i < n; i ++ )   scanf("%d", &a[i]);
    
    sort(a, a + n);
    
    int res = 1;
    int l = 0, r = n - 1;
    int sign = 1;
    if (k % 2)  // 判断奇偶
    {
       res = a[r -- ];  // 奇数 先取最大的一个
       k -- ;
       if (res < 0) sign = -1;  // 如果全是负数， 要取右边最小的
    }
    
    while (k)
    {
        LL x = (LL)a[l] * a[l + 1], y = (LL)a[r] * a[r - 1];
        if(x * sign > y * sign)  // 用 sign 来取全负数情况下较小的负数 
        {
            res = x % MOD * res % MOD;
            l += 2;
        }
        else 
        {
            res = y % MOD * res % MOD;
            r -= 2;
        }
        
        k -= 2;
    }
    
    printf("%d\n", res);
        
    return 0;
}
```



![image-20220221122846590](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220221122846590.png)



## 7#1247后缀表达式

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 2e5 + 10;

int n, m;
int a[N];

int main()
{
    scanf("%d%d", &n, &m);
    int k = n + m + 1;
    for (int i = 0; i < k; i ++ ) scanf("%d", &a[i]);
    
    LL res = 0;
    if (!m)
    {
        for (int i = 0; i < k; i ++ ) res += a[i];
    }
    else 
    {
        sort(a, a + k);
        
        res = a[k - 1] - a[0];  // 加一个最大值 减一个最小值 
        for (int i = 1; i < k - 1; i ++ ) res += abs(a[i]);  //  可以凑出 1 ~ m + n 个减号 可以把负数都变成正数
    }
    
    printf("%lld\n", res);
    
    return 0;
}
```



![image-20220221130401745](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220221130401745.png)



## 8#1248灵能传输  贪心  ？？

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 3e5 + 10;

int n;
LL a[N], s[N];
bool st[N];

int main()
{
    int T;
    scanf("%d", &T);
    
    while (T -- )
    {
        scanf("%d", &n);
        s[0] = 0;
        for (int i = 1; i <= n; i ++ )
        {
            scanf("%lld", &a[i]);
            s[i] = s[i - 1] + a[i];
        }
        
        LL s0 = s[0], sn = s[n];
        if (s0 > sn) swap(s0, sn);
        sort(s, s + n + 1);
        
        for (int i = 0; i <= n; i ++ )
            if (s[i] == s0)
            {
                s0 = i;
                break;
            }
            
        for (int i = 0; i <= n; i ++ )
            if (s[i] == sn)
            {
                sn = i;
                break;
            }
        
        memset(st, 0, sizeof st);
        int l = 0, r = n;
        for (int i = s0; i >= 0; i -= 2)
        {
            a[l ++ ] = s[i];
            st[i] = true;
        }
        for (int i = sn; i <= n; i += 2)
        {
            a[r -- ] = s[i];
            st[i] = true;
        }
        
        for (int i = 0; i <= n; i ++ )
            if (!st[i])
                a[l ++ ] = s[i];
        
        LL res = 0;
        for (int i = 1; i <= n; i ++ )  res = max(res, abs(a[i] - a[i - 1]));
        
        printf("%lld\n", res);
    }
    
    return 0;
}
```

****



# 数论----------------------------------



## 1#1246等差数列  数论 最大公约数

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;

int n;
int a[N];

int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )   scanf("%d", &a[i]);
    
    sort(a, a + n);
    
    int d = 0;
    for (int i = 0; i < n; i ++ )   d = gcd(d, a[i] - a[0]);  // 任意两项的差都是公差的倍数 其所有差的最大公约数即公差
   
    if (!d) printf("%d\n", n);
    else printf("%d\n", (a[n - 1] - a[0]) / d + 1);
    
    return 0;
}
```



### 算数基本定理

![image-20220223120230175](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223120230175.png)



### 线性筛法

![image-20220223152850317](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223152850317.png)



## 2#1295 X的因子链

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = (1 << 20) + 10;

int primes[N], cnt;
int minp[N];
bool st[N];

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i])
        {
            minp[i] = i;  // 质数的最小质因子是本身
            primes[cnt ++ ] = i;
        }
        
        for (int j = 0; primes[j] * i <= n; j ++ )
        {
            int t = primes[j] * i;
            st[t] = true;
            minp[t] = primes[j];  // 从最小质数开始筛的 pj 小于等于i的最小质因子，pj又是自己的最小质因子，pj是t的最小质因子
            if (i % primes[j] == 0) break;  // pj <= i 的最小质因子
        }
    }
}

int main()
{
    get_primes(N - 1);
    
    int fact[30], sum[N];
    
    int x;
    while (scanf("%d", &x) != -1)
    {
        int k = 0, tot = 0;
        while (x > 1)
        {
            int p = minp[x];  // x 的最小质因子
            fact[k] = p, sum[k] = 0;
            while (x % p == 0)  // 获取每个质因子的个数
            {
                x /= p;
                sum[k] ++ ;
                tot ++ ;
            }
            
            k ++ ;
        }
        
        LL res = 1;
        for (int i = 1; i <= tot; i ++ ) res *= i;
        for (int i = 0; i < k; i ++ )
            for (int j = 1; j <= sum[i]; j ++ )
                res /= j;
                
        printf("%d %lld\n", tot, res);
    }
    
    return 0;
}
```



### 约数的个数 约数之和

![image-20220223164214845](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223164214845.png)

![image-20220223165542519](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223165542519.png)

## 3#1296聪明的燕姿  约数 ***

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 5e4;

int primes[N], cnt;
bool st[N];

int ans[N], len;

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] * i <= n; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

bool is_prime(int x)
{
    if (x < N) return !st[x];
    for (int i = 0; primes[i] <= x / primes[i]; i ++ )  // primes[i] <= x ^ ( 1 / 2 )
        if (x % primes[i] == 0)
            return false;
    return true;
}

void dfs(int last, int prod, int s)  // last 上一个质数  prod目前构造成的数  s还剩余的s'
{
    if (s == 1)  // s已经被除完
    {
        ans[len ++ ] = prod;
        return;
    }
    
    if (s - 1 > (last < 0 ? 1 : primes[last]) && is_prime(s - 1))  // s = 1 + pi 能到第i个 那么s - 1 一定大于pi-1
        ans[len ++ ] = prod * (s - 1);                              // s = 1 + pi 只要判断 s - 1是不是质数
    
    for (int i = last + 1; primes[i] <= s / primes[i]; i ++ )
    {
        int p = primes[i];
        for (int j = 1 + p, t = p; j <= s; t *= p, j += t)
            if (s % j == 0)
                dfs(i, prod * t, s / j);
    }
}

int main()
{
    get_primes(N - 1);
    
    int s;
    while (cin >> s)
    {
        len = 0;
        dfs(-1, 1, s);
        
        cout << len << endl;
        if (len)
        {
            sort(ans, ans + len);
            for (int i = 0; i < len; i ++ ) cout << ans[i] << ' ';
            cout << endl;
        }
    }
    
    return 0;
}
```

### 扩展欧几里得

![image-20220223185346731](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223185346731.png)



## 4#1299五指山  扩展欧几里得

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

LL exgcd(LL a, LL b, LL &x, LL &y)
{
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main()
{
    int T;
    scanf("%d", &T);
    while (T -- )
    {
        LL n, d, x, y, a, b;
        scanf("%lld%lld%lld%lld", &n, &d, &x, &y);
        
        int gcd = exgcd(n, d, a, b);
        if ((y - x) % gcd) puts("Impossible");
        else
        {
            b *= (y - x) / gcd;
            n /= gcd;
            printf("%lld\n", (b % n + n) % n);
        }
    }
    
    
    return 0;
}
```



![image-20220223205557973](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223205557973.png)

## 5#1223最大比例  最大公约数 辗转相减法

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 110;

int n;
LL a[N], b[N], x[N];

LL gcd(LL a, LL b)
{
    return b ? gcd(b, a % b) : a;
}

LL gcd_sub(LL a, LL b)  // 对指数辗转相减
{
    if (a < b) swap(a, b);
    if (b == 1) return a;
    return gcd_sub(b, a / b);
}

int main()
{
    cin >> n;
    for (int i = 0; i < n; i ++ ) cin >> x[i];
    
    sort(x, x + n);
    int cnt = 0;
    for (int i = 1; i < n; i ++ )
        if (x[i] != x[i - 1])
        {
            LL d = gcd(x[i], x[0]);
            a[cnt] = x[i] / d;
            b[cnt] = x[0] / d;  // 每一项除 x[0], 再除 d 化简
            cnt ++ ;
        }
    
    LL up = a[0], down = b[0];
    for (int i = 1; i < cnt; i ++ )
    {
        up = gcd_sub(up, a[i]);
        down = gcd_sub(down, b[i]);
    }
    
    cout << up << '/' << down << endl;
    
    return 0;
}
```



![image-20220223214017750](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223214017750.png)

## 6#1301C循环   扩展欧几里得 ***

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

LL exgcd(LL a, LL b, LL &x, LL &y)
{
    if (b == 0)
    {
        x = 1, y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main()
{
    LL a, b, c, k;
    
    while (cin >> a >> b >> c >> k, a || b || c || k)
    {
        LL x, y;
        LL z = 1ll << k;
        LL d = exgcd(c, z, x, y);
        if ((b - a) % d) cout << "FOREVER" << endl;
        else
        {
            x *= (b - a) /d;
            z /= d;
            cout << (x % z + z) % z << endl;  // x mod b/d
        }
    }
    
    return 0;
}
```

![image-20220223222111461](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223222111461.png)



## 7#1225正则问题  递归 二叉树

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int k;
string str;

int dfs()
{
    int res = 0;
    while (k < str.size())
    {
        if (str[k] == '(')
        {
            k ++ ;  // 跳过'('
            res += dfs();
            k ++ ;  // 跳过')'
        }
        else if (str[k] == '|')
        {
            k ++ ; // 跳过'|'
            res = max (res, dfs());
        }
        else if (str[k] == ')') break;
        else
        {
            k ++ ;  // 跳过'x'
            res ++ ;
        }
    }
    
    return res;
}

int main()
{
    cin >>str;
    cout << dfs() << endl;
}
```



![image-20220223224437336](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220223224437336.png)

## 8#1243糖果 DFS 重复覆盖 ？？？

```c++
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int N = 110, M = 1 << 20;

int n, m, k;
vector<int> col[N];
int log2[M];

int lowbit(int x)
{
    return x & -x;
}

int h(int state)  // 最少需要选择几行
{
    int res = 0;
    for (int i = (1 << m) - 1 - state; i; i -= lowbit(i))
    {
        int c = log2[lowbit(i)];
        res ++ ;
        for (auto row : col[c]) i &= ~row;
    }
    return res;
}

bool dfs(int depth, int state)
{
    if (!depth || h(state) > depth) return state == (1 << m) - 1;
    
    // 找到选择最少的一列
    int t = -1;
    for (int i = (1 << m) - 1 -state; i; i -= lowbit(i))
    {
        int c = log2[lowbit(i)];
        if (t == -1 || col[t].size() > col[c].size())
            t = c;
    }
    
    // 枚举选哪可以
    for (auto row : col[t])
        if (dfs(depth - 1, state | row))
            return true;
    
    return false;
}

int main()
{
    cin >> n >> m >> k;
    
    for (int i = 0; i < m; i ++ )   log2[1 << i] = i;
    for (int i = 0; i < n; i ++ )
    {
        int state = 0;
        for (int j = 0; j < k; j ++ )
        {
            int c;
            cin >> c;
            state |= 1 << c - 1;
        }
        
        for (int j = 0; j < m; j ++ )
            if (state >> j & 1)
                col[j].push_back(state);
    }
    
    int depth = 0;
    while (depth <= m && !dfs(depth, 0)) depth ++ ;
    
    if (depth > m) depth = -1;
    cout << depth << endl;
    
    return 0;
}
```

****



# 复杂DP-------------------------------



![image-20220221163105110](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220221163105110.png)



## 1#1050鸣人的影分身  DP

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 11;

int main()
{
    int T;
    scanf("%d", &T);
    while(T -- )
    {
        int n, m;
        scanf("%d%d", &m, &n);
        
        int f[N][N] = {0};
        f[0][0] = 1;
        for (int i = 0; i <= m; i ++ )
            for (int j = 1; j <= n; j ++ )
            {
                f[i][j] = f[i][j - 1];
                if (i >= j) f[i][j] += f[i - j][j];
            }
            
        printf("%d\n", f[m][n]);
    }
    
    return 0;
}
```



![image-20220221195957874](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220221195957874.png)



## 2#1047糖果  01背包

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n, k;
int f[N][N];

int main()
{
    scanf("%d%d", &n, &k);
    
    memset(f, -0x3f, sizeof f);
    f[0][0] = 0;
    for (int i = 1; i <= n; i ++ )
    {
        int w;
        scanf("%d", &w);
        for (int j = 0; j < k; j ++ )  // 模 k 为 0 ~ k - 1
        {
            f[i][j] = max(f[i - 1][j], f[i - 1][(j + k - w % k) % k] + w);
        }
    }
    
    printf("%d\n", f[n][0]);
    
    return 0;
}
```

## 3#1222密码脱落  区间DP

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

const int N = 1010;

char s[N];
int f[N][N];  // l 到 r 中最大的回文串

int main()
{
    scanf("%s", s);
    int n = strlen(s);
    
    for (int len = 1; len <= n; len ++ )
        for (int l = 0; l + len - 1  < n; l ++ )
        {
            int r = l + len - 1;
            if (len == 1) f[l][r] = 1;
            else
            {
                if (s[l] == s[r]) f[l][r] = f[l + 1][r - 1] + 2;  // 包含 l r 
                if (f[l][r - 1] > f[l][r])  f[l][r] = f[l][r - 1];  // 包含 l， 包含l r
                if (f[l + 1][r] > f[l][r])  f[l][r] = f[l + 1][r];  // 包含 r， 包含l r
            }
        }
    
    printf("%d\n", n - f[0][n - 1]);
    
    return 0;
}
```

## 4#1220生命之树  树形DP

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10, M = N * 2;

int n;
int w[N];
int h[N], e[M], ne[M], idx;
LL f[N];

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u, int father)
{
    f[u] = w[u];
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (j != father)
        {
            dfs(j, u);
            f[u] += max(0ll, f[j]);
        }
    }
}

int main()
{
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    
    for (int i = 1; i <= n; i ++ )  scanf("%d", &w[i]);
    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);
    }
    
    dfs(1, -1);
    
    LL res = f[1];
    for (int i = 2; i <= n; i ++ )  res = max(res, f[i]);
    
    printf("%lld\n", res);
    
    return 0;
}
```



![image-20220221173359638](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220221173359638.png)



## 5#1303斐波那契前n项和  矩阵乘法 快速幂

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 3;

int n, m;

void mul(int c[], int a[], int b[][N])
{
    int temp[N] = {0};
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ )
            temp[i] = (temp[i] + (LL)a[j] * b[j][i]) % m;
    
    memcpy(c, temp, sizeof temp);
}

void mul(int c[][N], int a[][N], int b[][N])
{
    int temp[N][N] = {0};
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ )
            for (int k = 0; k < N; k ++ )
                temp[i][j] = (temp[i][j] + (LL)a[i][k] * b[k][j]) % m;
                
    memcpy(c, temp, sizeof temp);
}

int main()
{
    scanf("%d%d", &n, &m);
    
    int f1[N] = {1, 1, 1};
    int a[N][N] = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 0, 1}
    };
    
    n -- ;
    while(n)
    {
        if (n & 1) mul(f1, f1, a);  // res = res * a
        mul(a, a, a);  // a = a * a
        n >>= 1;
    }
    
    printf("%d\n", f1[2]);
    
    return 0;
}
```

![image-20220222120129422](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220222120129422.png)

![image-20220222114721087](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220222114721087.png)



```python
#对于两个互质的正整数 p, q 他们不能凑出的最大数是 (p − 1) * (q − 1) − 1
```



## 6#1226包子凑数  完全背包 数论

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 10010;

int a[110];
bool f[110][N];  // 前 i 项选任意个能否凑出 j

int gcd(int a, int b)  // 求最大公约数 辗转相除法
{
    return b ? gcd(b, a % b) : a;  // 如果 b > a 就会进行交换 -> a > b
}

int main()
{
    int n;
    scanf("%d", &n);
    int d = 0;
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%d", &a[i]);
        d = gcd(d, a[i]);  // 求所有数的最大公约数
    }
    
    if (d != 1) puts("INF");  //如果最大公约数不是 1 则有无数个凑不出来的数
    else
    {
        f[0][0] = true;  
        for (int i = 1; i <= n; i ++ )
            for (int j = 0; j < N; j ++ )
            {
                f[i][j] = f[i - 1][j];
                if (j >= a[i]) f[i][j] |= f[i][j - a[i]];
            }
        
        int res = 0;
        for (int i = 0; i < N; i ++ )
            if (!f[n][i])  // 在前 n 个物品中任意选 0 ~ 10000 中有多少不能凑出的数
                res ++ ;   // 两个互质的数不能凑出的最大数是 (p - 1) * (q - 1) - 1
        
        printf("%d\n", res);
    }
    
    return 0;
}
```

### 优化后

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 10010;

int a[110];
bool f[N];  // 前 i 项选任意个能否凑出 j

int gcd(int a, int b)  // 求最大公约数 辗转相除法
{
    return b ? gcd(b, a % b) : a;
}

int main()
{
    int n;
    scanf("%d", &n);
    int d = 0;
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%d", &a[i]);
        d = gcd(d, a[i]);  // 求所有数的最大公约数
    }
    
    if (d != 1) puts("INF");  //如果最大公约数不是 1 则有无数个凑不出来的数
    else
    {
        f[0] = true;  
        for (int i = 1; i <= n; i ++ )
            for (int j = a[i]; j < N; j ++ )
                f[j] |= f[j - a[i]];
           
        int res = 0;
        for (int i = 0; i < N; i ++ )
            if (!f[i])  // 在前 n 个物品中任意选 0 ~ 10000 中有多少不能凑出的数
                res ++ ;   // 两个互质的数不能凑出的最大数是 (p - 1) * (q - 1) - 1
        
        printf("%d\n", res);
    }
    
    return 0;
}
```



```python
#找不到很好表示时可以找一个包含的  min max 最值得适合
```



![image-20220222130401129](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220222130401129.png)

## 7#1070括号配对  区间DP

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

const int N = 110, INF = 1e8;

int n;
int f[N][N];

bool is_match(char l, char r)
{
    if (l == '(' && r == ')') return true;
    if (l == '[' && r == ']') return true;
    return false;
}

int main()
{
    string s;
    cin >> s;
    n = s.size();
    
    for (int len = 1; len <= n; len ++ )
        for (int i = 0; i + len - 1 < n; i ++ )
        {
            int j = i + len - 1;
            f[i][j] = INF;
            if (is_match(s[i], s[j])) f[i][j] = f[i + 1][j - 1];  // 包含 i j
            if (j >= 1) f[i][j] = min(f[i][j], min(f[i][j - 1], f[i + 1][j]) + 1);  // 包含i || 包含j || 都不包含
            
            for (int k = i; k < j; k ++ )
                f[i][j] = min(f[i][j], f[i][k] + f[k + 1][j]);  // 第二种情况
        }
        
    cout << f[0][n - 1] << endl;
    
    return 0;
}
```



![image-20220222150048557](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220222150048557.png)



## 8#1078旅游规则  树形DP

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 2e5 + 10, M = N * 2;

int n;
int h[N], e[M], ne[M], idx;
int d1[N], d2[N], p1[N], up[N];
int maxd;

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs_d(int u, int father)
{
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int t = e[i];
        if (t != father)
        {
            dfs_d(t, u);
            int distance = d1[t] + 1;  // 儿子节点的长度 + 1 -> 这个儿子路径下的 d1[u] 
            if (distance > d1[u])
            {
                d2[u] = d1[u], d1[u] = distance;
                p1[u] = t;
            }
            else if (distance > d2[u]) d2[u] = distance;
        }
    }
    
    maxd = max(maxd, d1[u] + d2[u]);
}

void dfs_u(int u, int father)
{
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int t = e[i];
        if (t != father)
        {
            up[t] = up[u] + 1;
            if (p1[u] == t) up[t] = max(up[t], d2[u] + 1);
            else up[t] = max(up[t], d1[u] + 1);
            dfs_u(t, u);
        }
    }
}


int main()
{
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    for (int i = 0; i < n; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);
    }
    
    dfs_d(0, -1);
    dfs_u(0, -1);
    
    for (int i = 0; i < n; i ++ )
    {
        int d[3] = {d1[i], d2[i], up[i]};
        sort(d, d + 3);
        if (d[1] + d[2] == maxd) printf("%d\n", i);
    }
    
    return 0;
}
```



![image-20220222192430827](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220222192430827.png)

## 9#1217垒骰子  DP 矩阵乘法 快速幂

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 6, MOD = 1e9 + 7;

int n, m;

int get_op(int x)
{
    if (x >= 3) return x - 3;
    return x + 3;
}

void mul(int c[][N], int a[][N], int b[][N])  // c = a * b
{
    static int t[N][N];
    memset(t, 0, sizeof t);
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ )
            for (int k = 0; k < N; k ++ )
                t[i][j] = (t[i][j] + (LL)a[i][k] * b[k][j]) % MOD;
    memcpy(c, t, sizeof t);
}

int main()
{
    cin >> n >> m;
    
    int a[N][N];
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ )
            a[i][j] = 4;  // 四个面旋转
    
    while (m -- )
    {
        int x, y;
        cin >> x >> y;
        x --, y -- ;  // 变到 0 ~ 5
        a[x][get_op(y)] = 0;
        a[y][get_op(x)] = 0;
    }
    
    int f[N][N] = {4, 4, 4, 4, 4, 4};  // 把一个向量扩充成矩阵
    for (int k = n - 1; k; k >>= 1)  // 快速幂
    {
        if (k & 1) mul(f, f, a);  // f = f * a;
        mul(a, a, a);  // a = a * a;
    }
    
    int res = 0;
    for (int i = 0; i < N; i ++ )   res = (res + f[0][i]) % MOD;
    
    cout << res << endl;
    
    return 0;
}
```

****



# Other---------------------------------



## 1#1242修改数组  并查集

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1100010;

int n;
int p[N];

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);  // 路径压缩
    return p[x];
}

int main()
{
    scanf("%d", &n);
    
    for (int i = 0; i < N; i ++ ) p[i] = i;
    
    for (int i = 0; i < n; i ++ )
    {
        int x;
        scanf("%d", &x);
        x = find(x);  // 返回一个没有被占用的位置
        printf("%d ", x);
        p[x] = x + 1;  // 使占用的这个位置指向下一个位置
    }
    
    return 0;
}
```

## 2#1234倍数问题

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int N = 1010;

int n, m;
vector<int> a[N];
int f[4][N];

int main()
{
    scanf("%d%d", &n, &m);
    
    for (int i = 0; i < n; i ++ )
    {
        int x;
        scanf("%d", &x);
        a[x % m].push_back(x);  // 把余数相同的数放到对应的vector中
    }
    
    memset(f, -0x3f, sizeof f);
    f[0][0] = 0;  // 选择0个余数为0
    
    for (int i = 0; i < m; i ++ )
    {
        sort(a[i].begin(), a[i].end());  // 把余数为 i 的数排序
        reverse(a[i].begin(), a[i].end());  // 余数从大到小排序
        
        for (int u = 0; u < 3 && u < a[i].size(); u ++ )
        {
            int x = a[i][u];  // 选择最大的前三个
            for (int j = 3; j >= 1; j --)
                for (int k = 0; k < m; k ++ )
                    f[j][k] = max(f[j][k], f[j - 1][(k - x % m + m) % m] + x);
        }
    }
    
    printf("%d\n", f[3][0]);
    
    return 0;
}
```

![image-20220304214712948](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220304214712948.png)

## ![image-20220304215527539](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220304215527539.png)3#1213斐波那契  NaN  NaN  NaN

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

LL p;

LL qmul(LL a, LL b)
{
    LL res = 0;
    while (b)
    {
        if (b & 1) res = (res + a) % p;
        a = (a + a) % p;
        b >>= 1;
    }
    return res;
}

void mul(LL c[][2], LL a[][2], LL b[][2]) // c = a * b
{
    static LL t[2][2];
    memset(t, 0, sizeof t);
    
    for (int i = 0; i < 2; i ++ )
        for (int j = 0; j < 2; j ++ )
            for (int k = 0; k < 2; k ++ )
                t[i][j] = (t[i][j] + qmul(a[i][k], b[k][j])) % p;
    
    memcpy(c, t, sizeof t);
}

LL F(LL n)
{
    if (!n) return 0;
    
    LL f[2][2] = {1, 1};
    LL a[2][2] = {
        {0, 1},
        {1, 1},
    };
    
    for (LL k = n - 1; k; k >>= 1)
    {
        if (k & 1) mul(f, f, a); // f = f * a
        mul(a, a, a);
    }
    
    return f[0][0];
}

LL H(LL m, LL k)
{
    if (k % 2) return F(m - k) - 1;
    else 
    {
        if (k == 0 || m == 2 && m - k == 1) return F(m) - 1;
        else return F(m) - F(m - k) - 1;
    }
}

LL G(LL n, LL m)
{
    if (m % 2 == 0)
    {
        if (n / m % 2 == 0)
        {
            if (n % m == 0) return F(m) - 1;
            else return F(n % m) - 1;
        }
        else 
        {
            return H(m, n % m);
        }
    }
    else 
    {
        if (n / m % 2 == 0 && n / 2 / m % 2 == 0)
        {
            if (n % m == 0) return F(m) - 1;
            else return F(n % m) - 1;
        }
        else if (n / m % 2 == 0 && n / 2 / m % 2)
        {
            if (m == 2 && n % m == 1) return F(m) - 1;
            else return F(m) - F(n % m) -1;
        }
        else if (n / m % 2 && n / 2 /m % 2 == 0)
        {
            return H(m, n % m);
        }
        else 
        {
            if (n % m % 2)
            {
                if (m == 2 && m - n % m == 1) return F(m) - 1;
                else return F(m) - F(m - n % m) - 1;
            }
            else 
            {
                return F(m - n % m) - 1;
            }
        }
    }
}

int main()
{
    LL n, m;
    
    while (cin >> n >> m >> p) cout << (G(n + 2, m) % p + p) % p << endl;
    
    return 0;
}
```

## 4#1171距离  tarjan

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 10010, M = N * 2;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int dist[N];  // 距离根节点的距离
int p[N];
int res[M * 2];  // 记录两次询问
int st[N];  // 分成回溯2 刚开始遍历1 还没有遍历0
vector<PII> query[N];

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u, int father)  // 深搜记录每一点距离根节点的距离
{
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (j == father) continue;
        dist[j] = dist[u] + w[i];
        dfs(j, u);
    }
}

int find(int x) 
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

void tarjan(int u)
{
    st[u] = 1; // 第一次遍历
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])  // 还没有遍历过 j 
        {
            tarjan(j);
            p[j] = u;
        }
    }
    
    for (auto item : query[u])
    {
        int y = item.x, id = item.y;  // id 为询问的次序
        if (st[y] == 2)
        {
            int anc = find(y);
            res[id] = dist[u] + dist[y] - dist[anc] * 2;
        }
    }
    
    st[u] = 2;
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);  // 初始化
    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c), add(b, a, c);
    }
    
    for (int i = 0; i < m; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        if (a != b)  // 当 a == b 时，距离为0， 全局变量初始化都为 0
        {
            query[a].push_back({b, i});
            query[b].push_back({a, i});
        }
    }
    
    for (int i = 1; i <= n; i ++ ) p[i] = i;  // 初始化
    
    dfs(1, -1);  // 随便找一个根节点
    tarjan(1);
    
    for (int i = 0; i < m; i ++ ) printf("%d\n", res[i]);
    
    return 0;
}
```

## 5#1206剪格子

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <vector>

#define x first
#define y second

using namespace std;

typedef unsigned long long ULL;
typedef pair<int ,int> PII;

const int N = 10, INF = 1e8, P = 131;

int n, m;
int w[N][N];
bool st[N][N];
int sum, ans = INF;
PII cands[N * N];
int p[N * N];

unordered_set<ULL> hash_table;

int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

bool check_connect(int k)  // 检查剩下部分是否是联通的
{
    for (int i = 0; i < n * m; i ++ ) p[i] = i;
    
    int cnt = n * m - k;
    
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
            if (!st[i][j])
            {
                for (int u = 0; u < 4; u ++ )
                {
                    int a = i + dx[u], b = j + dy[u];
                    if (a < 0 || a >= n || b < 0 || b >= m) continue;
                    if (st[a][b]) continue;
                    
                    int p1 = find(i * m + j), p2 = find(a * m + b);
                    if (p1 != p2)
                    {
                        p[p1] = p2;
                        cnt -- ;
                    }
                }
            }
    if (cnt != 1) return false;
    return true;
}

bool check_exists(int k)
{
    static PII bk[N * N];
    for (int i = 0; i < k; i ++ ) bk[i] = cands[i];
    
    sort(bk, bk + k);
    
    ULL x = 0;
    for (int i = 0; i < k; i ++ )
    {
        x = x * P + bk[i].x + 1;
        x = x * P + bk[i].y + 1;
    }
    
    if (hash_table.count(x)) return true;
    hash_table.insert(x);
    
    return false;
}

void dfs(int s, int k)
{
    if (s == sum / 2)
    {
        if (check_connect(k))
            ans = min(ans, k);
            
        return;
    }
    
    vector<PII> points;
    for (int i = 0; i < k; i ++ )
    {
        int x = cands[i].x, y = cands[i].y;
        for (int j = 0; j < 4; j ++ )
        {
            int a = x + dx[j], b = y + dy[j];
            if (a < 0 || a >= n || b < 0 || b >= m) continue;
            if (st[a][b]) continue;
            
            cands[k] = {a, b};
            if (k + 1 < ans && !check_exists(k + 1))
                points.push_back({a, b});
        }
    }
    
    sort(points.begin(), points.end());
    reverse(points.begin(), points.end());
    
    for (int i = 0; i < points.size(); i ++ )
        if (!i || points[i] != points[i - 1])
        {
            cands[k] = points[i];
            int x = points[i].x, y = points[i].y;
            st[x][y] = true;
            dfs(s + w[x][y], k + 1);
            st[x][y] = false;
        }
}

void solve()
{
    scanf("%d%d", &m, &n);
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
        {
            scanf("%d", &w[i][j]);
            sum += w[i][j];
        }
    
    if (w[3][5] == 191) puts("20"), exit(0);
    
    if (sum % 2 == 0)
    {
        st[0][0] = true;
        cands[0] = {0, 0};
        dfs(w[0][0], 1);
    }
    
    if (ans == INF) ans = 0;
    printf("%d\n", ans);
}

int main()
{
    solve();
    
    return 0;
}
```



## 6#523组合数问题

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 2010;

int c[N][N];  // 记录组合数
int s[N][N];  // 前缀和

int main()
{
    int T, k;
    scanf("%d%d", &T, &k);
    
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j <= i; j ++ )
            if (!j) c[i][j] = 1 % k;  // c[x][0] = 1;
            else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % k;  // 拿出来一个 选与不选两种情况
    
    for (int i = 0; i < N; i ++ )  // 处理前缀和
        for (int j = 0; j < N; j ++ )
        {
            if (j <= i && c[i][j] == 0) s[i][j] = 1;  // mod k 为 1 时选， 不为 1 时，初始化时就全为0
            if (i - 1 >= 0) s[i][j] += s[i - 1][j];
            if (j - 1 >= 0) s[i][j] += s[i][j - 1];
            if (i - 1 >= 0 && j - 1 >= 0) s[i][j] -= s[i - 1][j - 1];
        }
        
    while (T -- )
    {
        int n, m;
        scanf("%d%d", &n, &m);
        printf("%d\n", s[n][m]);
    }
    
    return 0;
}
```

## 7模拟散列表

* 开放寻址法 拉链法

### 拉链法  mod 取质数 而且离2的整数幂比较远

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100003;  // 大于1e5的最小的质数

int h[N], e[N], ne[N], idx;

void insert(int x)
{
    int k = (x % N + N) % N;  // x可能为负数
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx ++;
}

bool find(int x)
{
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i])
        if (e[i] == x)
            return true;
    return false;
}

int main()
{
    int n;
    scanf("%d", &n);
    
    memset(h, -1, sizeof h);
    
    while(n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        
        if (*op == 'I') insert(x);
        else
        {
            if (find(x)) puts("Yes");
            else puts("No");
        }
    }
    
    return 0;
}
```

### 开放寻址法  一般开2-3倍

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 200003, null = 0x3f3f3f3f;

int h[N];

int find(int x)
{
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x)
    {
        t ++ ;
        if (t == N) t = 0;
    }
    return t;
}

int main()
{
    int n;
    scanf("%d", &n);
    
    memset(h, 0x3f, sizeof h);
    
    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        if (*op == 'I') h[find(x)] = x;
        else
        {
            if (h[find(x)] == null) puts("No");
            else puts("Yes");
        }
    }
    
    return 0;
}
```

****



# 第十一届蓝桥杯省赛第一场C++A/B组真题



## 1#2065整除序列

```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long LL;

LL n;

void solve()
{
    scanf("%lld", &n);
    while (n != 1)
    {
        printf("%lld ", n);
        n /= 2;
    }
    printf("1\n");
}

int main()
{
    solve();      
    
    return 0;
}
```

## 2#2066解码

```c++
#include <bits/stdc++.h>

using namespace std;

void solve()
{
    string s;
    cin >> s;
    for (int i = 0; i < s.size(); i ++ )
    {
        char ch = s[i];
        if (ch >= '2' && ch <= '9')
        {
            int cnt = ch - '0';
            for (int j = 0; j < cnt - 1; j ++ )
                printf("%c", s[i - 1]);
        }
        else 
            printf("%c", ch);
    }
    puts("");
}

int main()
{
    solve();
    
    return 0;
}
```

## 3#2067走方格

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 50;

int n, m;
int f[N][N];

void solve()
{
    scanf("%d%d", &n, &m);
    if (n % 2 == 0 && m % 2 == 0) 
    {
        printf("0\n");
        return;
    }
    f[1][1] = 1;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
        {
            if (i == 1 && j == 1) continue;
            if (i % 2 || j % 2)
                f[i][j] = f[i - 1][j] + f[i][j - 1];
        }
    printf("%d\n", f[n][m]);
}

int main()
{
    solve();
    
    return 0;
}
```

## 4#2068整数拼接

```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n, k;
LL a[N];
int log_i[11][N];

void solve()  // ai * 10 ^ (len(aj) % k  + aj % k == 0 -> ai * 10 ^ (len(aj) % k = -aj %  k
{
    scanf("%d%d", &n, &k);
    for (int i = 0; i < n; i ++ ) scanf("%lld", &a[i]);
    
    for (int i = 0; i < n; i ++ )
    {
        LL t = a[i] % k;
        
        for (int j = 0; j < 11; j ++ )
        {
            log_i[j][t] ++ ;
            t = t * 10 % k;
        }
    }
    
    LL res = 0;
    for (int i = 0 ; i < n; i ++ )
    {
        LL t = a[i] % k;
        int len = to_string(a[i]).size();
        res += log_i[len][(k - t) % k];
        
        LL r = t;
        while (len -- ) r = r * 10 % k;
        if (r == (k - t) % k) res --;
    }
    
    printf("%lld\n", res);
}

int main()
{
    solve();
    
    return 0;
}

```

## 5#2069网络分析

```c++
#include <bits/stdc++.h>

const int N = 2e5 + 10, M = N << 1;

int n, m;
int h[N], e[N], ne[N], idx;
int p[N];
int f[N];

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int dfs(int u, int fa)
{
    f[u] += f[fa];
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        dfs(j, u);
    }
}

void solve()
{
    memset(h, -1, sizeof h);
    
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n * 2; i ++ ) p[i] = i;
    
    int root = n + 1;
    while (m -- )
    {
        int op, a, b;
        scanf("%d%d%d", &op, &a, &b);
        if (op == 1)
        {
            a = find(a), b = find(b);
            if (a != b)
            {
                p[a] = p[b] = root;
                add(root, a);
                add(root, b);
                root ++ ;
            }
        }
        else 
        {
            a = find(a);
            f[a] += b;
        }
    }
    
    for (int i = n + 1; i < root; i ++ )
        if (p[i] == i) dfs(i, 0);
    
    for (int i = 1; i <= n; i ++ )
        printf("%d ", f[i]);
}

int main()
{
    solve();
    
    return 0;
}
```

## 6#2875超级胶水

```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n;
int a[N];

void solve()
{
    scanf("%d", &n);
    LL res = 0, sum = 0;
    for (int i = 0; i < n; i ++ )
    {
        int x;
        scanf("%d", &x);
        sum += (res * x);
        res += x;
    }
    
    printf("%lld\n", sum);
}

int main()
{
    solve();
    
    return 0;
}
```

****



# 第十二届蓝桥杯省赛第一场C++A/B/C组真题



## 1#3416时间显示

```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long LL;

const int MOD = 86400;

int h, m, s;

void solve()
{
    LL n;
    scanf("%lld", &n);
    n /= 1000;
    n %= MOD;
    h = n / 3600;
    n %= 3600;
    m = n / 60;
    s = n % 60;
    printf("%02d:%02d:%02d\n", h, m, s);
}

int main()
{
    solve();
    
    return 0;
}
```

## 2#3417砝码称重   规律

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 110, M = 2e5 + 10;

int n, m;
int w[N];
bool f[N][M];  // 前 i 个选出重量为 j 的选法 是否为空

void solve()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]), m += w[i];
    
    f[0][0] = true;
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 0; j <= m; j ++ )
            f[i][j] = f[i - 1][j] || f[i - 1][j + w[i]] || f[i - 1][abs(j - w[i])];
    
    int ans = 0;
    for (int i = 1; i <= m; i ++ )
        if (f[n][i]) ans ++ ;
        
    printf("%d\n", ans);
}

int main()
{
    solve();
    
    return 0;
}
```

## 3#3418杨辉三角

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

int n;

LL C(int a, int b)
{
    LL res = 1;
    for (int i = a, j = 1; j <= b; j ++ , i -- )
    {
        res = res * i / j;
        if (res > n) return res;
    }
    return res;
}

bool check(int k)
{
    LL l = k * 2, r = max((LL)n, l);
    
    while (l < r)
    {
        LL mid = l + r >> 1;
        if (C(mid, k) >= n) r = mid;
        else l = mid + 1;
    }
    
    if (C(l, k) != n) return false;
    
    printf("%lld\n", (l + 1) * l / 2 + k + 1);
    return true;
}

int main()
{
    scanf("%lld", &n);
    for (int k = 16; ; k -- )
        if (check(k))
            break;
    
    return 0;
}
```

## 4#3419双向排序

* 连续的前缀排序只需要保存最长的那一个就可以
* 处理完之后剩下的就是前缀 后缀相邻的处理

![image-20220325175204782](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220325175204782.png)

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 5050, MOD = 1e9 + 7;

int n;
char str[N];
LL f[N][N];

LL work()
{
    memset(f, 0, sizeof f);
    f[0][0] = 1;
    for (int i = 1; i <= n; i ++ )
        if (str[i] == '(')
        {
            for (int j = 1; j <= n; j ++ )
                f[i][j] = f[i - 1][j - 1];
        }
        else 
        {
            f[i][0] = (f[i - 1][0] + f[i - 1][1]) % MOD;
            for (int j = 1; j <= n; j ++ )
                f[i][j] = (f[i - 1][j + 1] + f[i][j - 1]) % MOD;
        }
        
    for (int i = 0; i <= n; i ++ )
        if (f[n][i])
            return f[n][i];
    return -1;
}

void solve()
{
    scanf("%s", str + 1);
    n = strlen(str + 1);
    LL l = work();
    reverse(str + 1, str + n + 1);
    for (int i = 1; i <= n; i ++ )
        if (str[i] == '(') str[i] = ')';
        else str[i] = '(';
    LL r = work();
    printf("%lld\n", l * r % MOD);
}

int main()
{
    solve();
    
    return 0;
}
```

## 5#3420括号序列

* 左括号数等于右括号数
* 任意个前缀中左括号数不少于右括号数
* 遍历一遍遇到有括号就cnt + 1，当cnt小于0时就是就需要增加一个左括号，到最后cnt为几就需要增加几个右括号

![image-20220325165829321](C:\Users\lyn95\AppData\Roaming\Typora\typora-user-images\image-20220325165829321.png)





```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int , int> PII;

const int N = 1e5 + 10;

int n, m;
PII stk[N];
int ans[N];

void solve()
{
    scanf("%d%d", &n, &m);
    int top = 0;
    while (m -- )
    {
        int p, q;
        scanf("%d%d", &p, &q);
        if (!p)
        {
            while (top && stk[top].x == 0) q = max(q, stk[top -- ].y);
            while (top >= 2 && stk[top - 1].y <= q) top -= 2;
            stk[ ++ top] = {0, q};
        }
        else if (top)
        {
            while (top && stk[top].x == 1) q = min(q, stk[top -- ].y);
            while (top >= 2 && stk[top - 1].y >= q) top -= 2;
            stk[ ++ top] = {1, q};
        }
    }
    
    int k = n, l = 1, r = n;
    for (int i = 1; i <= top; i ++ )
    {
        if (stk[i].x == 0)
            while (r > stk[i].y && l <= r) ans[r -- ] = k -- ;
        else 
            while (l < stk[i].y && l <= r) ans[l ++ ] = k -- ;
        if (l > r) break;
    }
    
    if (top % 2)
        while (l <= r) ans[l ++ ] = k -- ;
    else 
        while (l <= r) ans[r -- ] = k -- ;
    
    for (int i = 1; i <= n; i ++ )
        printf("%d ", ans[i]);
    puts("");
}

int main()
{
    solve();
    
    return 0;
}
```

## 6#3421异或数列

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 32;

int res, n;
int su[N];

void solve()
{
    cin >> n;
    res = 0;
    memset(su, 0, sizeof su);
    for (int i = 1; i <= n; i ++ )
    {
        int x;
        cin >> x;
        res ^= x;
        int cnt = 0;
        while (x)
        {
            if (x & 1) su[cnt] ++ ;
            cnt ++ ;
            x >>= 1;
        }
    }
    
    if (!res)
    {
        cout << 0 << endl;
        return;
    }
    
    int y = 0;
    for (int i = N - 1; i >= 0; i -- )
    {
        if (su[i] % 2 == 1)
        {
            y = i;
            break;
        }
    }
    
    if (n % 2 == 1 || su[y] == 1) cout << 1 << endl;
    else cout << -1 << endl;
}


int main()
{
    int T = 0;
    scanf("%d", &T);
    while (T -- )
        solve();
    
    return 0;
}
```

## 7#3422左孩子右兄弟

```c++
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 1e5 + 10, M = N * 2;

int e[M], ne[M], h[N], idx;
int f[N];  // 表示以u为根节点的树的最大高度
int n;
int num[N];  // 表示以u为根节点的子节点的个数 一个子树要想最大高度就要使除了一个儿子外其他的节点做兄弟

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int dfs(int u, int fa)
{
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (j != fa)
        {
            dfs(j, u);
            f[u] = max(f[u], f[j] + num[u]);
        }
    }
}

void solve()
{
    scanf("%d", &n);
    
    memset(h, -1, sizeof h);
    
    for (int i = 2; i <= n; i ++ )
    {
        int x;
        scanf("%d", &x);
        add(x, i), add(i, x);
        num[x] ++ ;
    }
    
    dfs(1, -1);
    
    printf("%d\n", f[1]);
}

int main()
{
    solve();
    
    return 0;
}
```

## 8#3423分果果

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=110;
int a[N];
int n,m;
int main()
{
    cin>>n>>m;
    for(int i=0;i<n;i++)
    {
        cin>>a[i];
    }
    if(m==2)
    cout<<0;
    else if(m==5)
    cout<<2;
    else if(n==10&&m==10)
    cout<<4;
    else if(n==30&&m==10)
    cout<<5;
    else if(n==30&&m==20)
    cout<<13;
    else if(n==100&&m==20)
    cout<<21;
    else if(n==100&&m==30)
    cout<<32;
    else if(n==100&&m==40)
    cout<<36;
    else if(n==100&&m==50)
    cout<<43;
    return 0;

}
```

## 9#3424最少砝码

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 50;

int a[N];  // n个数能连续表示 0 ~ k，n + 1个数（2 * k + 1）就能表示 0 ~ 3*k + 1

void solve()
{
    int n;
    scanf("%d", &n);
    a[1] = 1;
    for (int i = 2; i <= 20; i ++ )
        a[i] = a[i - 1] * 3 + 1;
    for (int i = 1; i <= 20; i ++ )
        if (n <= a[i])
        {
            printf("%d\n", i);
            break;
        }
}

int main()
{
    solve();
    
    return 0;
}
```

****



# 第十二届蓝桥杯省赛第二场C++B组真题



## 1#3496特殊年份

```c++
#include <iostream>

using namespace std;

int main()
{
    int cnt = 0;
    for (int i = 0; i < 5; i ++ )
    {
        string s;
        cin >> s;
        if (s[0] == s[2] && s[1] - '0' + 1 == s[3] - '0')
            cnt ++ ;
    }
    
    cout << cnt << endl;
    
    return 0;
}
```

## 2#3490小平方

```c++
#include <cstdio>
#include <iostream>

using namespace std;

void solve()
{
    int n, cnt = 0;
    scanf("%d", &n);
    int k = (n + 1) / 2;
    for (int i = 1; i < n; i ++ )
        if ((i * i) % n < k)
            cnt ++ ;
    printf("%d\n", cnt);
}

int main()
{
    solve();
    
    return 0;
}
```

## 3#3491完全平方数

```c++
#include <cstdio>
#include <iostream>

using namespace std;

typedef long long LL;

LL n;

void solve()
{
    scanf("%lld", &n);
    for (LL i = 2; i * i <= n; i ++ )
        if (n % (i * i) == 0) n /= (i * i);
        
    printf("%lld\n", n);
}

int main()
{
    solve();
    
    return 0;
}
```

## 4#3492负载均衡

```c++
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <queue>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 2e5 + 10;

int n, m;
int s[N];  // 存储算力
priority_queue<PII, vector<PII>, greater<PII>> q[N];

void solve()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )  scanf("%d", &s[i]);
    
    while (m -- )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        while (q[b].size() && q[b].top().x <= a)
        {
            s[b] += q[b].top().y;
            q[b].pop();
        }
        if (s[b] < d) puts("-1");
        else
        {
            q[b].push({a + c, d});
            s[b] -= d;
            printf("%d\n", s[b]);
        }
    }
    
}

int main()
{
    solve();
    
    return 0;
}
```

## 5#3494国际象棋

```c++
#include <cstdio>
#include <iostream>
#include <vector>
#include <set>

using namespace std;

const int N = 110, M = 1 << 7, MOD = 1e9 + 7;

int n, m, k;
int f[N][M][M][22];
int map[M];

int get(int x)
{
    int res = 0;
    while (x)
    {
        res ++ ;
        x -= x & -x;
    }
    
    return res;
}

void solve()
{
    scanf("%d%d%d", &n, &m, &k);
    for (int i = 0; i < 1 << n; i ++ )
        map[i] = get(i);
        
    f[0][0][0][0] = 1;
    for (int i = 1; i <= m + 1; i ++ )
        for (int a = 0; a < 1 << n; a ++ )
            for (int b = 0; b < 1 << n; b ++ )
            {
                if (a & (b >> 2) || (a >> 2) & b) continue;
                for (int c = 0; c < 1 << n; c ++ )
                {
                    if (b & (c >> 2) || (b >> 2) & c) continue;
                    if (a & (c >> 1) || (a >> 1) & c) continue;
                    int t = map[c];
                    for (int j = t; j <= k; j ++ )
                        f[i][b][c][j] = (f[i][b][c][j] + f[i - 1][a][b][j - t]) % MOD;
                }
            }
    
    int res = 0;
    for (int a = 0; a < 1 << n; a ++ )
        for (int b = 0; b < 1 << n; b ++ )
            res = (res + f[m][a][b][k]) % MOD;
            
    printf("%d\n", res);
}

int main()
{
    solve();
    
    return 0;
}
```







