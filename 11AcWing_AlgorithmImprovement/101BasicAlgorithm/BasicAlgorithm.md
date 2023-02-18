## 位运算

### #90.64位整数乘法

**描述**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230214222116.png" alt="image-20230214222115048" style="zoom:80%;" />





**分析**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230214223530.png" alt="image-20230214223528941" style="zoom:80%;" />







**Code**

```c++
#include <cstdio>

typedef long long LL;

LL qadd(LL a, LL b, LL p)
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

int main()
{
    LL a, b, p;
    scanf("%lld%lld%lld", &a, &b, &p);
    printf("%lld", qadd(a, b, p));
    
    return 0;
}
```



#### 快速幂

```c++
#include <iostream>

using namespace std;

int qmi(int a, int k, int p)
{
    int res = 1;
    while (k)
    {
        if (k & 1) res = res * a % p;
        a = a * a % p;
        k >>= 1;
    }
    
    return res;
}

int main()
{
    int a, b, c;
    cin >> a >> b >> c;
    cout << qmi(a, b, c) << endl;
    
    return 0;
}
```



## 递推与递归


### #95.费解的开关

**描述**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215102021.png" alt="image-20230215101915510" style="zoom:80%;" />

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215102014.png" alt="image-20230215102013193" style="zoom:80%;" />



**分析**

枚举第一行的状态, 记录可以全部打情况下的最小步数

![image-20230215103210709](https://gitee.com/lynbz1018/image/raw/master/img/20230215103211.png)





**Code**

```c++
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 6;

char g[N][N], bg[N][N];
int dx[5] = {-1, 0, 1, 0, 0}, dy[5] = {0, 1, 0, -1, 0};

void turn(int x, int y)
{
    for (int i = 0; i < 5; i ++ )
    {
        int a = x + dx[i], b = y + dy[i];
        if (a < 0 || a >= 5 || b < 0 || b >= 5) continue;
        g[a][b] ^= 1;  // '0'48 二进制最后一位为0 '1'49 最后一位为1 只需要修改最后一位即可
    }
}

int main()
{
    int T = 0;
    scanf("%d", &T);
    while (T -- )
    {
        for (int i = 0; i < 5; i ++ ) scanf("%s", bg[i]);
        
        int res = 18;
        for (int op = 0; op < 32; op ++ )  // 枚举第一行的每一种情况
        {
            int cnt = 0;
            memcpy(g, bg, sizeof g);
            for (int i = 0; i < 5; i ++ )
                if (op >> i & 1)
                {
                    turn(0, i);
                    cnt ++ ;
                }
                
            // 根据第一行状态修改后边的状态
            for (int i = 0; i < 4; i ++ )  // 根据前一行修改下一行 遍历前四行
                for (int j = 0; j < 5; j ++ )
                    if (g[i][j] == '0')
                    {
                        turn(i + 1, j);
                        cnt ++ ;
                    }
                    
            bool flag = true;
            for (int i = 0; i < 5; i ++ )
                if (g[4][i] == '0')
                {
                    flag = false;
                    break;
                }
            
            if (flag && res > cnt) res = cnt;
        }
        
        if (res > 6) res = -1;
        printf("%d\n", res);
    }
    
    return 0;
}
```




### #97.约数之和

**描述**

![image-20230215105421443](https://gitee.com/lynbz1018/image/raw/master/img/20230215105422.png)





**分析**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215110055.png" alt="image-20230215110054145" style="zoom:80%;" />

![image-20230215110524496](https://gitee.com/lynbz1018/image/raw/master/img/20230215110525.png)





**Code**

```c++
#include <cstdio>

const int MOD = 9901;

int qmi(int a, int k)
{
    int res = 1;
    a %= MOD;
    while (k)
    {
        if (k & 1) res = res * a % MOD;
        a = a * a % MOD;
        k >>= 1;
    }
    
    return res;
}

int sum(int p, int k)
{
    if (k == 1) return 1;
    if (k % 2 == 0) return (1 + qmi(p, k / 2)) * sum(p, k / 2) % MOD;
    return (sum(p, k - 1) + qmi(p, k - 1)) % MOD;
}

int main()
{
    int a, b;
    scanf("%d%d", &a, &b);
    
    int res = 1;
    // 分解质因数
    for (int i = 2; i * i <= a; i ++ )
        if (a % i == 0)
        {
            int s = 0;
            while (a % i == 0)
            {
                a /= i;
                s ++ ;
            }
            
            res = res * sum(i, b * s + 1) % MOD;
        }
        
    if (a > 1) res = res * sum(a, b + 1) % MOD;
    if (a == 0) res = 0;
    
    printf("%d\n", res);
    
    return 0;
}
```




### #98.分形之城

**描述**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215115749.png" alt="image-20230215115748296" style="zoom:67%;" />

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215115802.png" alt="image-20230215115800817" style="zoom:67%;" />



**分析**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215115606.png" alt="image-20230215115604752" style="zoom:67%;" />

![image-20230215120905935](https://gitee.com/lynbz1018/image/raw/master/img/20230215120907.png)





**Code**

```c++
#include <cstdio>
#include <cstring>
#include <cmath>

typedef long long LL;

struct Point {
    LL x, y;
};

Point get(LL n, LL a)
{
    if (n == 0) return {0, 0};
    LL blocknum = 1ll << (n * 2 - 2), len = 1ll << (n - 1);
    auto p = get(n - 1, a % blocknum);
    LL x = p.x, y = p.y;
    int z = a / blocknum;
    
    if (z == 0) return {y, x};
    else if (z == 1) return {x, y + len};
    else if (z == 2) return {x + len, y + len};
    return {len + len - y - 1, len - x - 1};
}

int main()
{
    int T = 0;
    scanf("%d", &T);
    while (T -- )
    {
        LL n, a, b;
        scanf("%lld%lld%lld", &n, &a, &b);
        auto pa = get(n, a - 1);  // 从0开始计数
        auto pb = get(n, b - 1);
        double dx = pa.x - pb.x, dy = pa.y - pb.y;
        printf("%.0lf\n", sqrt(dx * dx + dy * dy) * 10);
    }
    
    return 0;
}
```





## 前缀和与分差


### #99.激光炸弹

**描述**

![image-20230215132430538](https://gitee.com/lynbz1018/image/raw/master/img/20230215132431.png)

![image-20230215132443340](https://gitee.com/lynbz1018/image/raw/master/img/20230215132444.png)



**分析**









**Code**

```c++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 5010;

int n, r;
int s[N][N];

int main()
{
    scanf("%d%d", &n, &r);
    r = min(r, 5001);
    
    for (int i = 0; i < n; i ++ )
    {
        int x, y, z;
        scanf("%d%d%d", &x, &y, &z);
        x ++ , y ++ ;  // 转换成从1开始计数
        s[x][y] += z;
    }
    
    for (int i = 1; i <= 5001; i ++ )
        for (int j = 1; j <= 5001; j ++ )
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
    
    int res = 0;
    for (int i = r; i <= 5001; i ++ )
        for (int j = r; j <= 5001; j ++ )
            res = max(res, s[i][j] - s[i - r][j] - s[i][j - r] + s[i - r][j - r]);
            
    printf("%d\n", res);
    
    return 0;
}
```




### #100.增减序列

**描述**

![image-20230215140010460](https://gitee.com/lynbz1018/image/raw/master/img/20230215140011.png)





**分析**

```markdown
求出差分数列后
b[1] = a[1];
没执行一次操作就是差分序列一个数加1 一个数减1
最终目标是b[2] 到 b[n]都为0 这样前缀和求出a[i]后都为a[1]
1. 2 <= i, j <= n
2. i = 1, 2 <= j <= n
3. 2 <= i <= n, j = n + 1
4. i = 1, j = n + 1  // 没有意义
优先选择1. 一次可以消除两个
pos - neg, pos 和 neg分别表示差分数列中所有正数和负数的绝对值之和
1.情况下 一个加一 一个减一
|pos - neg| 剩下额进行2. 3.操作

情况种类 比如剩下5, 这5个值对2. 和 3. 的分配
5 0
0 5
1 4
4 1
3 2
2 3
一共有|pos - neg| + 1
```









**Code**

```c++
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

typedef long long LL;

const int N = 100010;

int n;
int a[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    for (int i = n; i >= 1; i -- ) a[i] -= a[i - 1];
    
    LL pos = 0, neg = 0;
    for (int i = 2; i <= n; i ++ )
        if (a[i] > 0) pos += a[i];
        else neg -= a[i];
    
    printf("%lld\n", max(pos, neg));  // min(pos, neg) + abs(pos - neg) == max(pos, neg);
    printf("%lld\n", abs(pos - neg) + 1);
    
    return 0;
}
```



## 二分


### #102.最佳牛围栏

**描述**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215144706.png" alt="image-20230215144705869" style="zoom:80%;" />

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215144718.png" alt="image-20230215144717303" style="zoom:80%;" />



**分析**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230215144627.png" alt="image-20230215144626264" style="zoom:80%;" />







**Code**

```c++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 1e5 + 10;

int n, F;
double a[N], s[N];

bool check(double avg)
{
    for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + a[i] - avg;
    
    double mins = 0;
    for (int k = F; k <= n; k ++ )
    {
        mins = min(mins, s[k - F]);  // 记录s[0]开始的最小值
        if (s[k] >= mins) return true;
    }
    
    return false;
}

int main()
{
    scanf("%d%d", &n, &F);
    double l = 0, r = 0;
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%lf", &a[i]);
        r = max(r, a[i]);
    }
    
    while (r - l > 1e-5)  // 浮点二分
    {
        double mid = (l + r ) / 2;
        if (check(mid)) l = mid;
        else r = mid;
    }
    
    printf("%d\n", (int)(r * 1000));
    
    return 0;
}
```



### #113.特殊排序

**描述**

![image-20230215154653897](https://gitee.com/lynbz1018/image/raw/master/img/20230215154655.png)





**分析**

![image-20230215154555992](https://gitee.com/lynbz1018/image/raw/master/img/20230215154557.png)







**Code**

```c++
// Forward declaration of compare API.
// bool compare(int a, int b);
// return bool means whether a is less than b.

class Solution {
public:
    vector<int> specialSort(int N) {
        vector<int> res;
        res.push_back(1);
        for (int i = 2; i <= N; i ++ )
        {
            int l = 0, r = res.size() - 1;
            while (l < r)
            {
                int mid = l + r + 1 >> 1;
                if (compare(res[mid], i)) l = mid;
                else r = mid - 1;
            }
            
            res.push_back(i);
            for (int j = res.size() - 2; j > r; j -- ) swap(res[j], res[j + 1]);
            if (compare(i, res[r])) swap(res[r], res[r + 1]);
        }
        
        return res;
    }
};
```



## 排序


### #105.七夕祭

**描述**

![image-20230215215201019](https://gitee.com/lynbz1018/image/raw/master/img/20230215215202.png)



![image-20230215215235754](https://gitee.com/lynbz1018/image/raw/master/img/20230215215306.png)



**分析**

```markdown
操作行时(一个列中上下交换变化), 每列的总和不变
操作列时, 每行的总和不变
```

![ee5475e43f1938a9052a8a4e848d437](https://gitee.com/lynbz1018/image/raw/master/img/20230215225422.jpg)







**Code**

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n, m, k;
int col[N], row[N];
LL c[N];

LL get(int *a, int n)
{
    LL avg = 0;
    for (int i = 1; i <= n; i ++ ) avg += a[i];
    avg /= n;
    
    for (int i = 1; i <= n; i ++ ) a[i] -= avg;
    c[1] = 0;
    for (int i = 2; i <= n; i ++ ) c[i] = c[i - 1] + a[i];
    
    sort(c + 1, c + 1 + n);
    int mid = c[n / 2 + 1];
    
    LL sum = 0;
    for (int i = 1; i <= n; i ++ ) sum += abs(c[i] - mid);
    
    return sum;
}

int main()
{
    scanf("%d%d%d", &n, &m, &k);
    for (int i = 1; i <= k; i ++ )
    {
        int x, y;
        scanf("%d%d", &x, &y);
        row[x] ++, col[y] ++ ;  // col[i] 表示第i行中所有的点之和
    }
    
    if (k % n && k % m) puts("impossible");
    else if (k % n) printf("column %lld", get(col, m));
    else if (k % m) printf("row %lld", get(row, n));
    else printf("both %lld", get(row, n) + get(col, m));
    
    return 0;
}
```




### #106.动态中位数

**描述**

![image-20230218171523784](https://gitee.com/lynbz1018/image/raw/master/img/20230218171618.png)

![image-20230218171547310](https://gitee.com/lynbz1018/image/raw/master/img/20230218171548.png)

```markdown
动态输出中位数
一个一个输入数据时, 当输入个数为奇数时, 就输出一个中位数
```

**分析**

```markdown
维护一个大顶堆和一个小顶堆
以大顶堆的top(), 为基准, 大顶堆中的都大于等于top(), 小于大顶堆的top()的放入到小顶堆中
当个数位奇数时保证down.size() = up.size() + 1
中位数就是down.top
```

**Code**

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <queue>

using namespace std;

int main()
{
    int T;
    scanf("%d", &T);
    while (T -- )
    {
        int n, m;
        scanf("%d%d", &m, &n);
        printf("%d %d\n", m, (n + 1) / 2);  // 输出编号 和 动态中位数的个数
        
        priority_queue<int> down;
        priority_queue<int, vector<int>, greater<int>> up;
        
        int cnt = 0;
        for (int i = 1; i <= n; i ++ )
        {
            int tmp;
            scanf("%d", &tmp);
            
            if (down.empty() || tmp <= down.top()) down.push(tmp);
            else up.push(tmp);
            
            if (down.size() > up.size() + 1) up.push(down.top()), down.pop();
            if (up.size() > down.size()) down.push(up.top()), up.pop();
            
            if (i % 2)
            {
                printf("%d ", down.top());
                if ( ++ cnt % 10 == 0) puts("");
            }
        }
        
        if (cnt % 10) puts("");
    }
    
    return 0;
}
```




### #107.超快速排序

**描述**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230218175459.png" alt="image-20230218175458066" style="zoom:80%;" />

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20230218175513.png" alt="image-20230218175512685" style="zoom:80%;" />



**分析**

```markdown
只能交换两个相邻的数
从逆序到升序的有序排列 需要消除所有的逆序对数
最少的操作次数就是没交换一次减少一个逆序对数
```









**Code**

```c++
#include <cstdio>

typedef long long LL;

const int N = 5e5 + 10;

int n;
int q[N], w[N];

LL merge_sort(int l, int r)
{
    if (l == r) return 0;
    
    int mid = l + r >> 1;
    LL res = merge_sort(l, mid) + merge_sort(mid + 1, r);
    int i = l, j = mid + 1, k = 0;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) w[k ++ ] = q[i ++ ];
        else
        {
            res += mid - i + 1;  // l ~ mid 是从大到小有序的 如果q[j] > q[i] 那么i后边的包括i mid - i + 1个都可以和q[j]构成逆序对
            w[k ++ ] = q[j ++ ];
        }
    
    while (i <= mid) w[k ++ ] = q[i ++ ];
    while (j <= r) w[k ++ ] = q[j ++ ];
    
    for (int i = l, j = 0; i <= r; i ++ , j ++ ) q[i] = w[j];
    
    return res;
}

int main()
{
    while (scanf("%d", &n), n)
    {
        for (int i = 0; i < n; i ++ ) scanf("%d", &q[i]);
        
        printf("%lld\n", merge_sort(0, n - 1));
    }
    
    return 0;
}
```



## RMQ


### #1273.天才的记忆

**描述**

RMQ区间最值查询

![image-20230218182332240](https://gitee.com/lynbz1018/image/raw/master/img/20230218182333.png)

![image-20230218182355785](https://gitee.com/lynbz1018/image/raw/master/img/20230218182356.png)



**分析**

![image-20230218183610204](https://gitee.com/lynbz1018/image/raw/master/img/20230218183611.png)

**Code**

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

using namespace std;

const int N = 2e5 + 10, M = 18;

int n, m;
int w[N];
int f[N][M];  // f[i][j]表示从i开始长度为2^j次方长度区间中最大的值

void init()
{
    for (int j = 0; j < M; j ++ )
        for (int i = 1; i + (1 << j) - 1 <= n; i ++ )
            if (!j) f[i][j] = w[i];
            else f[i][j] = max(f[i][j - 1], f[i + (1 << j - 1)][j - 1]);
}

int query(int l, int r)
{
    int len = r - l + 1;
    int k = log(len) / log(2);
    
    return max(f[l][k], f[r - (1 << k) + 1][k]);
}

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    
    init();
    
    scanf("%d", &m);
    while (m -- )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        printf("%d\n", query(l, r));
    }
    
    return 0;
}
```

