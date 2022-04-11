# 算法基础

## 1.基础算法

****



### #785 快速排序

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int n;
int a[N];

void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;
    
    int x = q[l + r >> 1];
    int i = l - 1, j = r + 1;
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    
    quick_sort(q, l, j);
    quick_sort(q, j + 1, r);
}

void solve()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    
    quick_sort(a, 0, n - 1);
    
    for (int i = 0; i < n; i ++ ) printf("%d ", a[i]);
    puts("");
}

int main()
{
    solve();
    
    return 0;
}
```

****

  

### #787 归并排序

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int n;
int a[N], tmp[N];

void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;
    
    int mid = l + r >> 1;
    merge_sort(q, l, mid), merge_sort(q, mid + 1, r);
    
    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] < q[j]) tmp[k ++ ] = q[i ++ ];
        else             tmp[k ++ ] = q[j ++ ];
    while (i <= mid)     tmp[k ++ ] = q[i ++ ];
    while (j <= r)       tmp[k ++ ] = q[j ++ ];
    
    for (int i = l, j = 0; i <= r; i ++ , j ++ ) q[i] = tmp[j];
}

void solve()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    
    merge_sort(a, 0, n - 1);
    
    for (int i = 0; i < n; i ++ ) printf("%d ", a[i]);
    puts("");
}

int main()
{
    solve();
    
    return 0;
}
```

***

  

### #789 数的范围   *二分

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int q[N];

void solve()
{
    int x;
    scanf("%d", &x);
    
    int l = 0, r = n - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (x <= q[mid]) r = mid;
        else l = mid + 1;
    }
    if (q[l] != x) cout << "-1 -1" << endl;
    else 
    {
        cout << l << ' ';
        int l = 0, r = n - 1;
        while (l < r)
        {
            int mid = l + r + 1 >> 1;
            if (x >= q[mid]) l = mid;
            else r = mid - 1;
        }
        
        cout << l << endl;
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i < n; i ++ ) scanf("%d", &q[i]);
    
    while (m -- )
        solve();
    
    return 0;
}
```

​                   

### #790 数的三次方根

```c++
#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;

void solve()
{
    double x;
    scanf("%lf", &x);

    double l = -1000, r = 1000;
    while (r - l > 1e-8)
    {
        double mid = (l + r) / 2;
        if (mid * mid * mid >= x) r = mid;
        else l = mid;
    }
    
    printf("%lf\n", l);
}

int main()
{
    solve();
    
    return 0;
}
```

****

  

### #791 高精度加法   *高精度

```c++
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

vector<int> add(vector<int> A, vector<int> B)
{
    vector<int> C;
    
    int t = 0;
    for (int i = 0; i < A.size() || i < B.size(); i ++ )
    {
        if (i < A.size()) t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        
        t /= 10;
    }
    
    if (t) C.push_back(1);
    return C;
}

int main()
{
    vector<int> A, B;
    string a, b;
    cin >> a >> b;
    for (int i = a.size() - 1; i >= 0; i -- ) A.push_back(a[i] - '0');
    for (int i = b.size() - 1; i >= 0; i -- ) B.push_back(b[i] - '0');
    
    vector<int> C = add(A, B);
    for (int i = C.size() - 1; i >= 0; i -- ) printf("%d", C[i]);
    puts("");
    
    return 0;
}
```

​          

### #792 高精度减法

```c++
vector<int> sub(vector<int> A, vector<int> B)  // A > B
{
    vector<int> C;
    int t = 0;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }
    
    while (c.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

​        

### #793 高精度乘法

```c++
vector<int> mul(vector<int> A, int b)
{
    vector<int> C;
    
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    
    return C;
}
```

​          

### #794 高精度除法

```c++
vector<int> div(vector<int> A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

****

​        

### #795 前缀和   *前缀和与差分

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int s[N];

void solve()
{
    int l, r;
    scanf("%d%d", &l, &r);
    printf("%d\n", s[r] - s[l - 1]);
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )
    {
        scanf("%d", &s[i]);
        s[i] = s[i] + s[i - 1];
    }

    while (m -- )
        solve();
    
    return 0;
}
```

​         

### #796 子矩阵的和

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e3 + 10;

int n, m, q;
int a[N][N], s[N][N];

void solve()
{
    int x1, y1, x2, y2;
    scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
    printf("%d\n", s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1] + s[x1 - 1][y1 - 1]);
}

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
        solve();
        
    return 0;
}
```

​          

### #797 差分

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int n, m;
int s[N];

void solve()
{
    int l, r, c;
    scanf("%d%d%d", &l, &r, &c);
    s[l] += c, s[r + 1] -= c;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &s[i]);
    for (int i = n; i; i -- ) s[i] -= s[i - 1];
    
    while (m -- )
        solve();
    
    for (int i = 1; i <= n; i ++ ) s[i] += s[i - 1];
    
    for (int i = 1; i <= n; i ++ ) printf("%d ", s[i]);
    puts("");
    
    return 0;
}
```

​           

### #798 差分矩阵

```c++
#include <cstdio>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 1010, INF = 0x3f3f3f3f;

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
            a[i + 1][j + 1] +=x;
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
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            a[i][j] += a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1];
    
    for (int i = 1; i <= n; i ++ )
    {
        for (int j = 1; j <= m; j ++ )
            printf("%d ", a[i][j]);
        puts("");
    }
    
    return 0;
}
```

****

### 拆分语句   *双指针算法

```c++
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

int main()
{
    char a[1000];
    cin.getline(a, 1000);
    int n = strlen(a);
    for (int i = 0; i < n; i ++ )
    {
        int j = i;
        while (j < n && a[j] != ' ') j ++ ;
        for (int k = i; k < j; k ++ ) cout << a[k];
        cout << endl;
        i = j;
    }
    
    return 0;
}
```

​             

### #799 最长连续不重复子序列   

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int n;
int a[N], s[N];

void solve()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    
    int res = 0;
    for (int i = 0, j = 0; i < n; i ++ )
    {
        s[a[i]] ++ ;
        while (s[a[i]] > 1)
        {
            s[a[j]] -- ;
            j ++ ;
        }
        
        res = max(res, i - j + 1);
    }
    
    printf("%d\n", res);
}

int main()
{
    solve();
    
    return 0;
}
```

***



### #801 二进制中1的个数   *位运算

```c++
#include <cstdio>
#include <iostream>

using namespace std;

int n;

int lowbit(int x)
{
    return x & -x;
}

void solve()
{
    int x, res = 0;
    scanf("%d", &x);
    while (x)
    {
        x -= lowbit(x);
        res ++ ;
    }
    printf("%d ", res);
}

int main()
{
    scanf("%d", &n);
    
    while (n -- )
        solve();
    
    return 0;
}
```

****

  

### #802 区间和   *离散化

```c++
#include <cstdio>
#include <iostream>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 3e6 + 10;

int n, m;
int a[N], s[N];  // 离散化减少了数组要分配的空间 和 求前缀和的复杂度

vector<int> alls;  // 储存所有待离散化的数
vector<PII> add, query;

int find(int x)
{
    int l = 0, r = alls.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (alls.at(mid) >= x) r = mid;
        else l = mid + 1;
    }
    
    return r + 1;
}

void solve()
{
    scanf("%d%d", &n, &m);
    
    for (int i = 0; i < n; i ++ )
    {
        int x, c;
        scanf("%d%d", &x, &c);
        add.push_back({x, c});
        
        alls.push_back(x);  // 把要插入的位置 x 放到alls里
    }
    
    for (int i = 0; i < m; i ++ )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        query.push_back({l, r});
        
        alls.push_back(l), alls.push_back(r);  // 把询问的区间放到alls
    }
    
    // 排序去重  
    sort(alls.begin(), alls.end());  // unique 只会将相邻的重复元素放到尾部，所以要先进行排序
    alls.erase(unique(alls.begin(), alls.end()), alls.end());  // unique返回一个指向重复元素开头的迭代器
    
    // 通过find找到离散化后对应的位置， 插入到a[]中
    for (auto item : add)
    {
        int x = find(item.x);  // 传入实际的x 返回x在vector中对应的位置进行插入
        a[x] += item.y;
    }
    
    // 前缀和预处理
    for (int i = 1; i < alls.size(); i ++ ) s[i] = s[i - 1] + a[i];
    
    // 通过find找到离散化后再a[]中对应的位置
    for (auto item : query)
    {
        int l = find(item.x), r = find(item.y);
        printf("%d\n", s[r] - s[l - 1]);
    }
}

int main()
{
    solve();
    
    return 0;
}
```

****

  

### #803 区间合并   *区间合并

```c++
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

int n;
vector<PII> segs;

void merge(vector<PII>& segs)
{
    vector<PII> res;
    
    sort(segs.begin(), segs.end());
    
    int st = -2e9, ed = -2e9;
    for (auto seg : segs)
    {
        if (ed < seg.x)
        {
            if (ed != -2e9) res.push_back({st, ed});
            st = seg.x, ed = seg.y;
        }
        else ed = max(ed, seg.y);
    }
    
    if (st != -2e9) res.push_back({st, ed});
    segs = res;
}

void solve()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        segs.push_back({l, r});
    }
    
    merge(segs);
    
    printf("%ld\n", segs.size());
}

int main()
{
    solve();
    
    return 0;
}
```



## 2.数据结构

***

### 单链表

```c++
#include <cstdio>
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int head, e[N], ne[N], idx;

void init()
{
    head = -1;
    idx = 0;
}

void add_to_head(int x)
{
    e[idx] = x;
    ne[idx] = head;
    head = idx ++ ;
}

void add_to_k(int k, int x)
{
    e[idx] = x;
    ne[idx] = ne[k];
    ne[k] = idx ++ ;
}

void remove(int k)
{
    ne[k] = ne[ne[k]];
}

void solve()
{
    char op;
    int x, k;
    cin >> op;
    if (op == 'H')
    {
        scanf("%d", &x);
        add_to_head(x);
    }
    else if (op == 'D')
    {
        scanf("%d", &k);
        if (k == 0)
            head = ne[head];
        remove(k - 1);
    }
    else 
    {
        scanf("%d%d", &k, &x);
        add_to_k(k - 1, x);
    }
}

int main()
{
    int m;
    scanf("%d", &m);
    
    init();
    
    while (m -- )
        solve();
    
    for (int i = head; i != -1; i = ne[i])
        printf("%d ", e[i]);
    puts("");
    
    return 0;
}
```





