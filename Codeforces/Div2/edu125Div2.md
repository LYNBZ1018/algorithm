### A - Integer Moves
```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long LL;

void solve()
{
    int a, b;
    scanf("%d%d", &a, &b);
    if (a == 0 && b == 0)
    {
        printf("0\n");
        return;
    }
    
    LL it = a * a + b * b;
    
    for (int i = 1; i <= 100; i ++ )
        if (i * i == it)
        {
            printf("1\n");
            return;
        }
    
    printf("2\n");
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

****

### B - XY Sequence
```c++
#include <cstdio>
#include <iostream>

using namespace std;

typedef long long LL;

void solve()
{
    int n, B, x, y;
    scanf("%d%d%d%d", &n, &B, &x, &y);
    LL res = 0;
    LL end = 0;
    for (int i = 0; i <= n; i ++ )
    {

        res += end;
        if (end + x > B)
            end -= y;
        else 
            end += x;
    }

    printf("%lld\n", res);
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

****

### C - Bracket Sequence Deletion
```c++
#include <cstdio>
#include <iostream>

using namespace std;

int n;

void solve()
{
    scanf("%d", &n);
    string s;
    cin >> s;
    int ans = 0, x = 0;
    
    for (int i = 0; i < n; )
    {
        if (s[i] == '(' && i < n - 1)  // 左括号 而且不是最后一个 正则 和 回文 都可以删掉
        {
            ans ++ ;
            i += 2;
            x += 2;  // 一下删掉两个
        }
        else if (s[i] == '(') i ++ ;
        else  // 处理有括号的情况 只能是回文
        {
            int j = i + 1;
            while (j < n)
            {
                if (s[j] == ')')
                {
                    ans ++ ;
                    x += j - i + 1;
                    break;
                }
                
                j ++ ;
            }
            
            i = j + 1;
        }
    }
    
    printf("%d %d\n", ans, n - x);
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

****

### D - For Gamers. By Gamers.
```c++
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

typedef long long LL;

void solve()
{
    int n, C;
    scanf("%d%d", &n, &C);
    vector<LL> a(C + 1); // C 从1开始
    
    for (int i = 0; i < n; i ++ )
    {
        int c, d, h;
        scanf("%d%d%d", &c, &d, &h);
        a[c] = max(a[c], d * 1LL * h);  // 话费金币相同的话 选择伤害和生命乘积最大的
    }
    
    for (int c = 1; c <= C; c ++ )
    {
        a[c] = max(a[c], a[c - 1]);
        for (int xc = c; xc <= C; xc += c)  // 在消耗金币不超过C的情况下 记录单位金币伤害生命乘积最大的
            a[xc] = max(a[xc], a[c] * (xc / c));  
    }
    
    int m;
    scanf("%d", &m);
    
    for (int i = 0; i < m; i ++ )
    {
        int D;
        LL H;
        scanf("%d%lld", &D, &H);
        
        int ans = upper_bound(a.begin(), a.end(), D * H) - a.begin();  // 用二分查找找到大于等于怪物生命伤害乘积的最小的位置
        if (ans > C)
            ans = -1;
        printf("%d ", ans);
    }
    
}

int main()
{
    solve();
    
    return 0;
}
```

****

### E - Star MST ???
```c++
#include <cstdio>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 255, MOD = 998244353;

int dp[N][N];
int c[N][N];

LL fast(LL x, LL y)
{
    LL s = 1;
    while (y)
    {
        if (y % 2) s = s * x % MOD;
        x = x * x % MOD;
        y = y >> 1;
    }
    return s;
}

void solve()
{
    int n, k;
    scanf("%d%d", &n, &k);
    n -- ;
    for (int i = 0; i <= n; i ++ )
    {
        c[i][0] = c[i][i] = 1;
        for (int j = 1; j < i; j ++ )
            c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % MOD;
    }
    
    dp[0][0] = 1;
    for (int j = 0; j < k; j ++ )
        for (int t = 0; t <= n; t ++ )
            for (int i = 0; i <= n - t; i ++ )
            {
                dp[i + t][j + 1] += 1LL* dp[i][j] * c[n - i][t] % MOD * fast(k - j, 1ll * t * (t - 1) / 2 + t  * i) % MOD;
                dp[i + t][j + 1] %= MOD;
            }
    printf("%d\n", dp[n][k]);
}

int main()
{
    solve();
    
    return 0;
}
```
