# One Week

## 1#2058笨拙的手指

```c++
#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

void solve()
{
    string a, b;
    cin >> a >> b;
    for (int i = 0; i < a.size(); i ++ )
        for (int j = 0; j < b.size(); j ++ )
            for (int k = '0'; k != '3'; k ++ )
            {
                if (b[j] == k) continue;
                string btmp = b;
                btmp[j] = k;
                string atmp = a;
                atmp[i] ^= 1;
                int x = 0, y = 0;
                for (int t = 0; t < atmp.size(); t ++ ) x = x * 2 + atmp[t] - '0';
                for (int t = 0; t < btmp.size(); t ++ ) y = y * 3 + btmp[t] - '0';
                if (x == y)
                {
                    cout << x << endl;
                    return;
                }
            }
}

int main()
{
    solve();
    
    return 0;
}
```



## 2#2041干草堆

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e6 + 10;

int n, k;
int q[N];

void solve()
{
    memset(q, 0, sizeof q);
    
    scanf("%d%d", &n, &k);
    while (k -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        q[a] ++ ;
        q[b + 1] -- ;
    }
    
    
    for (int i = 1; i <= n; i ++ ) q[i] += q[i - 1];
    sort(q + 1, q + n + 1);
   
    printf("%d\n", q[(n + 1) / 2]);
}

int main()
{
    solve();
    
    return 0;
}
```



## 3#2060奶牛选美

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>

#define x first
#define y second

using namespace std;

typedef pair<int ,int> PII;

const int N = 60;

int n, m;
char g[N][N];
bool st[N][N];

int dx[4] = {-1, 0, 1, 0};
int dy[4] = {0, 1, 0, -1};
vector<PII> v1, v2;
bool f = false;

void bfs(int x, int y, vector<PII> &v)
{
    queue<PII> q;
    q.push({x, y});
    st[x][y] = true;
    v.push_back({x, y});
    while (q.size())
    {
        auto t = q.front();
        q.pop();
        for (int i = 0; i < 4; i ++ )
        {
            int a = t.x + dx[i], b = t.y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && g[a][b] == 'X' && !st[a][b])
            {
                q.push({a, b});
                st[a][b] = true;
                v.push_back({a, b});
            }
        }
    }
}

int length(int a, int b, int c, int d)
{
    return abs(a - c) + abs(b - d) - 1;
}

void solve()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
            cin >> g[i][j];
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
        if (!st[i][j] && g[i][j] == 'X')
        {
            if (!f)
            {
                bfs(i, j, v1);
                f = true;
            }
            else
                bfs(i, j, v2);
        }
    
    int res = 0x3f3f3f3f;
    for (auto a : v1)
        for (auto b : v2)
        if (res > length(a.x, a.y, b.x, b.y))
            res = length(a.x, a.y, b.x, b.y);
    
    printf("%d\n", res);
}

int main()
{
    solve();
    
    return 0;
}
```



## 4#拖拉机

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <deque>

#define x first
#define y second

using namespace std;

typedef pair<int ,int>  PII;

const int N = 1010;

int n, x, y;
deque<PII> q;
bool g[N][N];
int dist[N][N];
int dx[4] = {-1, 0, 1, 0};
int dy[4] = {0, 1, 0, -1};

int bfs(int sx, int sy)
{
    memset(dist, 0x3f, sizeof dist);
    dist[sx][sy] = 0;
    q.push_back({sx, sy});
    while (q.size())
    {
        auto t = q.front();
        q.pop_front();
        if (t.x == 0 && t.y == 0) return dist[0][0];
        for (int i = 0; i < 4; i ++ )
        {
            int a = t.x + dx[i], b = t.y + dy[i];
            if (a < 0 || a >= 1002 || b < 0 || b >= 1002) continue;
            if (dist[a][b] <= dist[t.x][t.y] + g[a][b]) continue;
                dist[a][b] = dist[t.x][t.y] + g[a][b];
            if (g[a][b])
                q.push_back({a, b});
            else 
                q.push_front({a, b});
        } 
    }
}

void solve()
{
    scanf("%d%d%d", &n, &x, &y);
    for (int i = 0; i < n; i ++ )   
    {
        int a, b;
        scanf("%d%d", &a, &b);
        g[a][b] = true;
    }
    
    cout << bfs(x, y) << endl;
}

int main()
{
    solve();
    
    return 0;
}
```



## 5#岛

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

int n;
int h[N];
PII q[N];

void solve()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )  scanf("%d", &h[i]);
    
    n = unique(h + 1, h + n + 1) - h - 1;
    h[n + 1] = 0;
    
    for (int i = 1; i <= n; i ++ ) q[i] = {h[i], i};
    
    sort(q + 1, q + n + 1);
    
    int res = 1, cnt = 1;
    for (int i = 1; i <= n; i ++ )
    {
        int k = q[i].y;
        if (h[k] > h[k + 1] && h[k] > h[k - 1]) cnt -- ;
        else if (h[k] < h[k + 1] && h[k] < h[k - 1]) cnt ++ ;
        
        if (q[i].x != q[i + 1].x)
            res = max(res, cnt);
    }
    
    printf("%d\n", res);
}

int main()
{
    solve();
    
    return 0;
}
```



## 6#马蹄铁

```c++

```



## 7#打乱的字母

```c++

```



****



# Two Week