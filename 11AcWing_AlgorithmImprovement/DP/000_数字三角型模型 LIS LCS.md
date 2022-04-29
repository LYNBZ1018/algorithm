# 动态规划  DP *18

数字三角形模型、最长上升子序列模型、背包模型、状态机、状态压缩DP、区间DP、树形DP、数位DP、单调队列优化DP、斜率优化DP

```c++
网格的		f[i][j]
现行的		f[i]
背包的		f[i][j]  // 第一个体积 第二个权重
```



## 数字三角形模型   线性DP *5

![4.jpg](https://cdn.acwing.com/media/article/image/2022/04/17/186034_df531097be-4.jpg)  


​      

### #898数字三角形

**描述**

```c++
        7
      3   8
    8   1   0
  2   7   4   4
4   5   2   6   5
从顶部到底部，可以走左下或者右下找出一条路径使数字之和最大。
输出最大路径的数字之和。
```

**分析**

![数字三角形](https://s2.loli.net/2022/04/13/ygkGEYNHVMI5abX.jpg)

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510;

int n;
int f[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= i; j ++ )
            scanf("%d", &f[i][j]);
    
    for (int i = n; i >= 1; i -- )  // 从最后一层开始往上走 f 一开始初始化成了0，取max不影响
        for (int j = i; j >= 1; j -- )
            f[i][j] = max(f[i + 1][j], f[i + 1][j + 1]) + f[i][j]; 
    
    printf("%d\n", f[1][1]);
    
    return 0;
}
```

​    

### #1015摘花生

**描述**

```c++
从(1, 1)开始，只能向右和向下走，走到东南角最多能拿到多少花生。
```

**分析**

```c++
集合划分要:
		 1.不重复
		 2.不漏
```



![image-20220411214205230.png](https://cdn.acwing.com/media/article/image/2022/04/11/186034_f24a25f2b9-image-20220411214205230.png)

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n, m;
int w[N][N], f[N][N];

void solve()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            scanf("%d", &w[i][j]);
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            f[i][j] = max(f[i - 1][j], f[i][j - 1]) + w[i][j];
    
    printf("%d\n", f[n][m]);
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

​     

### #1018最低通行费用

**描述**

```c++
商人从左上走到右下，每走一步消耗一个时间，要在不超过2N-1的时间内走下去，且花费最小。
```

**分析**

```c++
2N - 1  -> 不能走回头路
花费最小  -> 属性为最小值 
```



![image-20220411221111775.png](https://cdn.acwing.com/media/article/image/2022/04/11/186034_f8448eb4b9-image-20220411221111775.png) 

**代码**

``` c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110, INF = 1e9;

int n;
int w[N][N], f[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            scanf("%d", &w[i][j]);
            
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (i == 1 && j == 1) f[i][j] = w[i][j];
            else 
            {
                f[i][j] = INF;
                if (i > 1) f[i][j] = min(f[i][j], f[i - 1][j] + w[i][j]);  // 当 i > 1 时，才能从上边下来
                if (j > 1) f[i][j] = min(f[i][j], f[i][j - 1] + w[i][j]);  // 当 j > 1 时，才能从左边出来
            }
    
    printf("%d\n", f[n][n]);
    
    return 0;
}
```

​     

### #1027方格取数

**描述**

```c++
从A到B，走两次，每个格子只能取一次数，取后变为0,
问走两次取得的数字和最大为多少
```

![image-20220411224508506.png](https://cdn.acwing.com/media/article/image/2022/04/11/186034_fbc6aa91b9-image-20220411224508506.png)

**分析**

![image-20220411225258498.png](https://cdn.acwing.com/media/article/image/2022/04/11/186034_ff4cd86bb9-image-20220411225258498.png)

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 15;

int n;
int w[N][N];
int f[N * 2][N][N];  // k 表示横纵坐标之和 -> 走的步数

int main()
{
    scanf("%d", &n);
    
    int a, b, c;
    while (cin >> a >> b >> c, a || b || c) w[a][b] = c;  // a b c 不同时为 0 
    
    for (int k = 2; k <= n + n; k ++ )  // 从(1, 1) 开始 k = 2
        for (int i1 = 1; i1 <= n; i1 ++ )
            for (int i2 = 1; i2 <= n; i2 ++ )
            {
                int j1 = k - i1, j2 = k - i2;
                if (j1 >= 1 && j1 <= n && j2 >= 1 && j2 <= n)
                {
                    int t = w[i1][j1];
                    if (i1 != i2)   t += w[i2][j2];  // 不相等时，两个点都取
                    int &x = f[k][i1][i2];
                    x = max(x, f[k - 1][i1 - 1][i2 - 1] + t);  // 上 上
                    x = max(x, f[k - 1][i1 - 1][i2] + t);  // 上 右
                    x = max(x, f[k - 1][i1][i2 - 1] + t);  // 右 上
                    x = max(x, f[k - 1][i1][i2] + t);  // 右 右
                }
            }
    
    printf ("%d\n", f[n + n][n][n]);
    
    return 0;
}
```

​         

### #275传纸条

**描述**

```c++
0 3 9
2 8 5
5 7 0
// 从左上到右下走两边，不能重复，两个人不能让同一个人传，取到最大值
```

**分析**

```c++
// 1 <= i <= n
// 1 <= k - i <= m
// i <= k - 1 && i >= k - m
i >= max(1, k - m) && i <= min(k - 1, n)
```

![传纸条](https://s2.loli.net/2022/04/13/5vYPNtzIax87CKi.png)

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 55;

int n, m;
int w[N][N];
int f[N * 2][N][N];

int main()
{
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            scanf("%d", &w[i][j]);
            
    for (int k = 2; k <= n + m; k ++ )
        for (int i = max(1, k - m); i <= min(n, k - 1); i ++ )
            for (int j = max(1, k - m); j <= min(n, k -1); j ++ )
                for (int a = 0; a <= 1; a ++ )
                    for (int b = 0; b <= 1; b ++ )
                    {
                        int t = w[i][k - i];
                        if (i != j || k == 2 || k == n + m)  // k == 2 在顶部 k == m + n 在右下角 i == j同一个点
                        {
                            t += w[j][k - j];
                            f[k][i][j] = max(f[k][i][j], f[k - 1][i - a][j - b] + t);
                        }
                    }
    
    printf("%d\n", f[n + m][n][n]);
    
    return 0;
}
```

​      

## 最长上升子序列模型   LIS LCS *13

​         ![3.jpg](https://cdn.acwing.com/media/article/image/2022/04/17/186034_c1cd6c12be-3.jpg) 

### #895最长上升子序列

**描述**

```c++
给定一个长度为N的数列，求数值严格单调递增的自序列
input:
7
3 1 2 1 8 5 6
output:
4
```

**分析**

![be7b1922f36f01a131872719b59b7ac.png](https://cdn.acwing.com/media/article/image/2022/04/14/186034_cb90cb0ebc-be7b1922f36f01a131872719b59b7ac.png) 

**代码**

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
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    
    for (int i = 1; i <= n; i ++ )
    {
        f[i] = 1;
        for (int j = 1; j < i; j ++ )
            if (a[j] < a[i])
                f[i] = max(f[i], f[j] + 1);
    }
    
    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = max(res, f[i]);
    printf("%d\n", res);
    
    return 0;
}
```

​                

### #896最长上升子序列II

**描述**

```markdown
求数值严格单调递增的子序列的长度最长是多少。
数据范围:
1 ≤ N ≤ 100000 
−1e9 ≤ 数列中的数 ≤ 1e9
```

**分析**

```
stk中存放的不是最长上升子序列

我们用a[i]去替代f[i]的含义是：以a[i]为最后一个数的严格单调递增序列,这个序列中数的个数为i个。
```

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int N = 1e5 + 10;

int n;
int a[N];
vector<int> stk;

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    
    stk.push_back(a[0]);
    
    for (int i = 1; i < n; i ++ )  // 如果a[i]大于栈顶元素就入栈
        if (stk.back() < a[i])
            stk.push_back(a[i]);
        else  // 如果小于栈顶元素 替换掉栈中第一个大于等于a[i]的元素
            *lower_bound(stk.begin(), stk.end(), a[i]) = a[i];
    
    cout << stk.size() << endl;
    
    return 0;
}
```

​     

### #1017怪盗基德的滑翔翼

**描述**

```c++
他可以选择一个方向逃跑，但是不能中途改变方向（因为中森警部会在后面追击）。
因为滑翔翼动力装置受损，他只能往下滑行（即：只能从较高的建筑滑翔到较低的建筑）。
他希望尽可能多地经过不同建筑的顶部，这样可以减缓下降时的冲击力，减少受伤的可能性。
请问，他最多可以经过多少幢不同建筑的顶部(包含初始时的建筑)？
```

**分析**

```
input:
8
300 207 155 299 298 170 158 65
output:
6

即找到一个点向一个方向可以找到的最长的序列中数的个数
向跳的方向反向看就是找一个最长上升子序列
求两个方向的最长上升子序列
```

![c0f26ba373c3b46f85db3563ec2f967.png](https://cdn.acwing.com/media/article/image/2022/04/14/186034_ce698ea5bc-c0f26ba373c3b46f85db3563ec2f967.png)

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n;
int a[N], f[N];

void solve()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    
    // 正向求解LIS问题
    int res = 0;
    for (int i = 1; i <= n; i ++ )
    {
        f[i] = 1;
        for (int j = 1; j < i; j ++ )
            if (a[i] > a[j])
                f[i] = max(f[i], f[j] + 1);
        res = max(res, f[i]);
    }
    
    // 逆向求解LIS问题
    memset(f, 0, sizeof f);
    for (int i = n; i; i -- )
    {
        f[i] = 1;
        for (int j = n; j > i; j -- )
            if (a[i] > a[j])
                f[i] = max(f[i], f[j] + 1);
        res = max(res, f[i]);
    }
    
    printf("%d\n", res);
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

​     

### #1014登山

**描述**

```c++
五一到了，ACM队组织大家去登山观光，队员们发现山上一共有N个景点，并且决定按照顺序来浏览这些景点，即每次所浏览景点的编号都要大于前一个浏览景点的编号。
同时队员们还有另一个登山习惯，就是不连续浏览海拔相同的两个景点，并且一旦开始下山，就不再向上走了。
队员们希望在满足上面条件的同时，尽可能多的浏览景点，你能帮他们找出最多可能浏览的景点数么？
```

**分析**

```
按照编号递增浏览
不能浏览海拔相同的景点
一旦开始下山就不能上升

子序列  先严格单调上升 再严格单调下降
从左往右以a[k]为结尾的最大上升子序列 从右往左以a[k]为结尾的最大上升子序列 相加
```

![2446dbe0b4d59e630590189e4833707.png](https://cdn.acwing.com/media/article/image/2022/04/14/186034_c21708c0bc-2446dbe0b4d59e630590189e4833707.png) 

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n;
int a[N];
int f[N], g[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    
    for (int i = 1; i <= n; i ++ ) 
    {
        f[i] = 1;  // 只有一个的情况
        for (int j = 1; j < i; j ++ )  // 枚举1 到 i - 1
            if (a[i] > a[j])
                f[i] = max(f[i], f[j] + 1);
    }
    
    for (int i = n; i; i -- )
    {
        g[i] = 1;
        for (int j = n; j > i; j -- )
            if (a[i] > a[j])
                g[i] = max(g[i], g[j] + 1);
    }
    
    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = max(res, f[i] + g[i] - 1);
    printf("%d\n", res);
    
    return 0;
}
```

​      

### #482合唱队形

**描述**

```markdown
N 位同学站成一排，音乐老师要请其中的 (N−K) 位同学出列，使得剩下的 K 位同学排成合唱队形。     
合唱队形是指这样的一种队形：设 K 位同学从左到右依次编号为 1，2…，K，他们的身高分别为 T1，T2，…，TK，  则他们的身高满足 T1<…<Ti>Ti+1>…>TK(1≤i≤K)。     
你的任务是，已知所有 N 位同学的身高，计算最少需要几位同学出列，可以使得剩下的同学排成合唱队形。
```

**分析**

```markdown
和登山问题类似
找出以k为节点的顺序最长上升子序列和逆序最长上升子序列
用总数减去序列数
```

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n;
int a[N];
int f[N], g[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    
    // 顺序最长上升子序列
    for (int i = 1; i <= n; i ++ )
    {
        f[i] = 1;
        for (int j = 1; j < i; j ++ )
            if (a[i] > a[j])
                f[i] = max(f[i], f[j] + 1);
    }
    
    // 逆序最长上升子序列
    for (int i = n; i; i -- )
    {
        g[i] = 1;
        for (int j = n; j > i; j -- )
            if (a[i] > a[j])
                g[i] = max(g[i], g[j] + 1);
    }
    
    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = max(res, f[i] + g[i] - 1);
    
    printf("%d\n", n - res);
    
    return 0;
}
```

​     

### #1012友好城市

**描述**

```c++
北岸的每个城市有且仅有一个友好城市在南岸，而且不同城市的友好城市不相同。
每对友好城市都向政府申请在河上开辟一条直线航道连接两个城市，但是由于河上雾太大，政府决定避免任意两条航道交叉，以避免事故。
编程帮助政府做出一些批准和拒绝申请的决定，使得在保证任意两条航线不相交的情况下，被批准的申请尽量多。
```

**分析**

```
每个城市只有一个友好城市
每个城市只能建立一个桥
所有桥之间不能交叉

先按照一边桥梁排序 然后再寻找另一个序列的最长上升子序列
```

![2.jpg](https://cdn.acwing.com/media/article/image/2022/04/15/186034_22b22340bc-2.jpg) 

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 5010;

int n;
int f[N];
PII a[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d%d", &a[i].x, &a[i].y);
    
    sort(a, a + n);  // 先对一边进行排序
    
    int res = 0;
    for (int i = 0; i < n; i ++ )  // 找另一边的最长上升子序列
    {
        f[i] = 1;
        for (int j = 0; j < i; j ++ )
            if (a[i].y > a[j].y)
                f[i] = max(f[i], f[j] + 1);
        
        res = max(res, f[i]);
    }
    
    printf("%d\n", res);
    
    return 0;
}

```

​       

### #1016最大上升子序列和

**描述**

```markdown
求 最大的 上升子序列的 和

最长的上升子序列的和不一定是最大的，比如序列(100,1,2,3)的最大上升子序列和为100，而最长上升子序列为(1,2,3)。
```

**分析**

![824889a77b27d215908feb5da63a5dd.png](https://cdn.acwing.com/media/article/image/2022/04/15/186034_e1bb1d5dbc-824889a77b27d215908feb5da63a5dd.png) 

**代码**

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
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    
    int res = 0;
    for (int i = 0; i < n; i ++ )
    {
        f[i] = a[i];  // 空的情况下 只有a[i]一个
        for (int j = 0; j < i; j ++ )
            if (a[i] > a[j])
                f[i] = max(f[i], f[j] + a[i]);
        res = max(res, f[i]);
    }
    
    printf("%d\n", res);
    
    return 0;
}
```

​       

### #1010拦截导弹    DP

**描述**

```markdown
第一发炮弹能够到达任意的高度，但是以后每一发炮弹都不能高于前一发的高度。
    
系统可以拦截的最多的导弹数  ————  最长上升子序列
需要多少系统可以拦截所有导弹
```

**分析**

```markdown
贪心法: 放到各个子序列结尾的最小值后面
如果最优解不是放到结尾的后面，那么最优解的也可以放到贪心可以放的位置，
```

经过**调整**可以使**贪心**==**最优解**

![1.jpg](https://cdn.acwing.com/media/article/image/2022/04/16/186034_68e9a66bbd-1.jpg) 

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n;
int a[N];
int f[N], g[N];

int main()
{
    while (cin >> a[n]) n ++ ;
    
    int res = 0;
    for (int i = 0; i < n; i ++ )
    {
        f[i] = 1;
        for (int j = 0; j < i; j ++ )
            if (a[j] >= a[i])
                f[i] = max(f[i], f[j] + 1);
        res = max(res, f[i]);
    }
    
    cout << res << endl;
    
    // 贪心
    int cnt = 0;
    for (int i = 0; i < n; i ++ )
    {
        int k = 0;   						   // g的顺序就是序列结尾值升序的排列 因为能放到后边序列的一定是比第一个序列的结尾值大的数
        while (k < cnt && g[k] < a[i]) k ++ ;  // g存放每个序列的结尾值 找到可以存放的结尾后面
        g[k] = a[i];  // 更新序列结尾                          
        if (k >= cnt) cnt ++ ;  // 若果每个序列的结尾都大于a[i] 就新增一个序列
    }
    
    cout << cnt << endl;
    
    return 0;
}
```

​          

### #187导弹防御系统

**描述**

```markdown
一套防御系统的导弹拦截高度要么一直严格单调上升要么一直 严格单调 下降。、
请你求出至少需要多少套防御系统，就可以将它们全部击落。
```

**分析**

```
dfs 爆搜
```

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 55;

int n;
int a[N];
int up[N], down[N];
int ans;

void dfs(int u, int su, int sd)
{
    if (su + sd >= ans) return;
    if (u == n)
    {
        ans = su + sd;
        return;
    }
    
    // 放到当前上升子序列中
    int k = 0;
    while (k < su && up[k] >= a[u]) k ++ ;
    int t = up[k];
    up[k] = a[u];
    if (k < su) dfs(u + 1, su, sd);  // 加入到现有序列中
    else dfs(u + 1, su + 1, sd);  // 需要开辟新序列
    up[k] = t;
    
    // 放到当前下降子序列中
    k = 0;
    while (k < sd && down[k] <= a[u]) k ++ ;
    t = down[k];
    down[k] = a[u];
    if (k < sd) dfs(u + 1, su, sd);
    else dfs(u + 1, su, sd + 1);
    down[k] = t;
    
}

int main()
{
    while (cin >> n, n)
    {
        for (int i = 0; i < n; i ++ ) cin >> a[i];
        
        ans = n;  // 最多每个数是一个序列
        dfs(0, 0, 0);
        
        cout << ans << endl;
    }
    
    return 0;
}
```

​                           

### #897最长公共子序列

**描述**

```markdown
两个序列求最长公共子序列
```

**分析**

```
f[i][j] 表示以从（1,1）到（i，j）的公共序列的集合的最大值
如果a[i] == b[j], f[i][j] = f[i - 1][j - 1] + 1;  包含a[i] \ b[j]
若果a[i] != b[j], 就要从 f[i - 1][j] f[i][j - 1] f[i - 1][j - 1] 中取最大值 
f[i - 1][j] && f[i][j - 1] 包含 f[i - 1][j - 1]
```

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
char a[N], b[N];
int f[N][N];

int main()
{
    cin >> n >> m >> a + 1 >> b + 1;
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            if (a[i] == b[j])
                f[i][j] = f[i - 1][j - 1] + 1;
            else 
                f[i][j] = max(f[i - 1][j], f[i][j - 1]);
    
    cout << f[n][m] << endl;
    
    return 0;
}
```

​          

### #272最长公共上升子序列

**描述**

```markdown
对于两个数列 A 和 B，如果它们都包含一段位置不一定连续的数，且数值是严格递增的，
那么称这一段数是两个数列的公共上升子序列，
而所有的公共上升子序列中最长的就是最长公共上升子序列了。
```

**分析**

![2.jpg](https://cdn.acwing.com/media/article/image/2022/04/16/186034_2a209bc1bd-2.jpg) 

**代码**

暴力

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 3010;

int n;
int a[N], b[N];
int f[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &b[i]);
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
        {
            f[i][j] = f[i - 1][j];
            if (a[i] == b[j])
            {
                f[i][j] = max(f[i][j], 1);
                for (int k = 1; k < j; k ++ )
                    if (b[k] < b[j])
                        f[i][j] = max(f[i][j], f[i][k] + 1);
            }
        }
        
    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = max(res, f[n][i]);
    
    printf("%d\n", res);
    
    return 0;
}
```

优化

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 3010;

int n;
int a[N], b[N];
int f[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &b[i]);
    
    for (int i = 1; i <= n; i ++ )
    {
        int maxv = 1;
        for (int j = 1; j <= n; j ++ )
        {
            f[i][j] = f[i - 1][j];  // 不包含a[i]
            if (a[i] == b[j]) f[i][j] = max(f[i][j], maxv);
            if (b[j] < a[i]) maxv = max(maxv, f[i][j] + 1);
            // {
            //     f[i][j] = max(f[i][j], 1);
            //     for (int k = 1; k < j; k ++ )
            //         if (b[k] < a[i])
            //             f[i][j] = max(f[i][j], f[i][k] + 1);
            // }
        }
    }
    
    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = max(res, f[n][i]);
    
    printf("%d\n", res);
    
    return 0;
}
```

​          

### #902最短编辑距离

**描述**

```markdown
给定两个字符串A和B，现在要将A经过若干操作变为B，可进行的操作有：
删除–将字符串A中的某个字符删除。
插入–在字符串A的某个位置插入某个字符。
替换–将字符串A中的某个字符替换为另一个字符。
现在请你求出，将A变为B至少需要进行多少次操作。
```

**分析**

```
1)删除操作：把a[i]删掉之后a[1~i]和b[1~j]匹配
            所以之前要先做到a[1~(i-1)]和b[1~j]匹配
            f[i-1][j] + 1
2)插入操作：插入之后a[i]与b[j]完全匹配，所以插入的就是b[j] 
            那填之前a[1~i]和b[1~(j-1)]匹配
            f[i][j-1] + 1 
3)替换操作：把a[i]改成b[j]之后想要a[1~i]与b[1~j]匹配 
            那么修改这一位之前，a[1~(i-1)]应该与b[1~(j-1)]匹配
            f[i-1][j-1] + 1
            但是如果本来a[i]与b[j]这一位上就相等，那么不用改，即
            f[i-1][j-1] + 0
```

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
char a[N], b[N];
int f[N][N];

int main()
{
    cin >> n >> a + 1;
    cin >> m >> b + 1;
    
    for (int i = 1; i <= n; i ++ ) f[i][0] = i;  // 全删除
    for (int i = 1; i <= m; i ++ ) f[0][i] = i;  // 全插入
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m ;j ++ )
        {
            f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;  // 插入 和 删除
            f[i][j] = min(f[i][j], f[i - 1][j - 1] + (a[i] != b[j]));
            // a[i] == b[j] 不用操作 a[i] != b[j] 需要进行替换
        }
    
    printf("%d\n", f[n][m]);
    
    return 0;
}
```

​          

### #899编辑距离

**描述**

```markdown
给定n个长度不超过10的字符串以及m次询问，每次询问给出一个字符串和一个操作次数上限。
对于每次询问，请你求出给定的n个字符串中有多少个字符串可以在上限操作次数内经过操作变成询问给出的字符串。
每个对字符串进行的单个字符的插入、删除或替换算作一次操作。
```

**分析**

```
对每一次询问用最短编辑距离进行DP
```

**代码**

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010, M = 15;

int n, m;
char str[N][M];
int f[N][N];

int edit_distance(char a[], char b[])
{
    int lena = strlen(a + 1), lenb = strlen(b + 1);
    for (int i = 1; i <= lena; i ++ ) f[i][0] = i;  // 删除
    for (int i = 1; i <= lenb; i ++ ) f[0][i] = i;  // 插入
    
    for (int i = 1; i <= lena; i ++ )
        for (int j = 1; j <= lenb; j ++ )
        {
            f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
            f[i][j] = min(f[i][j], f[i - 1][j - 1] + (a[i] != b[j]));
        }
    
    return f[lena][lenb];
}

int main()
{
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) cin >> str[i] + 1;
    
    while (m -- )
    {
        int limit = 0;
        char ch[M];
        cin >> ch + 1 >> limit;
        int res = 0;
        for (int i = 0; i < n; i ++ )
            if (edit_distance(str[i], ch) <= limit)
                res ++ ;
        
        cout << res << endl;
    }
    
    return 0;
}
```
