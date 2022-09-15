# Week 1

## 1#4268.性感素数  质数

**题目描述**

<img src="https://gitee.com/lynbz1018/image/raw/master/img/20220912125744.png" alt="image-20220912125742999" style="zoom:67%;" />

   

**分析**

```markdown
质数不包含负数
如果n是质数，那么 n+6 或 n-6 也是质数。
当n不是质数时，从 n+1 开始进行判断，符合“性感质数”的就输出停止。
```

   

**Code**

```c++
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e8 + 10;

bool is_prime(int n)
{
    if (n < 2) return false;
    for (int i = 2; i <= n / i; i ++ )
        if (n % i == 0) return false;
    
    return true;
}

int main()
{
    int n;
    cin >> n;
    if (is_prime(n) && (is_prime(n - 6) || is_prime(n + 6)))
        printf("Yes\n%d", is_prime(n - 6) ? n -6 : n + 6);
    else
    {
        printf("No\n");
        for (int i = n + 1; ; i ++ )
            if (is_prime(i) && (is_prime(i - 6) || is_prime(i + 6)))
            {
                printf("%d\n", i);
                break;
            }
    }
    
    return 0;
}
```

   

## 2#4269.校庆

**题目描述**

![image-20220912134528015](https://gitee.com/lynbz1018/image/raw/master/img/20220912134529.png)

   

**分析**

```markdown
先用hash表存下校友的信息
输入参会人员时，看看参会者有没有在校友表中
a用来存校友中出生日期最小的，b用来存参会者最小的
```

   

**Code**

```c++
#include <iostream>
#include <unordered_set>

using namespace std;

int main()
{
    int n, m;
    cin >> n;
    
    unordered_set<string> hash;
    while (n -- )
    {
        string name;
        cin >> name;
        hash.insert(name);
    }
    
    int cnt = 0;
    string a, b;
    cin >> m;
    while (m -- )
    {
        string name;
        cin >> name;
        if (hash.count(name))
        {
            cnt ++ ;
            if (a.empty() || a.substr(6, 8) > name.substr(6, 8)) a = name;
        }
        
        if (b.empty() || b.substr(6, 8) > name.substr(6,8)) b = name;
    }
    
    printf("%d\n%s", cnt, cnt != 0 ? a.c_str() : b.c_str());
    
    return 0;
}
```

​    

# Week 2

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    

## 2#

**题目描述**



   

**分析**



   

**Code**

```c++

```

​    