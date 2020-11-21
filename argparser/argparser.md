# Python Module : argparse

> Python 官方標準含式庫中推薦的命令列解析模組

``` python
import argparse
```

* 幫程式製作簡單的命令列設定
* 傳進指定的參數、要開啟的設定

### example

#### Initialize

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()

parser1.add_argument("--Foo",help="here is the usage of --Foo arg")
parser1.add_argument("bar",help="print the val of bar")

args = parser1.parse_args()

print(args.bar)
```
##### result

```
PS C:\Users\Naichen\learning_python\argparser> python practice.py -h
usage: practice.py [-h] [--Foo FOO] bar

positional arguments:
  bar         print the val of bar

optional arguments:
  -h, --help  show this help message and exit
  --Foo FOO     here is the usage of --Foo arg    
```

在生成一個`ArgumentParser()`物件後，就可以在命令列將各種設定、參數傳入執行程式，其中`-h`或`--help`是固定參數，可以顯示這個程式有多少參數能用，以及這些參數的用途，記住`-h`、`--help`調用後不會執行其他命令。


|                           | usage    |
|           -------         | -------- |
|   positional arguments    | 位置參數，可以直接輸入數值(預設是字串)傳進，例如以上範例就是要在第三個位置輸入數值 |
|    optional arguments     | 選項參數，輸入指定的參數名稱後接輸入數值。   |

#### DataType and Variable

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()

parser1.add_argument("--Foo",help="here is the usage of --Foo arg",type=float)

args = parser1.parse_args()

print(args.Foo)
```
##### result

```
PS C:\Users\Naichen\learning_python\argparser> python practice.py  --Foo 66.6
66.6 
```

這樣可以指定傳入參數的型態，若輸入不可變為指定型態的話就會報錯。另外在程式內調用時會保存在`ArgumentParser`內，名稱是dash後的名字，例如指定了`--Foo 66.6`，程式內的變數就是`ArgumentParser().Foo`。

#### Boolean

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()

parser1.add_argument("--Foo",help="here is the usage of ---Foo arg",action='store_true')

args = parser1.parse_args()

print(1 if args.Foo else 0)
```
##### result

```
PS C:\Users\Naichen\learning_python\argparser> python practice.py  --Foo
1
PS C:\Users\Naichen\learning_python\argparser> python practice.py     
0
```

指定`'store_true'`這個參數在`action`內就可以將這個參數變成一種開關，執行命領有加這個參數就會將`arg.f`設為`True`，反之不特別打上`--Foo`時在程式內變數就為`False`。

#### Short Command and Choices

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()

parser1.add_argument("-f","--foo",help="here is the usage of -f arg",type=int,choices=[1,2,3,4,5])

args = parser1.parse_args()

print(args.foo)
```
##### result

```
PS C:\Users\Naichen\learning_python\argparser> python practice.py -f  
usage: practice.py [-h] [-f {1,2,3,4,5}]
practice.py: error: argument -f/--foo: expected one argument
PS C:\Users\Naichen\learning_python\argparser> python practice.py -f 3
3
```

可以看到在`add_argument`內我們還是可以設定自己喜歡的變數名，但可以用短指令來實現縮寫，所以習慣在長的名字前面加2個dash，短指令前面加1個dash，短指令最方便的是可以不同的短指令互相疊加，概念類似下述的`count`。

另外可以透過`choices`這個選項傳入一個`list`來限制輸入的數值。

#### Count and Default

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()

parser1.add_argument("-f","--foo",help="here is the usage of -f arg",action='count')

args = parser1.parse_args()

print(args.foo)
```
##### result
```
PS C:\Users\Naichen\learning_python\argparser> python practice.py -f   
1
PS C:\Users\Naichen\learning_python\argparser> python practice.py -ff
2
PS C:\Users\Naichen\learning_python\argparser> python practice.py -fffff
5
PS C:\Users\Naichen\learning_python\argparser> python practice.py --foo 
1
PS C:\Users\Naichen\learning_python\argparser> python practice.py --foo --foo
2
PS C:\Users\Naichen\learning_python\argparser> python practice.py
None
```

可以透過指定`'count'`參數在`action`內來紀錄指定參數出現的次數，短參數可以直接疊加符號，長參數只能單獨多打。和`'store_true'`最大的不同是，被設定成`'store_true'`的參數如果沒有接收到使用者的命令，會自動設定成`False`，但`'count'`的場合會讓變數的值為`None`而不是0，我們可以透過在`add_argument`內加入參數`default=0`來指定預設數值避免這個錯誤。

#### Conflict with Boolean

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()

parser1.add_argument("-f","--foo",help="here is the usage of -f arg",action='store_true')
parser1.add_argument("-b","--bar",help="here is the usage of -f arg",action='store_true')

args = parser1.parse_args()

print(f"args.foo={args.foo},args.bar={args.bar}")
```
##### result
```
PS C:\Users\Naichen\learning_python\argparser> python practice.py
args.foo=False,args.bar=False
PS C:\Users\Naichen\learning_python\argparser> python practice.py -f
args.foo=True,args.bar=False
PS C:\Users\Naichen\learning_python\argparser> python practice.py -b
args.foo=False,args.bar=True
PS C:\Users\Naichen\learning_python\argparser> python practice.py -fb
args.foo=True,args.bar=True
```

##### code
```python
import argparse

parser1 = argparse.ArgumentParser()
group = parser1.add_mutually_exclusive_group()
group.add_argument("-f","--foo",help="here is the usage of -f arg",action='store_true')
group.add_argument("-b","--bar",help="here is the usage of -f arg",action='store_true')

args = parser1.parse_args()

print(f"args.foo={args.foo},args.bar={args.bar}")
```
##### result
```
PS C:\Users\Naichen\learning_python\argparser> python practice.py -f 
args.foo=True,args.bar=False
PS C:\Users\Naichen\learning_python\argparser> python practice.py -b 
args.foo=False,args.bar=True
PS C:\Users\Naichen\learning_python\argparser> python practice.py -fb
usage: practice.py [-h] [-f | -b]
practice.py: error: argument -b/--bar: not allowed with argument -f/--foo
```

可以看到將`ArgumentParser()`加上`add_mutually_exclusive_group()`的物件，在`group`內去`add_argument()`時就會讓`group`內部的指令不能同時使用，在調用`-fb`時就會出錯。但嘗試過`-ff`的調用是可行的。

#### Description



tags: `Python` `argparse`

