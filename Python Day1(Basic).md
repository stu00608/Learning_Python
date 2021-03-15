# Python Day1(Basic)

此篇將介紹Python基本語法

#### (1) 輸出
```
Hellow = 'Hellow ShaoTX'
print(Hellow)
```

#### 輸出結果
```
Hellow ShaoTX
```

#### (2) 宣告與運算
```
a = 10
b = 1.5
c = 3


print('a + b = ', a + b)
print('a - c = ', a - c)
print('a * b = ', a * b)
print('a / c = ', a / c)
```

#### 執行結果
```
a + b =  11.5
a - c =  7
a * b =  15.0
a / c =  3.3333333333333335
```

#### (3) 查看型態
```
print('Type of Hellow is ', type(Hellow))
print('Type of a is ', type(a))
```
#### 執行結果
```
Type of Hellow ShaoTX is  <class 'str'>
Type of a is  <class 'int'>
```

#### (4) 轉換型態
```
# float 轉 int
newc = int(b)
print('1.149029 after int() = ', newc)

# int 轉 float
newa = float(a) 
print('9 after float() = ', newa)

# 字符轉 ASCII code
ASCII_A = ord('A')
print('A after ord() = ', ASCII_A)

# 
newHellow = str(Hellow)
print('Hellow ShaoTX after str() = ', newHellow)

# ASCII code 轉 字符
ASCII_65 = chr(65)
print('65 after chr() = ', ASCII_A)
```
#### 執行結果
```
1.149029 after int() =  1
10 after float() =  10.0
A after ord() =  65
Hellow ShaoTX after str() =  Hellow ShaoTX
65 after chr() =  65
```

#### (5) 輸入
* Python的輸入方式為input()，會於輸出端顯示()裡的字，並將值賦予前方所宣告的變數
* 以此範例說明，將會在輸出端顯示Input a letter : ，若此時輸入Hellow World，Hellow World則會存入X
```
X = input('Input a letter : ')
print(X)
```
#### 執行結果
```
Input a letter : 
```
```
Hellow World
```
#### (5) 布林運算
```
# 比較運算符號優先級高於賦值運算値
# 先做 == 再做 = 

flag0 = 1 == 1
flag1 = 3 > 2
flag2 = 3 < 2
flag3 = flag1 and flag2
flag4 = flag1 or flag2
flag5 = not (1 != 2)

print('flag0 = ', flag0)
print('flag1 = ', flag1)
print('flag2 = ', flag2)
print('flag3 = ', flag3)
print('flag4 = ', flag4)
print('flag5 = ', flag5)
```
#### 執行結果
```
flag0 =  True
flag1 =  True
flag2 =  False
flag3 =  False
flag4 =  True
flag5 =  False
```