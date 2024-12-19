# 设计思路

## 对禁手判断的改进

`segment`: 一条线上可能对当前点的胜负判断造成影响的所有点，即：连续9个点

使用dp预处理出一个`segment`对应的`pattern`

## 对 MCTS 的改进

### 限制棋盘大小

### 考虑连五点直接结束

### 对局面以及动作评分

#### 使用一个小型 CNN

#### 使用trivial的评分函数

## 对 minimax 的改进

### 威胁空间搜索 （暂译）

原论文：[Go-Moku and Threat-Space Search, Allis et.al., 1994](https://www.researchgate.net/publication/2252447_Go-Moku_and_Threat-Space_Search)

威胁空间搜索, Threat-Space Search, 是一种算杀的高效实现

#### 几种威胁的定义

- (a)Four: 连续五格中攻击者占4个，空1个
- (b)Strait Four: 连续六格中攻击者占中间4个，两端空
- (b)Three: 以下两者之一
	- 连续7格中攻击者占3格，其余空
	- 连续6格中中心4格由攻击者占用，