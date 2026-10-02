# dsh-linalg（中文简介）

DeepSeek Harness (dsh) 线性代数工具箱。五个纯本地计算工具，零运行时依赖：

- `matrix_multiply` — 矩阵乘法，自动检查维度
- `matrix_compute` — 转置 / 行列式 / 逆矩阵 / 迹 / 行最简形（rref）/ 秩（rank）/ 整数次幂（power）
- `matrix_eigen` — 实对称矩阵特征分解：全部特征值（降序）+ 单位特征向量 + 实测残差
- `solve_linear` — 解线性方程组 Ax=b，明确分类：唯一解 / 无穷多解（返回特解 + 零空间基 + 自由变量个数）/ 无解
- `vector_ops` — 点积 / 叉积（3D）/ 模长 / 投影 / 夹角（度）

实现采用部分主元高斯消元，输出统一舍入到 10 位小数（可配），奇异矩阵与矛盾方程组给出明确报错而不是编造结果。

`matrix_eigen` 走循环 Jacobi 旋转（全双精度）：特征向量统一单位长度，符号约定为首个非零分量为正；返回里带 `maxResidual`（实测的 max|A·v − λ·v|）。非方阵、非对称矩阵、不收敛一律报错，不给猜测的谱。

`power` 走二进制快速幂：负指数先求逆（奇异矩阵明确报错），指数限整数 −64…64。

安装：

```sh
dsh plugin --profile web add github:TYEclipse/dsh-linalg
```

完整文档见 [README.md](README.md)。MIT 许可。
