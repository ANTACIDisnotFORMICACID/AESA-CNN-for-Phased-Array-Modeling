# 理想4×4阵列零陷理论及适用边界

## 理想阵因子

令阵元位于YZ平面：

\[
r_{mn}=(0,md_y,nd_z),\qquad m,n=0,1,2,3.
\]

方向余弦记为 `u=q_y`、`v=q_z`，可见域满足 `u²+v²≤1`。理想同阵元阵列的阵因子为

\[
AF(u,v)=\sum_{m=0}^{3}\sum_{n=0}^{3}w_{mn}
e^{jk(md_yu+nd_zv)}.
\]

若权值为二维线性相位

\[
w_{mn}=A e^{j(m\beta_y+n\beta_z)},
\]

则阵因子可分离：

\[
AF=A S_4(kd_yu+\beta_y)S_4(kd_zv+\beta_z),
\]

其中

\[
S_4(\Psi)=1+e^{j\Psi}+e^{j2\Psi}+e^{j3\Psi}
=e^{j3\Psi/2}\frac{\sin(2\Psi)}{\sin(\Psi/2)}.
\]

当

\[
\Psi=\ell\pi/2,\qquad \ell\notin4\mathbb Z,
\]

有严格零点。对应方向余弦为

\[
u_\ell=\frac{\lambda}{d_y}\left(\frac{\ell}{4}-\frac{\beta_y}{2\pi}\right),\qquad
v_q=\frac{\lambda}{d_z}\left(\frac{q}{4}-\frac{\beta_z}{2\pi}\right).
\]

## 当前几何

47 GHz下 `λ≈6.3786 mm`。实际AEDT端口网格间距为 `d_y=6.8 mm`、`d_z=15 mm`。全同相时，两个轴最靠近轴向的零点方向余弦间隔分别为

\[
\lambda/(4d_y)\approx0.2345,
\qquad
\lambda/(4d_z)\approx0.1063.
\]

由于两个间距均远大于λ/2，可见空间内会出现多次相长与相消，栅瓣、多峰和零陷是理想阵列傅里叶结构的自然结果。

## 与四点任意相位优化的区别

严格Dirichlet零点结论要求二维线性、可分离相位。O3允许16个相位完全独立，因此不能把“任意O3权值一定存在严格零点”作为已证明命题。

对任意恒模权值，可以用二维Parseval关系限制平均阵因子能量，但Parseval本身只保证存在不高于平均值的位置，不保证总场严格为零。真实HFSS还包含两个远场极化分量和不同嵌入阵元方向图，理想零陷通常会被填充。

本归档中更适用于任意O3权值的严格结论来自 `HᴴH/60` 的单方向谱上界和恒模SDP上界，而不是线性扫描的Dirichlet零点公式。

