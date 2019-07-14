## 辐射通量(Radiant Flut)

辐射通量$\Phi$(radiant flut)又称功率，单位时间内穿过表面或空间区域的全部能量，单位为瓦特($W$,光学为$lm$流明)用符号$\Phi$表示。

辐射通量公式为：$\Phi=\frac {dQ}{dt}$

## 辐射强度(Intensity)

辐射强度$I$(intensity)表示每单位立体角的辐射通量，单位$W/sr$。辐射强度描述了光源的方向性分布，但只对点光源才有意义，因为点光源的面积为0，无法使用辐射照度。

辐射强度公式为：$I=\frac {d\Phi}{d\omega}$

## 辐射照度(Irradiance)

辐射照度$E$(Irradiance)是单位面积上接收到的光源辐射通量，也就是辐射通量的密度，单位为$W/m^2$(光学中为$lux$勒克斯)，用符号$E$表示。

辐射照度公式为：$E=\frac {d\Phi}{dA}$

## 辐射亮度(Radiance)

辐射亮度$L$(radiance)光源在每单位立体角和每单位投影面积上的辐射通量，单位为$W/m^2sr$(光学中为尼特$nit$)。

辐射亮度公式为：$L=\frac{d^2\Phi}{d\omega dA^\bot}=\frac{d^2\Phi}{d\omega dAcos\theta}$

$L=\frac{d^2\Phi}{d\omega dAcos\theta}=\frac{dE_L}{d\omega cos\theta}\Rightarrow dE_L=Lcos\theta d\omega$

从而得到：$E_L=\int_\Omega cos\theta d\omega=L\int_\Omega cos\theta d\omega=\pi L$
