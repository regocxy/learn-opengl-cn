# Specular-IBL

在上一篇教程中，我们通过预先使用预计算辐照度图作为照明的间接漫射(Diffuse)部分，将PBR与IBL结合起来。在本教程中，我们将关注反射率方程的镜面(Specular)部分:

$$L_o(p,\omega_o) = \int\limits_{\Omega} (k_d\frac{c}{\pi} + k_s\frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)})L_i(p,\omega_i) n \cdot \omega_i  d\omega_i$$

你会注意到Cook-Torrance镜面部分(乘以$kS$)在积分上不是常数，它依赖于入射光的方向，也依赖于入射视图的方向。试图求解所有入射光方向(包括所有可能的视图方向)的积分是一种组合过载，而且太昂贵，无法实时计算。Epic Games提出了一种解决方案，他们能够对镜面部分进行实时预卷积，并给出了一些妥协，即所谓的分割和近似(split sum approximation)。

分割和近似将反射率方程的高光部分分割成两个独立的部分，我们可以分别对其进行卷积，然后在PBR着色器中结合起来，实现基于高光间接图像的照明。与我们对辐照度图进行预卷积的方法类似，分割和近似需要一个HDR环境图作为卷积输入。为了理解分割和近似，我们将再次查看反射率方程，但这次只关注镜面部分(我们在之前的教程中提取了漫反射部分):

$$L_o(p,\omega_o) = \int\limits_{\Omega} (k_s\frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)}L_i(p,\omega_i) n \cdot \omega_i  d\omega_i=\int\limits_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p,\omega_i) n \cdot \omega_i  d\omega_i$$

由于与辐照度卷积相同的(性能)原因，我们不能实时求解积分的镜面部分，期望得到合理的性能。所以我们最好预先计算这个积分得到像高光IBL映射这样的东西，用片段的法线采样这个映射然后用它来完成。然而，这就有点棘手了。我们能够pre-compute辐照度图的积分只取决于ωi我们可以移动不变散射反照率计算的积分。这一次,积分不仅仅取决于ωi明显的双向反射:

$$f_r(p, w_i, w_o) = \frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)}$$

这一次，积分也依赖于$w_o$，我们不能用两个方向向量对一个预先计算好的cubemap进行采样。这里的位置$p$与前面的教程中描述的不相关。对$\omega_i和\omega_o$的每个可能组合预先计算此积分在实时设置中不实用。

Epic Games的split sum近似解决了这一问题，它将预计算拆分为两个单独的部分，稍后我们可以将这些部分组合起来得到我们想要的预计算结果。分裂和近似将镜面积分分解为两个独立的积分:

$$L_o(p,\omega_o) = \int\limits_{\Omega} L_i(p,\omega_i) d\omega_i*\int\limits_{\Omega} f_r(p, \omega_i, \omega_o) n \cdot \omega_i d\omega_i$$

第一部分(卷积时)称为预滤波环境图，它是(类似于辐照度图)预计算的环境卷积图，但这次考虑了粗糙度。为了增加粗糙度，环境图被更多分散的样本向量卷积，产生更多模糊的反射。对于我们进行卷积的每个粗糙度级别，我们将顺序模糊的结果存储在预先过滤的映射的mipmap级别中。例如，一个预滤波的环境图，它存储了5个mipmap级别中5个不同粗糙度值的预卷积结果如下:

![](../img/pbr/ibl_prefilter_map.png)

我们使用Cook-Torrance BRDF的正态分布函数(NDF)生成样本向量及其散射强度，该函数同时作为法线方向和视图方向的输入。时我们不知道事先视图方向旋绕的环境地图,史诗般的游戏让进一步的近似假设视图方向(因此镜面反射方向)总是等于输出样本ωo方向。这转换成以下代码:

```glsl
vec3 N = normalize(w_o);
vec3 R = N;
vec3 V = R;
```
这样，预先过滤的环境卷积就不需要知道视图的方向。这意味着当我们从如下图所示的角度观察镜面反射时，我们不能得到很好的掠射镜面反射(感谢PBR文章中移动的冻伤);然而，这通常被认为是一种体面的妥协:

![](../img/pbr/ibl_grazing_angles.png)

方程的第二部分等于镜面积分的BRDF部分。如果我们假装传入的光芒完全白色的各个方向$L(p, x) = 1.0$我们可以计算双向反射响应给定一个输入粗糙度和后面一个输入角之间的正常$n$和光线方向,或者$\omega_i$。Epic Games在2D查找纹理(LUT)中存储了预先计算的BRDF对不同粗糙度值的每个正常方向和光照方向组合的响应，该纹理被称为BRDF集成映射。2D查找纹理输出一个比例尺(红色)和一个偏置值(绿色)到表面的菲涅耳响应，给我们分裂镜面积分的第二部分:

![](../img/pbr/ibl_brdf_lut.png)

我们生成纹理查找治疗水平纹理坐标(介于0.0和1.0之间)的一架飞机作为双向输入$n\cdot\omega_i$及其垂直纹理坐标作为输入粗糙度值。使用这个BRDF积分图和预滤波的环境图，我们可以结合两者得到镜面积分的结果:

```glsl
float lod             = getMipLevelFromRoughness(roughness);
vec3 prefilteredColor = textureCubeLod(PrefilteredEnvMap, refVec, lod);
vec2 envBRDF          = texture2D(BRDFIntegrationMap, vec2(NdotV, roughness)).xy;
vec3 indirectSpecular = prefilteredColor * (F * envBRDF.x + envBRDF.y) 
```

这应该会让你对Epic Games的拆分和近似大致接近反射率方程的间接镜面部分有一个大致的了解。现在让我们尝试自己构建预卷积部分。

## Pre-filtering an HDR environment map

对环境映射进行预过滤与对辐照度映射进行卷积非常相似。不同之处在于，我们现在考虑了粗糙度，并在预过滤后的地图的mip级别中按顺序存储更粗糙的反射。
