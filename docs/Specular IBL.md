# Specular-IBL

在上一篇教程中，我们将预计算的辐照度贴图和IBL的漫反射(Diffuse)部分结合了起来。在本教程中，我们将关注反射方程的镜面(Specular)部分:

$$L_o(p,\omega_o) = \int\limits_{\Omega} (k_d\frac{c}{\pi} + k_s\frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)})L_i(p,\omega_i) n \cdot \omega_i  d\omega_i$$

你会注意到Cook-Torrance镜面部分(乘以$kS$)在积分上不是常数，它依赖于入射光的方向，也依赖于入射的视线方向。试图对所有入射光方向和所有可能的视线方向一起积分，计算量太大，是无法实时完成的。Epic Games提出了一种解决方案，作了一些妥协，使对镜面部分做实时卷积成为可能，这就是有名的split sum approximation。

split sum approximation将反射方程的镜面反射部分分割成两个独立的部分，我们可以分别对其进行卷积，然后在PBR着色器中重新组合起来，最终实现IBL渲染。与我们对辐照度图进行预卷积的方法类似，split sum approximation做卷积时需要一张HDR环境贴图作为输入。为了理解split sum approximation，再次看下反射方程，但这次我们只关注镜面部分(我们在之前的教程中提取了漫反射部分):

$$L_o(p,\omega_o) = \int\limits_{\Omega} (k_s\frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)}L_i(p,\omega_i) n \cdot \omega_i  d\omega_i=\int\limits_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p,\omega_i) n \cdot \omega_i  d\omega_i$$

由于与辐照度卷积相同的(性能上)原因，对镜面部分实时积分并有不错的性能是不大可能的。所以我们最好对这个积分预计算，并得到一张镜面IBL贴图，然后用的微表面的法向量$N$采样这张贴图，最后得到积分值。但是现在有点棘手，我们能够预计算辐照度图是因为积分只跟$\omega_i$有关，然后我们可以将漫反射常量项提取到积分外面。然而这次,积分不仅仅取决于$\omega_i$,而且明显和BRDF有关:

$$f_r(p, w_i, w_o) = \frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)}$$

这一次，积分也依赖于$w_o$，我们没法使用两个方向向量对一个预计算得到的立方体贴图进行采样。这里的位置$p$与前面的教程中描述的不相关。对$\omega_i和\omega_o$的每种可能组合的积分做预计算，在实时渲染中也是没法实现的。

Epic Games的split sum approximation解决了这一问题，它将方程拆分为两个独立的部分分别做预计算，再将两部分结果重新组合得到预计算的值。split sum approximation将镜面部分的积分分成两个独立的积分:

$$L_o(p,\omega_o) = \int\limits_{\Omega} L_i(p,\omega_i) d\omega_i*\int\limits_{\Omega} f_r(p, \omega_i, \omega_o) n \cdot \omega_i d\omega_i$$

第一部分(卷积时)称为pre-filtered环境贴图，它是(类似于辐照度图)预卷积得到的环境贴图，但这次考虑了粗糙度。为了生成递增的多级粗糙度贴图，环境贴图被多种分散的向量采样并卷积，从而产生多种模糊的效果。对于不同级别的粗糙度，我们将按顺序将模糊的结果存储在pre-filtered贴图的多级mipmap中。例如，一张pre-filtered环境贴图，它存储了5张不同粗糙度的mipmap的预卷积结果如下:

![](../img/pbr/ibl_prefilter_map.png)

我们使用Cook-Torrance BRDF的法线分布函数(NDF)生成采样向量及其散射强度，该函数同时将法向量和视线向量作为输入。在对环境贴图做卷积前，我们并不知道视线方向，Epic Game做了进一步的近似，假设视线方向(就是镜面反射的方向)总是等于采样方向$\omega_o$。就可以转换成以下代码:

```glsl
vec3 N = normalize(w_o);
vec3 R = N;
vec3 V = R;
```
这样，对pre-filtered贴图做卷积时就不需要知道视线方向了。这意味着当我们从如下图所示的角度观察镜面反射时，我们不能得到一个较好的以掠射角为观察角度的镜面反射(感谢the Moving Frostbite to PBR article)，不过这通常被认为是一种合适的妥协:

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

这应该会让你对Epic Games的拆分和近似大致接近反射方程的间接镜面部分有一个大致的了解。现在让我们尝试自己构建预卷积部分。

## Pre-filtering an HDR environment map

对环境映射进行预过滤与对辐照度映射进行卷积非常相似。不同之处在于，我们现在考虑了粗糙度，并在预过滤后的地图的mip级别中按顺序存储更粗糙的反射。
