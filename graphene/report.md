# Multi-Layer Graphene with Buckling

## Phase Field Crystal Model

The Phase Field Crystal (PFC) model describes the evolution of a coarse-grained microscopic density field, $\psi(\mathbf{r},t)$, which distinguishes between liquid and solid phases. In the solid phase, $\psi$ exhibits spatial periodicity corresponding to the crystal lattice, while in the liquid phase it remains approximately uniform. For multi-layer graphene systems, the PFC model can be extended with a **height field** $h(\mathbf{r},t)$ to capture buckling in the vertical direction. Unlike $\psi$, the height field is **non-conserved**, representing local relaxations of the surface or membrane.

The coarse-grained Hamiltonian functional for the two-field PFC model is:

$$
\mathcal{E}[\psi, h] = \int d\mathbf{r} \left[ \frac{\psi}{2} \left( r + \left( \nabla^2 + q_0^2 \right)^2 \right) \psi + \frac{\psi^4}{4} + \frac{\kappa}{2} (\nabla^2 h)^2 + e_{\rm coupling}(\psi,h) \right],
$$

where $r$ and $q_0$ are parameters controlling the density modulation, $\kappa$ is the bending rigidity of the height field, and $e_{\rm coupling}$ represents coupling between $\psi$ and $h$.

> **Note:** The coupling term between $h$ and $\psi$ also includes nonlinear coupling of the field gradients.

The dynamics of the two fields are governed by:

1. **Conserved dynamics for $\psi$ (Model B):**
$$\frac{\partial \psi}{\partial t} = \nabla^2 \frac{\delta \mathcal{E}}{\delta \psi} = \nabla^2 \left[ \left( r + \left( \nabla^2 + q_0^2 \right)^2 \right) \psi + \psi^3 + \frac{\partial e_{\rm coupling}}{\partial \psi} \right].$$

2. **Non-conserved dynamics for $h$ (Model A):**
$$\frac{\partial h}{\partial t} = - \frac{\delta \mathcal{E}}{\delta h} = - \left[ \kappa \nabla^4 h + \frac{\partial e_{\rm coupling}}{\partial h} \right].$$

### Semi-Implicit Spectral Method

To solve these PDEs efficiently, we use a **semi-implicit spectral method**:

- **Linear terms** (stiff operators) are treated **implicitly in Fourier ($k$-) space**, enabling stable time integration without prohibitively small time steps.
- **Nonlinear terms** (e.g., $\psi^3$ and the coupling contributions) are treated **explicitly in real space**.

The numerical scheme proceeds as follows:

1. Transform $\psi$ and $h$ to Fourier space:
$$\hat{\psi}(\mathbf{k},t) = \mathcal{F}[\psi(\mathbf{r},t)], \quad \hat{h}(\mathbf{k},t) = \mathcal{F}[h(\mathbf{r},t)].$$

2. Compute the nonlinear terms and their Fourier transforms.

3. Advance the **density field** (linear terms implicit, nonlinear terms explicit):
$$\hat{\psi}(\mathbf{k}, t+\Delta t) = \frac{\hat{\psi}(\mathbf{k},t) + \Delta t (-k^2) \left[ \widehat{\psi^3 + \frac{\partial e_{\rm coupling}}{\partial \psi}}(\mathbf{k},t)\right]}{1 - \Delta t (-k^2)  \left[ r - k^2 + k^4 \right]}.$$

4. Advance the **height field** (linear terms implicit, nonlinear terms explicit):
$$\hat{h}(\mathbf{k}, t+\Delta t) = \frac{\hat{h}(\mathbf{k},t) - \Delta t \left[\widehat{\frac{\partial e_{\rm coupling}}{\partial h}} \right]}{1+\Delta t \left[\kappa k^4\right] }.$$

5. Inverse Fourier transform $\hat{\psi}(\mathbf{k}, t+\Delta t)$ and $\hat{h}(\mathbf{k}, t+\Delta t)$ back to real space.

This approach combines the **stability of implicit integration for linear terms** with the **simplicity of explicit treatment for nonlinearities**, making it well-suited for large-scale GPU simulations of multi-layer graphene with buckling.

### Computing Field Derivatives

The energy functional and equations of motion depend on spatial derivatives of the fields. Since a forward Fourier transform is required at every time step anyway, derivatives are most conveniently and accurately computed in $k$-space: $\partial_x f(\mathbf{r}) \leftrightarrow ik_x \hat{f}(\mathbf{k})$ (and analogously for other coordinates). This formula is local in $k$-space, requiring only the value of the field at each wavenumber.

However, the nonlinear coupling terms depend on field derivatives and must be evaluated in real space, necessitating several inverse Fourier transforms per time step. Nearly all versions of the PFC code in this repository use $k$-space derivatives.

As an optimization, we also explored computing derivatives via real-space finite differences, where the derivative at grid point $(i,j)$ depends only on the values at its eight nearest neighbors. This reduced the total number of Fourier transforms by a factor of three, yielding a **2x speed-up**. See the [two-layers-light](two-layers-light) folder for this implementation. Initial tests show that the total energy of the system does not change significantly, suggesting that equilibrium states are nearly identical between the two approaches.

> **Note:** Further testing, optimization, and validation are needed to confirm correctness and efficiency.

---

## Project Development

### Base Code

The starting point was a two-layer, single-GPU code written by CVA and Ken Elder, which had been previously validated and used in research. We first cleaned the code and established test cases.

The program adopts a **structure-of-arrays** approach, with a single struct encapsulating all fields, their derivatives, Fourier transform handles, and intermediate computation buffers:

```c
typedef struct {
   clock_t *ttttime;
   cufftrealvector hone;
   cufftrealvector Chone;
   cufftrealvector dxhone;
   cufftrealvector dxxhone;
   cufftrealvector htwo;
   cufftrealvector Chtwo;
   cufftrealvector dxhtwo;
   cufftrealvector dxxhtwo;
   cufftcomplexvector honek;
   cufftcomplexvector Chonek;
   cufftcomplexvector dxhonek;
   cufftcomplexvector dxxhonek;
   cufftcomplexvector htwok;
   cufftcomplexvector Chtwok;
   cufftcomplexvector dxhtwok;
   cufftcomplexvector dyhtwok;
   cufftrealvector psione;
   cufftrealvector Hpsione;
   cufftrealvector hxhxLmHpsione;
   cufftcomplexvector hxhxLmHpsionek;
   cufftrealvector dxxpsione;
   cufftrealvector Lpsione;
   cufftrealvector nntone;
   cufftcomplexvector psionek;
   cufftcomplexvector Hpsionek;
   cufftcomplexvector Lpsionek;
   cufftcomplexvector dxxpsionek;
   cufftcomplexvector nntonek;
   cufftrealvector Lpsitwo;
   cufftrealvector psitwo;
   cufftrealvector Hpsitwo;
   cufftrealvector hxhxLmHpsitwo;
   cufftcomplexvector hxhxLmHpsitwok;
   cufftrealvector dxxpsitwo;
   cufftrealvector nnttwo;
   cufftcomplexvector psitwok;
   cufftcomplexvector Hpsitwok;
   cufftcomplexvector Lpsitwok;
   cufftcomplexvector dxxpsitwok;
   cufftcomplexvector nnttwok;
   cufftHandle D2Z;
   cufftHandle Z2D;
} field;
```

This two-layer code is hard-coded; extending it to an arbitrary number of layers is not straightforward in its original form.

### Decoupling the Layers

The first step toward a more general code was to refactor the two-layer struct into a single-layer version:

```c
typedef struct {
   clock_t *ttttime;
   cufftrealvector hone;
   cufftrealvector Chone;
   cufftrealvector dxhone;
   cufftrealvector dxxhone;
   cufftcomplexvector honek;
   cufftcomplexvector Chonek;
   cufftcomplexvector dxhonek;
   cufftcomplexvector dxxhonek;
   cufftrealvector psione;
   cufftrealvector Hpsione;
   cufftrealvector hxhxLmHpsione;
   cufftcomplexvector hxhxLmHpsionek;
   cufftrealvector dxxpsione;
   cufftrealvector Lpsione;
   cufftrealvector nntone;
   cufftcomplexvector psionek;
   cufftcomplexvector Hpsionek;
   cufftcomplexvector Lpsionek;
   cufftcomplexvector dxxpsionek;
   cufftcomplexvector nntonek;
   cufftHandle D2Z;
   cufftHandle Z2D;
} field;
```

An $n$-layer system is then represented as an array of `field` of size $n$.

### Multi-GPU Implementation

The ultimate goal is to distribute the computation across multiple GPUs. For generality and ease of development, we adopted a **1 MPI process per GPU** model: each layer is handled by one MPI process using one GPU, so an $n$-layer system requires $n$ MPI processes and $n$ GPUs.

Communication is needed only between nearest-neighbor layers: at each time step, layer $i$ requires data from layers $i-1$ and $i+1$. Because these communications are local, their impact on scalability is small. The current implementation uses **`MPI_Sendrecv`** for neighbor data exchange.

> **Note:** The inter-layer coupling term is not yet included in this implementation.

Despite the missing coupling, scaling tests already show promising results: using two, three, or four layers on a single node, wall time remains approximately constant. Tests across multiple nodes are still needed.

### Bonus: Real-Space Gradient Computation

We also investigated the effect of gradient computation strategy on both wall time and correctness. In the baseline code, gradients are computed by (i) forward Fourier transforming the field, (ii) multiplying by the appropriate operator in $k$-space, and (iii) inverse transforming to obtain real-space gradients. Since Fourier transforms are computationally expensive, we tested replacing this procedure with stencil operations, which require no Fourier transforms and reduced the transform count from nine to three per layer. The resulting speed-up was **2x**. Initial tests show that equilibrium states have the same energy in both versions, though further validation is needed.

---

## References

- Elder, K., et al. [*Modeling buckling and topological defects in stacked two-dimensional layers of graphene and hexagonal boron nitride*](https://www.researchgate.net/publication/350033146_Modeling_buckling_and_topological_defects_in_stacked_two-dimensional_layers_of_graphene_and_hexagonal_boron_nitride). ResearchGate (2021).
