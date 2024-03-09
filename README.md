# analytic_grad_shafranov #
Analytic description of Grad Shafranov equation.

We apply a method described in [Antoine J. Cerfon, Jeffrey P. Freidberg; “One size fits all” analytic solutions to the Grad–Shafranov equation. Phys. Plasmas 1 March 2010; 17 (3): 032502.](https://doi.org/10.1063/1.3328818) modified slightly for favourable sign conventions.

## Method ##

The Grad-Shafranov equation is

$$ R \frac{\partial}{\partial R} \left( \frac{1}{R} \frac{\partial \Psi}{\partial R} \right) + \frac{\partial^2 \Psi}{\partial Z^2} = -\mu_0 p'(\Psi) R^2 - F(\Psi) F'(\Psi) = -\mu_0 R j_\phi $$

The toroidal current density $j_\phi$ has a contribution from the plasma pressure $p(\Psi)$ and the current function $F(\Psi)$. We introduce normalised coordinates $x = R/R_0$, $y=Z/R_0$ and $\psi = \Psi / \Psi_0$ where $R_0$ is the major radius [m] and $\Psi_0$ is an arbitrary constant [Wb]. The Grad-Shafranov equation in these normalised co-ordinates is

$$ x \frac{\partial}{\partial x} \left( \frac{1}{x} \frac{\partial \psi}{\partial x} \right) + \frac{\partial^2 \psi}{\partial y^2} = -\mu_0 \frac{R_0^4}{\Psi_0^2} p'(\psi) x^2 - \frac{R_0^2}{\Psi_0^2} F(\psi) F'(\psi) $$

The well known choices for $p$ and $F$ were given by Solov'ev

$$ -\mu_0 \frac{R_0^4}{\Psi_0^2} p'(\psi) = C \qquad - \frac{R_0^2}{\Psi_0^2} F(\psi) F'(\psi) = A $$

where $A, C$ are constants. As $\Psi_0$ is arbitrary we can rescale it to enforce $A + C = 1$. We also note we can rescale $\psi$ by an arbitary additive constant so the paper can define the seperatrix as the surface $\psi(x, y) = 0$.

We also diverge from the paper by setting $A = -A$ and $\psi =-\psi$ to eliminate some minus signs and ensure the value of $\psi$ on axis is positive. Note this is equivalent to flipping the sign of $j_\phi$. Therefore the Grad-Shafranov equation becomes

$$ x \frac{\partial}{\partial x} \left( \frac{1}{x} \frac{\partial \psi}{\partial x} \right) + \frac{\partial^2 \psi}{\partial y^2} = -A + (1 + A)x^2 $$

$$ \mu_0 \frac{R_0^4}{\Psi_0^2} p'(\psi) = 1 + A \qquad \frac{R_0^2}{\Psi_0^2} F(\psi) F'(\psi) = -A $$

We will see $\Psi_0$ is related to the total plasma current and $A$ is a parameter relating to the total pressure. The paper proceeds by writing $\psi$ as a linear combination of polynomials $\psi_i$ which is truncated at order $x^6, y^6$ e.g.

$$ \psi(x, y) = \sum_i c_i \psi_i(x, y) $$

For up-down symmetric plasmas e.g. Limiter, Double Null only even functions in $y$ are retained. Otherwise for e.g. Single Null, functions which are odd in $y$ are also retained. The weights are determined by fitting the limiter contour $\psi(x, y) = 0$ to the classic D shaped model in inverse aspect ratio $\epsilon = a / R_0$, elongation $\kappa$ and triangularity $\delta$.

$$ x = 1 + \epsilon \cos \left( \theta + \alpha \sin \theta \right) \qquad y = \kappa \epsilon \sin \left( \theta \right) \qquad \alpha = \arcsin \left( \delta \right) $$

The exact conditions are explained in the paper. To summarise we form a linear system we solve for the coefficients $c_i$. We shall instead skip to a discussion of 'figures of merit'.

Firstly we note we can solve directly for $p(\psi)$ and $F(\psi)$

$$ p(\psi) = \frac{\Psi_0^2}{\mu_0 R_0^4} (1 + A) \psi \qquad F(\psi) = R_0^2 \left( B_0^2 - \frac{2 A \Psi_0^2}{R_0^4} \psi \right)^{1/2} $$

We can also write the toroidal current density $j_\phi$ as

$$ \mu_0 R j_\phi = \frac{\Psi_0}{R_0^2} \left( A - (1 + A)x^2 \right) $$

$$ \implies j_\phi(x, y) = \frac{1}{\mu_0} \frac{\Psi_0}{R_0} \left[ \frac{A}{x} - (1 + A)x \right] $$

By integrating over the plasma cross section we can write the total plasma current $I_p$ as

$$ I_p = \frac{1}{\mu_0} \frac{\Psi_0}{R_0} \int \left[ \frac{A}{x} - (1 + A)x \right] dx dy = \frac{1}{\mu_0} \frac{\Psi_0}{R_0} I_1 $$
$$ I_1 := \int \left[ \frac{A}{x} - (1 + A)x \right] dx dy $$

Once we choose $A$ we can therefore set the correct plasma current by setting $\Psi_0 = \mu_0 I_p R_0 / I_1$.

We can also find the volume averaged plasma pressure $<p> = \int p dV / \int dV$

$$ <p>= \frac{\int p(R, Z) 2\pi R dR dZ}{\int 2 \pi R dR dZ} = \frac{2 \pi R_0^3 \int p(x, y) x dx dy}{2 \pi R_0^3 \int x dx dy} = \frac{1}{V} \int p(x, y) x dx dy $$

where we have introduced the normalised volume $V = \int x dx dy$. Substituting in the expression for $p(x, y)$ from above

$$ <p> = \frac{\Psi_0^2}{\mu_0 R_0^4} (1 + A) \int \psi(x, y) x dx dy = \frac{\Psi_0^2}{\mu_0 R_0^4} (1 + A) I_2 $$
$$ I_2 = \int \psi(x, y) x dx dy $$

The poloidal beta is defined as $\beta_p = 2 \mu_0 <p> / \bar{B}_p^2 $ where $\bar{B} = \mu_0 I / R_0 C_p$ is the average poloidal magnetic field on the boundary $\psi = 0$ and $C_p$ is the plasma cross section circumference normalised by $R_0$. Substituting in the above expressions

$$ \beta_p = \frac{2 \mu_0 \frac{\Psi_0^2}{\mu_0 R_0^4} (1 + A) \frac{I_2}{V}}{\left(\frac{\mu_0 I}{R_0 C_p}\right)^2} = \frac{2 \frac{\Psi_0^2}{R_0^4} (1 + A) \frac{I_2}{V}}{\frac{(\Psi_0 I_1)^2}{R_0^4 C_p^2}} = 2(1 + A) \frac{C_p^2}{V} \frac{I_2}{I_1^2} $$
