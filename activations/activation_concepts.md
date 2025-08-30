# Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to approximate complex functions.

---

## 1. Identity
**Formula:**
$$ f(x) = x $$
**Derivative:**
$$ f'(x) = 1 $$
- Output = input  
- Rarely used (mostly in output layer for regression)

---

## 2. Sigmoid
**Formula:**
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
**Derivative:**
$$ \sigma'(x) = \sigma(x)(1 - \sigma(x)) $$
- Range: (0, 1)  
- Smooth, probabilistic interpretation  
- Problems: vanishing gradients for large $|x|$  

---

## 3. Tanh
**Formula:**
$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
**Derivative:**
$$ \tanh'(x) = 1 - \tanh^2(x) $$
- Range: (-1, 1)  
- Zero-centered (better than sigmoid)  
- Still suffers from vanishing gradients  

---

## 4. ReLU (Rectified Linear Unit)
**Formula:**
$$ f(x) = \max(0, x) $$
**Derivative:**
$$ f'(x) =
\begin{cases} 
1 & x > 0 \\\\
0 & x \leq 0
\end{cases}
$$
- Very efficient, avoids vanishing gradient (for positive inputs)  
- Issue: **Dying ReLU** → neurons stuck with 0 output  

---

## 5. Leaky ReLU
**Formula:**
$$ f(x) =
\begin{cases}
x & x > 0 \\\\
\alpha x & x \leq 0
\end{cases}
$$
**Derivative:**
$$ f'(x) =
\begin{cases}
1 & x > 0 \\\\
\alpha & x \leq 0
\end{cases}
$$
- $\alpha$ (e.g., 0.01) allows small negative slope  
- Fixes dying ReLU problem  

---

## 6. Softmax
**Formula:**
For a vector $z = [z_1, z_2, ..., z_K]$:
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Properties:**
- Outputs values in the range (0, 1)  
- All outputs sum to **1** → can be interpreted as probabilities  
- Used in **final layer for multi-class classification**  

**Derivative (Jacobian Matrix):**
For softmax output vector $s$:
$$
\frac{\partial s_i}{\partial z_j} =
\begin{cases}
s_i(1 - s_i) & i = j \\\\
- s_i s_j & i \neq j
\end{cases}
$$

**Pros:**
- Probabilistic output, interpretable  
- Works naturally with **cross-entropy loss**  

**Cons:**
- Can be computationally expensive for large number of classes  
- Sensitive to large input values (often stabilized using log-sum-exp trick)  
