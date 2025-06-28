# Introduction to linear algebra

Linear algebra provides the mathematical framework for a lot of operations and algorithms. Key aspects include:
- **Data representation**: Data is often represented in the form of matrices and vectors, which are main objects of linear algebra.
- **Transformations**: Linear algebra allows for transformations of data through matrix operations, which is essential for tasks like scaling, rotating, and translating data in multi-dimensional space.
- **Dimensionality reduction**: Linear algebra provides techniques to reduce the number of attributes while preserving the essential information, making data easier to analyze.
- **Solving equations**: Many ML algos rely on solving systems of linear equations to fit the model to the data.
- **Optimization**: Linear algebra is used in optimization problems, where a goal is to minimize or maximize a function, often involving gradients and Hessians

# Vectors

## Data representation
- A vector is a mathematical object that has both **magnitude** (also known as *length* or *size*) and **direction**.
- In machine learning, vectors provide a structured way to represent **data points**. 
- Each data point can be viewed as a vector in a multi-dimensional space, where each dimension corresponds to a feature.

## Vector operations
- 2 main operations: **addition** and **scalar multiplication**.
- Vector addition involves placing one vector at the end of another, and it is:
    - **commutative**: $ \mathbf{r} + \mathbf{s} = \mathbf{s} + \mathbf{r} $
    - **associative**: $ \mathbf{r} + (\mathbf{s} + \mathbf{t}) = (\mathbf{r} + \mathbf{s}) + \mathbf{t} $
- Scalar multiplication **scales** a vector by a number and **reverses the direction** if the number is negative.
- We can define a **coordinate system** using **basis vectors** to algebraically represent vectors as a list of numeric components which are weights of basis vectors. Algebraically, a vector looks like:
    $$
    \mathbf{r} = 
    \begin{bmatrix}
    r_1 \\
    r_2 \\
    \vdots \\
    r_n
    \end{bmatrix}
    $$
- Now, given the algebraic representation of vectors, vector addition (and subtraction) can be performed by adding (or subtracting) their components, i.e.,:
    $$
    \mathbf{r} + \mathbf{s} = \begin{bmatrix} r_1 \\ r_2 \\ \vdots \\ r_n \end{bmatrix} +
                              \begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_n \end{bmatrix}
                            = \begin{bmatrix} r_1 + s_1 \\ r_2 + s_2 \\ \vdots \\ r_n + s_n \end{bmatrix}
    $$
- Similarly, scalar multiplication can be performed by multiplying each component to the scalar:
    $$
    a \times \mathbf{r} = a \times \begin{bmatrix} r_1 \\ r_2 \\ \vdots \\ r_n \end{bmatrix}
                        = \begin{bmatrix} ar_1 \\ ar_2 \\ \vdots \\ ar_n \end{bmatrix}
    $$

## A vector's size, angle and projection

### The size of a vector
- The length (or size) of a vector is denoted as the modulus of the vector, $|\mathbf{r}|$, and determined by the square root of the sum of the squares of its components, i.e.,
    $$
    |\mathbf{r}| = \sqrt{{r_1}^2 + {r_2}^2 + ... + {r_n}^2}
    $$
- This definition applies universally, regardless of the coordinate system or the physical units of the components.

### Dot Product of Vectors
- The dot product is calculated by multiplying corresponding components of two vectors and summing the results, providing a scalar value.
    $$
    \mathbf{r} \cdot \mathbf{s} = r_1 s_1 + r_2 s_2 + ...+ r_n s_n
    $$
- Key properties of the dot product include:
    - **commutativity**: $\mathbf{r} \cdot \mathbf{s} = \mathbf{s} \cdot \mathbf{r}$
    - **distributivity over addition**: $\mathbf{r} \cdot (\mathbf{s} + \mathbf{t}) = \mathbf{r} \cdot \mathbf{s} + \mathbf{r} \cdot \mathbf{t} $
    - **associativity with scalar multiplication**: $\mathbf{r} \cdot (a \mathbf{s}) = a(\mathbf{r} \cdot \mathbf{s})$

### Cosine and dot product
- From the Cosine Rule, we have:
    $$
    \mathbf{r} \cdot \mathbf{s} = |\mathbf{r}| |\mathbf{s}| \cos\theta
    $$
- The above formula can be used to determine the angle between 2 vectors.
- Interpreting the above formula, we have some indications.
    - If $\theta = 0$, then $\cos\theta = 1$, the vectors are aligned.
    - If $\theta= 90$ degrees, then $\cos\theta = 0$, the dot product is zero, meaning the vectors are orthogonal (perpendicular).
    - If $\theta= 180$ degrees, then $\cos\theta = -1$, the dot product is negative, implying that the vectors are in opposite directions.

### Projection
- Projecting one vector (**s**) onto another vector (**r**) provides a "shadow" of **s** on **r**. If **s** is perpendicular to **r**, then no projection.
- The dot product can be expressed as the product of the magnitude of the destination vector with the scalar projection of the vector being projected.
- The scalar projection of **s** on **r** is calculated by dividing the dot product by the magnitude of **r**, giving a measure of *how much **s** extends in the direction of **r***.
    $$
    \text{Scalar projection of}\ \mathbf{s} \ \text{on}\ \mathbf{r} \ = \frac{\mathbf{r} \cdot \mathbf{s}}{|\mathbf{r}|}
    $$
- The vector projection is *a vector* obtained by multiplying the scalar projection by a unit vector in the direction of **r**, providing both magnitude and direction of the shadow of **s** on **r**.
    $$
    \text{Vector projection of}\ \mathbf{s} \ \text{on}\ \mathbf{r} \ 
    = \mathbf{s'} 
    = \frac{\mathbf{r} \cdot \mathbf{s}}{|\mathbf{r}|} \times \frac{\mathbf{r}}{|\mathbf{r}|} 
    = \frac{\mathbf{r} \cdot \mathbf{s}}{{|\mathbf{r}|}^2} \times \mathbf{r}
    $$

## Changing basis
- **Applications**: Changing basis is an important technique in data science and machine learning, such as in PCA, where data is transformed to highlight important features.
- **Basis vectors**: A basis is a set of vectors that are linearly independent and span a vector space. Changing the basis involves transforming vectors from one coordinate system to another.
- **Transformation Matrices**: When the basis vectors are not orthogonal, matrices are used to perform the transformation. The transformation matrix contains the new basis vectors and is used to convert coordinates from one basis to another.
- **Orthogonal Basis**: When the basis vectors are orthogonal (at 90 degrees to each other), the process of changing basis can be simplified using dot products. This allows for easier calculations when projecting vectors onto new axes.
    - Orthogonality can be tested by checking whether the dot product of each pair of basis vectors is 0.
- **Projection**: When changing to an orthogonal basis, the vector projection can be used in determining how to express the vector in the new coordinate system.
    - Given a vector $\mathbf{v}$ in some original basis, to change it into a new orthogonal basis $\{\mathbf{b_1}, \mathbf{b_2},..., \mathbf{b_n}\}$, the components of $\mathbf{v}$ in the new basis are obtained by projecting $\mathbf{v}$ onto each of the orthogonal basis vectors $\mathbf{b_i}$
        $$
        \text{proj}_{\mathbf{b_i}} \mathbf{v} 
        = \frac{\mathbf{v} \cdot \mathbf{b_i}}{{|\mathbf{b_i}|}^2} \times \mathbf{b_i}
        $$
    - Looking at the right hand side, the first term gives the component of $\mathbf{v}$ in the new orthogonal basis.
    - If the basis vectors are **unit vectors** (i.e., $|\mathbf{b_i}|=1$), this simplifies to:
        $$ \text{proj}_{\mathbf{b_i}} \mathbf{v} = (\mathbf{v} \cdot \mathbf{b_i}) \mathbf{b_i} $$
    - Thus, the coefficients $(\mathbf{v} \cdot \mathbf{b_i})$ give the components of $\mathbf{v}$ in the new orthogonal basis.


## Linear independence
**Linear independence**: A set of vectors is said to be **linearly independent** if no vector in the set can be expressed as a linear combination of the others. Otherwise, the set if **linearly dependent**.

### Algebraic definition:
Given a set of vectors $\mathbf{v_1}, \mathbf{v_2}, ..., \mathbf{v_n}$ in a vector space, they are linearly independent if the only solution to the equation:
    $$
    c_1 \mathbf{v_1} + c_2 \mathbf{v_2} + \cdots + c_n \mathbf{v_n} = \mathbf{0}
    $$
is:
    $$
    c_1 = c_2 = \cdots = c_n = 0
    $$
where:
- $c_1, c_2,..., c_n$ are scalars
- $\mathbf{0}$ is the zero vector.

If there exists a non-trivial solution (i.e., at least 1 $c_i \neq 0$), then the vectors are linearly dependent.

### Geometric Interpretation
2 vectors in 2D or 3D:
- linearly independent if they are not parallel.
- linearly dependent if one vector is a scalar multiple of the other.

3 vectors in 3D:
- linearly independent if they do not all lie in the same plane.

### Test for Linear Independence
1. **Matrix Rank**: Place the vectors as rows or columns of a matrix. The vectors are linear independent if the matrix has full rank.
2. **Determinants**: For *n* vectors in *n*-dimensional space, calculate the determinant of the matrix formed by these vectors. If the determinant is non-zero, the vectors are linearly independent.
