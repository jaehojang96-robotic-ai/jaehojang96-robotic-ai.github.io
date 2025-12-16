# Extended Kalman Filter for IMU-Based Orientation Estimation

## Complete Mathematical Framework & Python Implementation

---

## Part 1: Theoretical Foundation

### 1.1 State Vector Definition

The state vector contains the quaternion representation of orientation and gyroscope bias estimates:

\[
\mathbf{x}_k = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ b_x \\ b_y \\ b_z \end{bmatrix}_{7 \times 1}
\]

Where:
- **Quaternion** \(\mathbf{q} = [q_0, q_1, q_2, q_3]^T\) (scalar-first convention): Represents 3D rotation
  - \(q_0\) is the scalar part (related to rotation angle)
  - \([q_1, q_2, q_3]^T\) is the vector part (rotation axis)
  - Normalized constraint: \(q_0^2 + q_1^2 + q_2^2 + q_3^2 = 1\)

- **Gyro Bias** \(\mathbf{b} = [b_x, b_y, b_z]^T\): Accounts for gyroscope constant bias

### 1.2 Quaternion Properties

**Quaternion Multiplication** (needed for rotation composition):

\[
\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix} 
q_{1,0}q_{2,0} - q_{1,1}q_{2,1} - q_{1,2}q_{2,2} - q_{1,3}q_{2,3} \\
q_{1,0}q_{2,1} + q_{1,1}q_{2,0} + q_{1,2}q_{2,3} - q_{1,3}q_{2,2} \\
q_{1,0}q_{2,2} - q_{1,1}q_{2,3} + q_{1,2}q_{2,0} + q_{1,3}q_{2,1} \\
q_{1,0}q_{2,3} + q_{1,1}q_{2,2} - q_{1,2}q_{2,1} + q_{1,3}q_{2,0}
\end{bmatrix}
\]

**Matrix Form** (useful for computational efficiency):

\[
\mathbf{q}_1 \otimes \mathbf{q}_2 = \Omega(\mathbf{q}_1)\mathbf{q}_2
\]

Where the left multiplication matrix is:

\[
\Omega(\mathbf{q}) = \begin{bmatrix}
q_0 & -q_1 & -q_2 & -q_3 \\
q_1 & q_0 & -q_3 & q_2 \\
q_2 & q_3 & q_0 & -q_1 \\
q_3 & -q_2 & q_1 & q_0
\end{bmatrix}
\]

**Quaternion Conjugate**:

\[
\mathbf{q}^* = [q_0, -q_1, -q_2, -q_3]^T
\]

**Quaternion to Rotation Matrix**:

\[
\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
1 - 2(q_2^2 + q_3^2) & 2(q_1q_2 - q_0q_3) & 2(q_1q_3 + q_0q_2) \\
2(q_1q_2 + q_0q_3) & 1 - 2(q_1^2 + q_3^2) & 2(q_2q_3 - q_0q_1) \\
2(q_1q_3 - q_0q_2) & 2(q_2q_3 + q_0q_1) & 1 - 2(q_1^2 + q_2^2)
\end{bmatrix}
\]

### 1.3 Continuous-Time State Model

**Quaternion Kinematics** (how quaternion evolves with angular velocity):

\[
\dot{\mathbf{q}} = \frac{1}{2}\Omega(\mathbf{q})(\boldsymbol{\omega} - \mathbf{b})
\]

Expanded form (component-wise):

\[
\begin{align}
\dot{q}_0 &= \frac{1}{2}[-q_1(\omega_x - b_x) - q_2(\omega_y - b_y) - q_3(\omega_z - b_z)] \\
\dot{q}_1 &= \frac{1}{2}[q_0(\omega_x - b_x) + q_2(\omega_z - b_z) - q_3(\omega_y - b_y)] \\
\dot{q}_2 &= \frac{1}{2}[q_0(\omega_y - b_y) + q_3(\omega_x - b_x) - q_1(\omega_z - b_z)] \\
\dot{q}_3 &= \frac{1}{2}[q_0(\omega_z - b_z) + q_1(\omega_y - b_y) - q_2(\omega_x - b_x)]
\end{align}
\]

**Gyro Bias Model** (bias evolves as random walk):

\[
\dot{\mathbf{b}} = \mathbf{0}_{3 \times 1}
\]

This assumes the bias is slowly time-varying, modeled as a random walk in the EKF process noise.

### 1.4 Discrete-Time State Propagation (Prediction Step)

Using **first-order Euler integration** with timestep \(\Delta t\):

\[
\hat{\mathbf{q}}_k^- = \hat{\mathbf{q}}_{k-1}^+ + \Delta t \cdot \dot{\hat{\mathbf{q}}}_{k-1}^+
\]

\[
\hat{\mathbf{b}}_k^- = \hat{\mathbf{b}}_{k-1}^+
\]

More explicitly, the quaternion update becomes:

\[
\hat{\mathbf{q}}_k^- = \hat{\mathbf{q}}_{k-1}^+ + \frac{\Delta t}{2}\Omega(\hat{\mathbf{q}}_{k-1}^+)(\boldsymbol{\omega}_k - \hat{\mathbf{b}}_{k-1}^+)
\]

**Normalization** (critical for numerical stability):

\[
\hat{\mathbf{q}}_k^- \leftarrow \frac{\hat{\mathbf{q}}_k^-}{\|\hat{\mathbf{q}}_k^-\|}
\]

### 1.5 State Transition Jacobian (Prediction Covariance Update)

The Jacobian \(\mathbf{F}_k\) for EKF linearization:

\[
\mathbf{F}_k = \frac{\partial \mathbf{x}_k}{\partial \mathbf{x}_{k-1}}\bigg|_{\hat{\mathbf{x}}_{k-1}^+}
\]

Breaking into blocks:

\[
\mathbf{F}_k = \begin{bmatrix}
\mathbf{F}_{q,q} & \mathbf{F}_{q,b} \\
\mathbf{0}_{3 \times 4} & \mathbf{I}_{3 \times 3}
\end{bmatrix}
\]

Where \(\mathbf{F}_{q,q}\) comes from quaternion kinematics (4×4):

\[
\mathbf{F}_{q,q} = \mathbf{I}_{4 \times 4} + \frac{\Delta t}{2}\Omega_w
\]

With \(\Omega_w = \Omega(\boldsymbol{\omega}_k - \hat{\mathbf{b}}_{k-1}^+)\)

And the coupling term (4×3):

\[
\mathbf{F}_{q,b} = -\frac{\Delta t}{2}\Omega(\hat{\mathbf{q}}_{k-1}^+)
\]

**Covariance Prediction**:

\[
\mathbf{P}_k^- = \mathbf{F}_k \mathbf{P}_{k-1}^+ \mathbf{F}_k^T + \mathbf{Q}_k
\]

Where \(\mathbf{Q}_k\) is the **process noise covariance** (7×7):

\[
\mathbf{Q}_k = \begin{bmatrix}
\sigma_q^2 \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 3} \\
\mathbf{0}_{3 \times 4} & \sigma_b^2 \mathbf{I}_{3 \times 3}
\end{bmatrix}
\]

Typical values: \(\sigma_q \approx 10^{-4}\), \(\sigma_b \approx 10^{-6}\)

---

## Part 2: Measurement Model & Update Step

### 2.1 Accelerometer Measurement Model

**Physical principle**: In the inertial frame, acceleration measured by accelerometer should equal gravity:

\[
\mathbf{a}_{measured} = \mathbf{R}(\mathbf{q}) \mathbf{g}_{body} + \mathbf{n}_a
\]

Where:
- \(\mathbf{g}_{body}\) = normalized gravity in body frame (approximately [0, 0, -g] for stationary)
- \(\mathbf{n}_a\) = accelerometer measurement noise
- \(\mathbf{R}(\mathbf{q})\) = rotation matrix from quaternion

**Observation function**:

\[
\mathbf{z}_{acc,k} = \mathbf{h}_{acc}(\mathbf{x}_k) = \mathbf{R}(\mathbf{q}_k) \begin{bmatrix} 0 \\ 0 \\ -9.81 \end{bmatrix} + \mathbf{n}_a
\]

In expanded form (using rotation matrix elements):

\[
\begin{align}
z_{acc,x} &= 9.81 \cdot 2(q_1q_3 - q_0q_2) + n_{a,x} \\
z_{acc,y} &= 9.81 \cdot 2(q_2q_3 + q_0q_1) + n_{a,y} \\
z_{acc,z} &= 9.81(1 - 2(q_1^2 + q_2^2)) + n_{a,z}
\end{align}
\]

### 2.2 Observation Jacobian for Accelerometer

The Jacobian is the partial derivative with respect to state:

\[
\mathbf{H}_{acc} = \frac{\partial \mathbf{h}_{acc}}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_k^-}
\]

Taking derivatives of the observation function:

\[
\mathbf{H}_{acc} = g \begin{bmatrix}
-2q_2 & -2q_3 & 2q_0 & 2q_1 & 0 & 0 & 0 \\
2q_1 & 2q_0 & 2q_3 & -2q_2 & 0 & 0 & 0 \\
-4q_1 & -4q_2 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\]

Where this is evaluated at predicted state \(\hat{\mathbf{x}}_k^-\).

### 2.3 Measurement Noise Covariance

\[
\mathbf{R}_{acc} = \sigma_{acc}^2 \mathbf{I}_{3 \times 3}
\]

Typical accelerometer noise: \(\sigma_{acc} \approx 0.05\) m/s²

---

## Part 3: EKF Update Step

### 3.1 Innovation (Measurement Residual)

\[
\boldsymbol{\gamma}_k = \mathbf{z}_{acc,k} - \mathbf{h}_{acc}(\hat{\mathbf{x}}_k^-)
\]

### 3.2 Innovation Covariance

\[
\mathbf{S}_k = \mathbf{H}_{acc} \mathbf{P}_k^- \mathbf{H}_{acc}^T + \mathbf{R}_{acc}
\]

### 3.3 Kalman Gain

\[
\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_{acc}^T \mathbf{S}_k^{-1}
\]

### 3.4 State Update

\[
\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \boldsymbol{\gamma}_k
\]

**Special handling for quaternion**: After update, re-normalize:

\[
\hat{\mathbf{q}}_k^+ \leftarrow \frac{\hat{\mathbf{q}}_k^+}{\|\hat{\mathbf{q}}_k^+\|}
\]

### 3.5 Covariance Update

\[
\mathbf{P}_k^+ = (\mathbf{I}_7 - \mathbf{K}_k \mathbf{H}_{acc})\mathbf{P}_k^-
\]

Or the Joseph form (more numerically stable):

\[
\mathbf{P}_k^+ = (\mathbf{I}_7 - \mathbf{K}_k \mathbf{H}_{acc})\mathbf{P}_k^-(\mathbf{I}_7 - \mathbf{K}_k \mathbf{H}_{acc})^T + \mathbf{K}_k \mathbf{R}_{acc} \mathbf{K}_k^T
\]

---

## Part 4: Initialization

### 4.1 Quaternion Initialization

From initial accelerometer reading (assuming static):

\[
\mathbf{a}_{init} = [a_x, a_y, a_z]^T
\]

Compute roll and pitch:

\[
\phi = \arctan2(a_y, -a_z)
\]

\[
\theta = \arctan2(a_x, \sqrt{a_y^2 + a_z^2})
\]

Convert to quaternion:

\[
\begin{align}
q_0 &= \cos(\phi/2)\cos(\theta/2) \\
q_1 &= \sin(\phi/2)\cos(\theta/2) \\
q_2 &= \cos(\phi/2)\sin(\theta/2) \\
q_3 &= \sin(\phi/2)\sin(\theta/2)
\end{align}
\]

### 4.2 Covariance Initialization

\[
\mathbf{P}_0^+ = \begin{bmatrix}
0.1^2 \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 3} \\
\mathbf{0}_{3 \times 4} & 0.01^2 \mathbf{I}_{3 \times 3}
\end{bmatrix}
\]

---

## Part 5: Complete Algorithm Summary

**At each timestep k:**

1. **Prediction Phase** (using gyroscope at high frequency):
   - Update quaternion using kinematics equation
   - Normalize quaternion
   - Compute state Jacobian \(\mathbf{F}_k\)
   - Propagate covariance: \(\mathbf{P}_k^- = \mathbf{F}_k \mathbf{P}_{k-1}^+ \mathbf{F}_k^T + \mathbf{Q}_k\)

2. **Update Phase** (when accelerometer measurement arrives, lower frequency):
   - Compute predicted accelerometer measurement: \(\hat{\mathbf{a}}_k^- = \mathbf{h}_{acc}(\hat{\mathbf{x}}_k^-)\)
   - Compute observation Jacobian: \(\mathbf{H}_{acc}\)
   - Compute innovation: \(\boldsymbol{\gamma}_k = \mathbf{a}_{measured} - \hat{\mathbf{a}}_k^-\)
   - Compute innovation covariance: \(\mathbf{S}_k = \mathbf{H}_{acc} \mathbf{P}_k^- \mathbf{H}_{acc}^T + \mathbf{R}_{acc}\)
   - Compute Kalman gain: \(\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_{acc}^T \mathbf{S}_k^{-1}\)
   - Update state: \(\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \boldsymbol{\gamma}_k\)
   - Normalize quaternion in updated state
   - Update covariance: \(\mathbf{P}_k^+ = (\mathbf{I}_7 - \mathbf{K}_k \mathbf{H}_{acc})\mathbf{P}_k^-\)

---

## Part 6: Python Implementation

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IMUConfig:
    """Configuration for EKF"""
    g: float = 9.81  # Gravity magnitude
    
    # Process noise
    q_var: float = 1e-4  # Quaternion variance
    b_var: float = 1e-6  # Bias variance
    
    # Measurement noise
    acc_var: float = 0.05**2  # Accelerometer variance
    
    dt: float = 0.01  # Timestep in seconds

class QuaternionEKF:
    """Extended Kalman Filter for IMU-based orientation estimation"""
    
    def __init__(self, config: IMUConfig = IMUConfig()):
        self.config = config
        self.state = None
        self.covariance = None
        self.initialized = False
        
    def initialize(self, acc_init: np.ndarray, gyro_init: np.ndarray = None):
        """
        Initialize EKF with first accelerometer reading
        
        Args:
            acc_init: Initial acceleration vector [ax, ay, az] in m/s^2
            gyro_init: Optional initial gyro bias estimate
        """
        # Normalize accelerometer reading
        a_norm = acc_init / np.linalg.norm(acc_init)
        
        # Extract roll and pitch from accelerometer
        phi = np.arctan2(a_norm[1], -a_norm[2])  # Roll
        theta = np.arctan2(a_norm[0], 
                          np.sqrt(a_norm[1]**2 + a_norm[2]**2))  # Pitch
        
        # Convert Euler angles to quaternion
        q = self._euler_to_quaternion(phi, theta, 0.0)
        
        # Initialize state [q0, q1, q2, q3, bx, by, bz]
        self.state = np.zeros(7)
        self.state[:4] = q
        if gyro_init is not None:
            self.state[4:7] = gyro_init
        else:
            self.state[4:7] = np.zeros(3)  # Zero initial bias
        
        # Initialize covariance
        self.covariance = np.eye(7)
        self.covariance[:4, :4] *= 0.1**2  # Quaternion uncertainty
        self.covariance[4:7, 4:7] *= 0.01**2  # Bias uncertainty
        
        self.initialized = True
        
    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion.
        
        Args:
            roll: Rotation around x-axis (phi)
            pitch: Rotation around y-axis (theta)
            yaw: Rotation around z-axis (psi)
            
        Returns:
            Quaternion [q0, q1, q2, q3] where q0 is scalar part
        """
        # Half angles
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        q = np.zeros(4)
        q[0] = cr * cp * cy + sr * sp * sy  # q0 (scalar)
        q[1] = sr * cp * cy - cr * sp * sy  # q1
        q[2] = cr * sp * cy + sr * cp * sy  # q2
        q[3] = cr * cp * sy - sr * sp * cy  # q3
        
        return q
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q1 ⊗ q2
        
        Args:
            q1: First quaternion [q0, q1, q2, q3]
            q2: Second quaternion [q0, q1, q2, q3]
            
        Returns:
            Product quaternion [q0, q1, q2, q3]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def _quaternion_omega_matrix(omega: np.ndarray) -> np.ndarray:
        """
        Create the omega matrix for quaternion kinematics.
        
        The quaternion derivative is: dq/dt = 0.5 * Omega(omega) * q
        
        Args:
            omega: Angular velocity [wx, wy, wz]
            
        Returns:
            4x4 omega matrix
        """
        wx, wy, wz = omega
        
        return 0.5 * np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        R = I + 2*qw*[qv]_x + 2*[qv]_x^2
        
        Args:
            q: Quaternion [q0, q1, q2, q3]
            
        Returns:
            3x3 rotation matrix
        """
        q0, q1, q2, q3 = q
        
        # Ensure unit quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q0, q1, q2, q3 = q / q_norm
        
        return np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
    
    def predict(self, gyro: np.ndarray) -> None:
        """
        Prediction step: propagate state and covariance using gyroscope.
        
        Args:
            gyro: Gyroscope measurement [wx, wy, wz] in rad/s
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized. Call initialize() first.")
        
        q = self.state[:4].copy()
        b = self.state[4:7].copy()
        
        # Remove bias from gyroscope measurement
        omega_corrected = gyro - b
        
        # === Quaternion Update ===
        # dq/dt = 0.5 * Omega(omega) * q
        omega_matrix = self._quaternion_omega_matrix(omega_corrected)
        q_derivative = omega_matrix @ q
        
        # Euler integration: q_new = q_old + dt * dq/dt
        q_new = q + self.config.dt * q_derivative
        
        # Normalize quaternion
        q_norm = np.linalg.norm(q_new)
        q_new = q_new / q_norm
        
        # === Bias remains constant ===
        b_new = b
        
        # Update state
        self.state[:4] = q_new
        self.state[4:7] = b_new
        
        # === Covariance Prediction ===
        # F = [dq/dq, dq/db; db/dq, db/db]
        
        # Partial derivatives of q_new with respect to q and b
        # Since q_new = q + 0.5*dt*Omega(omega_corrected)*q
        # dq_new/dq = I + 0.5*dt*Omega(omega_corrected)
        dq_dq = np.eye(4) + 0.5 * self.config.dt * omega_matrix
        
        # dq_new/db = 0.5*dt*Omega(q)*(-I) = -0.5*dt*Omega(q)
        omega_q = self._quaternion_omega_matrix(q)
        dq_db = -0.5 * self.config.dt * omega_q
        
        # F matrix (7x7)
        F = np.eye(7)
        F[:4, :4] = dq_dq
        F[:4, 4:7] = dq_db
        # Bias part: db/db = I, db/dq = 0
        
        # Process noise covariance
        Q = np.eye(7)
        Q[:4, :4] *= self.config.q_var
        Q[4:7, 4:7] *= self.config.b_var
        
        # Covariance update: P = F*P*F^T + Q
        self.covariance = F @ self.covariance @ F.T + Q
        
        # Ensure symmetry (numerical stability)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
    
    def update(self, acc: np.ndarray) -> None:
        """
        Update step: correct state and covariance using accelerometer.
        
        Args:
            acc: Accelerometer measurement [ax, ay, az] in m/s^2
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized. Call initialize() first.")
        
        q = self.state[:4].copy()
        R_matrix = self._quaternion_to_rotation_matrix(q)
        
        # === Measurement Model ===
        # Expected accelerometer reading (gravity in body frame)
        g_inertial = np.array([0, 0, self.config.g])
        g_body_expected = R_matrix.T @ g_inertial  # Transform to body frame
        
        # Innovation (measurement residual)
        innovation = acc - g_body_expected
        
        # === Observation Jacobian ===
        # H = ∂h/∂x where h is measurement model
        # We only care about quaternion part since bias doesn't affect acceleration
        H = np.zeros((3, 7))
        
        # Derivatives of g_body = R^T * g_inertial with respect to q
        # This requires the derivative of rotation matrix w.r.t. quaternion
        q0, q1, q2, q3 = q
        
        # dR/dq0, dR/dq1, dR/dq2, dR/dq3 and then multiply by g_inertial
        # Using: g_body = R^T * [0, 0, g] = [2*g*(q1*q3 - q0*q2), 
        #                                      2*g*(q2*q3 + q0*q1), 
        #                                      g*(1 - 2*(q1^2 + q2^2))]
        
        g = self.config.g
        
        H[0, 0] = -2 * g * q2  # ∂(g_body_x)/∂q0
        H[0, 1] = 2 * g * q3   # ∂(g_body_x)/∂q1
        H[0, 2] = -2 * g * q0  # ∂(g_body_x)/∂q2
        H[0, 3] = 2 * g * q1   # ∂(g_body_x)/∂q3
        
        H[1, 0] = 2 * g * q1   # ∂(g_body_y)/∂q0
        H[1, 1] = 2 * g * q0   # ∂(g_body_y)/∂q1
        H[1, 2] = 2 * g * q3   # ∂(g_body_y)/∂q2
        H[1, 3] = 2 * g * q2   # ∂(g_body_y)/∂q3
        
        H[2, 0] = 0            # ∂(g_body_z)/∂q0
        H[2, 1] = -4 * g * q1  # ∂(g_body_z)/∂q1
        H[2, 2] = -4 * g * q2  # ∂(g_body_z)/∂q2
        H[2, 3] = 0            # ∂(g_body_z)/∂q3
        
        # === Kalman Filter Update ===
        # Innovation covariance: S = H*P*H^T + R
        R = self.config.acc_var * np.eye(3)
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain: K = P*H^T*S^(-1)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            S_inv = np.linalg.pinv(S)
        
        K = self.covariance @ H.T @ S_inv
        
        # State update: x = x + K*innovation
        delta_x = K @ innovation
        self.state += delta_x
        
        # Normalize quaternion after update
        self.state[:4] /= np.linalg.norm(self.state[:4])
        
        # Covariance update: P = (I - K*H)*P
        P_new = (np.eye(7) - K @ H) @ self.covariance
        
        # Ensure symmetry (numerical stability)
        self.covariance = 0.5 * (P_new + P_new.T)
        
        # Ensure positive definiteness (optional, but good practice)
        self.covariance = self._ensure_positive_definite(self.covariance)
    
    @staticmethod
    def _ensure_positive_definite(P: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite by adjusting eigenvalues.
        """
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals[eigvals < min_eig] = min_eig
        P_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return P_fixed
    
    def get_quaternion(self) -> np.ndarray:
        """Return current quaternion estimate [q0, q1, q2, q3]"""
        return self.state[:4].copy()
    
    def get_euler_angles(self) -> np.ndarray:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Returns:
            [roll, pitch, yaw] in radians
        """
        q0, q1, q2, q3 = self.state[:4]
        
        # Roll (phi)
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        
        # Pitch (theta)
        pitch = np.arcsin(2*(q0*q2 - q3*q1))
        
        # Yaw (psi)
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        
        return np.array([roll, pitch, yaw])
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Return current rotation matrix from body to inertial frame"""
        return self._quaternion_to_rotation_matrix(self.state[:4])
    
    def get_gyro_bias(self) -> np.ndarray:
        """Return current gyro bias estimate [bx, by, bz]"""
        return self.state[4:7].copy()
    
    def get_state_covariance(self) -> np.ndarray:
        """Return current state covariance matrix (7x7)"""
        return self.covariance.copy()


# ============= Example Usage =============

if __name__ == "__main__":
    # Configuration
    config = IMUConfig(
        dt=0.01,  # 100 Hz
        q_var=1e-4,
        b_var=1e-6,
        acc_var=0.05**2
    )
    
    # Initialize EKF
    ekf = QuaternionEKF(config)
    
    # Initial measurements
    acc_init = np.array([0.1, 0.05, 9.81])  # Slight tilt + gravity
    ekf.initialize(acc_init)
    
    # Simulation: simple rotation
    num_steps = 1000
    dt = config.dt
    
    # Store results
    quaternions = [ekf.get_quaternion()]
    euler_angles = [ekf.get_euler_angles()]
    gyro_biases = [ekf.get_gyro_bias()]
    
    # Simulate constant angular velocity with noise
    true_angular_vel = np.array([0.1, 0.05, 0.02])  # rad/s
    
    for step in range(num_steps):
        # Simulate true rotation
        q_true = ekf._quaternion_multiply(
            ekf._quaternion_multiply(
                quaternions[-1],
                ekf._quaternion_multiply(
                    np.array([1, true_angular_vel[0]*dt/4, 
                             true_angular_vel[1]*dt/4, 
                             true_angular_vel[2]*dt/4]),
                    np.array([1, 0, 0, 0])  # Dummy
                )
            ),
            np.array([1, 0, 0, 0])
        )
        
        # Gyro measurement with noise and bias
        bias = ekf.get_gyro_bias()
        gyro_noise = np.random.normal(0, 0.01, 3)
        gyro_measurement = true_angular_vel + bias + gyro_noise
        
        # Accelerometer measurement
        R_mat = ekf.get_rotation_matrix()
        g_inertial = np.array([0, 0, 9.81])
        acc_true = R_mat.T @ g_inertial
        acc_noise = np.random.normal(0, 0.05, 3)
        acc_measurement = acc_true + acc_noise
        
        # EKF prediction and update
        ekf.predict(gyro_measurement)
        
        # Update every 2 steps (accelerometer at 50 Hz)
        if step % 2 == 0:
            ekf.update(acc_measurement)
        
        # Store results
        quaternions.append(ekf.get_quaternion())
        euler_angles.append(ekf.get_euler_angles())
        gyro_biases.append(ekf.get_gyro_bias())
    
    # Convert to arrays
    quaternions = np.array(quaternions)
    euler_angles = np.array(euler_angles)
    gyro_biases = np.array(gyro_biases)
    
    # Print final results
    print("=== EKF Final Results ===")
    print(f"Final Quaternion: {ekf.get_quaternion()}")
    print(f"Final Euler Angles (deg): {np.degrees(ekf.get_euler_angles())}")
    print(f"Final Gyro Bias (rad/s): {ekf.get_gyro_bias()}")
    print(f"State Covariance Diagonal: {np.diag(ekf.get_state_covariance())}")
```

---

## Part 7: Key Implementation Notes

### 7.1 Numerical Stability

1. **Quaternion Normalization**: Always normalize after integration and update
2. **Covariance Symmetry**: Enforce P = 0.5*(P + P^T) after each update
3. **Positive Definiteness**: Check eigenvalues of P remain positive

### 7.2 Parameter Tuning

- **Q (process noise)**: Higher values trust measurements more, lower values trust prediction more
- **R (measurement noise)**: Should match actual sensor noise characteristics
- **Initial P**: Large values indicate high initial uncertainty

### 7.3 Common Pitfalls

- ❌ Not normalizing quaternions → divergence
- ❌ Incorrect Jacobian computation → filter becomes suboptimal
- ❌ Using wrong quaternion convention (scalar-last vs scalar-first)
- ✓ Always use consistent convention throughout

### 7.4 Extensions

For production systems, consider:
- Adding magnetometer for yaw drift correction
- Adaptive noise covariance (based on innovation magnitude)
- Robust EKF variants (M-estimators for outlier rejection)
- Error-State EKF formulation (better for large rotations)
