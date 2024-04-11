import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Заданные параметры
l = 4.0  # длина стержня
m1 = 3.0  # масса массы 1
m2 = 10.0  # масса массы 2
J = 2.0  # момент инерции
g = 9.81  # ускорение свободного падения


# Система уравнений движения
def equations(t, y):
    x, y, theta, vx, vy, omega = y
    dxdt = vx
    dydt = vy
    dthetadt = omega
    dvxdt = 0
    dvydt = -g
    domegadt = 0
    return [dxdt, dydt, dthetadt, dvxdt, dvydt, domegadt]


# Начальные условия
y0 = [1.0, 1.0, np.pi / 2, 4.0, 10.0, 20.0]

# Время
t_span = [0, 3]

# Решение системы уравнений
sol = solve_ivp(equations, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000))

# Извлечение решения
t = sol.t
X, Y, Th = sol.y[0], sol.y[1], sol.y[2]

# Построение графиков
plt.figure(figsize=(10, 8))

# Траектория движения ЦМ дубинки
plt.subplot(2, 1, 1)
plt.plot(X, Y, linewidth=2)
plt.grid(True)
plt.title('Траектория движения ЦМ дубинки')
plt.xlabel('m')
plt.ylabel('m')

# Изменения угла theta со временем
plt.subplot(2, 1, 2)
plt.plot(t, np.degrees(Th), linewidth=2)
plt.grid(True)
plt.title('Изменения угла $\\theta$ со временем')
plt.xlabel('t')
plt.ylabel('deg')

plt.tight_layout()


# Анимация
# Анимация
# Анимация
# Анимация
def animate_baton(t, X, Y, Th, l):
    fig, ax = plt.subplots()
    ax.set_xlim([-5, 15])  # Подгоните границы по вашему усмотрению
    ax.set_ylim([-10, 10])  # Подгоните границы по вашему усмотрению
    ax.grid(True)
    mass1, = ax.plot([], [], 'bo', markersize=10)
    mass2, = ax.plot([], [], 'ro', markersize=10)
    line_1, = ax.plot([], [], 'k-', linewidth=2)

    trajectory, = ax.plot([], [], 'g--', label='Траектория дубинки')
    trajectory1, = ax.plot([], [], 'b--', label='Траектория массы 1')
    trajectory2, = ax.plot([], [], 'r--', label='Траектория массы 2')
    ax.legend(loc='lower left')

    def update(frame):
        x_cm = (m1 * 0 + m2 * l) / (m1 + m2)
        mass1.set_data([X[frame] - x_cm * np.cos(Th[frame])], [Y[frame] - x_cm * np.sin(Th[frame])])
        mass2.set_data([X[frame] + (l - x_cm) * np.cos(Th[frame])], [Y[frame] + (l - x_cm) * np.sin(Th[frame])])
        line_1.set_data([X[frame] - x_cm * np.cos(Th[frame]), X[frame] + (l - x_cm) * np.cos(Th[frame])],
                        [Y[frame] - x_cm * np.sin(Th[frame]), Y[frame] + (l - x_cm) * np.sin(Th[frame])])

        trajectory.set_data(X[:frame + 1], Y[:frame + 1])  # Отображение траектории дубинки
        trajectory1.set_data(X[:frame + 1] - x_cm * np.cos(Th[:frame + 1]),
                              Y[:frame + 1] - x_cm * np.sin(Th[:frame + 1]))  # Траектория массы 1
        trajectory2.set_data(X[:frame + 1] + (l - x_cm) * np.cos(Th[:frame + 1]),
                              Y[:frame + 1] + (l - x_cm) * np.sin(Th[:frame + 1]))  # Траектория массы 2

        return mass1, mass2, line_1, trajectory, trajectory1, trajectory2

    ani = FuncAnimation(fig, update, interval=25, blit=True)
    plt.show()


animate_baton(t, X, Y, Th, l)