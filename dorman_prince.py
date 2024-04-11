import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Заданные параметры
l = 2.0  # длина стержня
m1 = 3.0  # масса массы 1
m2 = 3.0  # масса массы 2
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


# Метод Дормана-Принса для решения системы ОДУ
def runge_kutta_dopri5(equations, t_span, y0, t_eval):
    t0, tf = t_span
    h = (tf - t0) / (len(t_eval) - 1)
    t = t0
    y = np.array(y0)
    sol = [y0]

    while t < tf:
        if t + h > tf:
            h = tf - t
        k1 = np.array(equations(t, y))
        k2 = np.array(equations(t + 1 / 5 * h, y + 1 / 5 * h * k1))
        k3 = np.array(equations(t + 3 / 10 * h, y + 3 / 40 * h * k1 + 9 / 40 * h * k2))
        k4 = np.array(equations(t + 4 / 5 * h, y + 44 / 45 * h * k1 - 56 / 15 * h * k2 + 32 / 9 * h * k3))
        k5 = np.array(equations(t + 8 / 9 * h, y + 19372 / 6561 * h * k1 - 25360 / 2187 * h * k2 +
                                 64448 / 6561 * h * k3 - 212 / 729 * h * k4))
        k6 = np.array(equations(t + h, y + 9017 / 3168 * h * k1 - 355 / 33 * h * k2 + 46732 / 5247 * h * k3 +
                                 49 / 176 * h * k4 - 5103 / 18656 * h * k5))
        y += h * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
        t += h
        sol.append(y.copy())

    return np.array(sol).T[:, :len(t_eval)]  # Ensure correct dimensions

# Начальные условия
y0 = [1.0, 1.0, np.pi / 2, 7.0, 12.0, 25.0]

# Время
t_span = [0, 3]

# Решение системы уравнений
t_eval = np.linspace(t_span[0], t_span[1], 1000)
sol = runge_kutta_dopri5(equations, t_span, y0, t_eval)

# Извлечение решения
t = t_eval
X, Y, Th = sol[0], sol[1], sol[2]

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

    ani = FuncAnimation(fig, update, frames=len(t), interval=25, blit=True)
    plt.show()


animate_baton(t, X, Y, Th, l)
