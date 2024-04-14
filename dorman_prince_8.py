#ПОКА НЕ РОБИТ

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
def runge_kutta_dopri8(equations, t_span, y0, t_eval):
    t0, tf = t_span
    h = (tf - t0) / (len(t_eval) - 1)
    t = t0
    y = np.array(y0)
    sol = [y0]

    while t < tf:
        if t + h > tf:
            h = tf - t
        k1 = np.array(equations(t, y))
        k2 = np.array(equations(t + 1 / 18 * h, y + 1 / 18 * h * k1))
        k3 = np.array(equations(t + 1 / 12 * h, y + 1 / 48 * h * k1 + 1 / 16 * h * k2))
        k4 = np.array(equations(t + 1 / 8 * h, y + 1 / 32 * h * k1 + 1 / 16 * h * k2 + 1 / 8 * h * k3))
        k5 = np.array(equations(t + 5 / 16 * h, y + 5 / 16 * h * k1 - 15 / 64 * h * k2 - 3 / 32 * h * k3 + 15 / 32 * h * k4))
        k6 = np.array(equations(t + 3 / 8 * h, y + 3 / 80 * h * k1 + 3 / 16 * h * k2 + 3 / 20 * h * k3 - 9 / 400 * h * k4 + 3 / 20 * h * k5))
        k7 = np.array(equations(t + 59 / 400 * h, y + 29443841 / 614563906 * h * k1 + 77736538 / 692538347 * h * k2 - 28693883 / 1125000000 * h * k3 + 23124283 / 1800000000 * h * k4 - 3 / 250 * h * k5))
        k8 = np.array(equations(t + 93 / 200 * h, y + 16016141 / 946692911 * h * k1 + 61564180 / 158732637 * h * k2 + 22789713 / 633445777 * h * k3 + 545815736 / 2771057229 * h * k4 - 180193667 / 1043307555 * h * k5 + 1 / 4 * h * k7))
        k9 = np.array(equations(t + 5490023248 / 9719169821 * h, y + 39632708 / 573591083 * h * k1 - 433636366 / 683701615 * h * k2 - 421739975 / 2616292301 * h * k3 + 100302831 / 723423059 * h * k4 + 790204164 / 839813087 * h * k5 + 800635310 / 3783071287 * h * k7))
        k10 = np.array(equations(t + 13 / 20 * h, y + 246121993 / 1340847787 * h * k1 - 37695042795 / 15268766246 * h * k2 - 309121744 / 1061227803 * h * k3 - 12992083 / 490766935 * h * k4 + 6005943493 / 2108947869 * h * k5 + 393006217 / 1396673457 * h * k7 + 123872331 / 1001029789 * h * k8))
        k11 = np.array(equations(t + h, y + -1028468189 / 846180014 * h * k1 + 8478235783 / 508512852 * h * k2 + 1311729495 / 1432422823 * h * k3 - 10304129995 / 1701304382 * h * k4 - 48777925059 / 3047939560 * h * k5 + 15336726248 / 1032824649 * h * k7 - 45442868181 / 3398467696 * h * k8 + 3065993473 / 597172653 * h * k9))
        y += h * (14005451 / 335480064 * k1 + 0 * k2 + 0 * k3 + 0 * k4 + 0 * k5 + 0 * k6 + 0 * k7 + 0 * k8 + 0 * k9 + 0 * k10 + 0 * k11)
        t += h
        sol.append(y.copy())

    return np.array(sol).T[:, :len(t_eval)]  # Ensure correct dimensions


# Начальные условия
y0 = [1.0, 1.0, np.pi / 2, 7.0, 12.0, 25.0]

# Время
t_span = [0, 300]

# Решение системы уравнений
t_eval = np.linspace(t_span[0], t_span[1], 5000)
sol = runge_kutta_dopri8(equations, t_span, y0, t_eval)

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
    ax.set_xlim([-5, 15])
    ax.set_ylim([-10, 10])
    ax.grid(True)
    mass1, = ax.plot([], [], 'bo', markersize=10)
    mass2, = ax.plot([], [], 'ro', markersize=10)
    line_1, = ax.plot([], [], 'k-', linewidth=2)

    trajectory, = ax.plot([], [], 'g--', label='Траектория дубинки')
    trajectory1, = ax.plot([], [], 'b--', label='Траектория массы 1')
    trajectory2, = ax.plot([], [], 'r--', label='Траектория массы 2')
    ax.legend(loc='lower left')

    def update(frame):
        if frame < len(t):  # Гарантируем, что индекс кадра не превышает длину массивов данных
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

    ani = FuncAnimation(fig, update, interval = 25, blit=True)
    plt.show()

animate_baton(t, X, Y, Th, l)
