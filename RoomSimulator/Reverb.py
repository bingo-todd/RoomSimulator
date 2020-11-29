import numpy as np


def Absorb2RT(A, room_size, F_abs=None, c=343, A_air=None,
              estimator='Norris_Eyring'):
    """ Estimate reverberation time based on room acoustic parameters,
    translated from matlab code developed by Douglas R Campbell
    Args:
        A: sound absorption coefficients of six wall surfaces
        room_size: three-dimension measurement of shoebox room
        c: sound speed, default to 343 m/s
        F_abs: center frequency of each frequency band
        A_air: absorption coefficients of air, if not specified, it will
            calculated based on humidity of 50
        estimator: estimate methods, choose from [Sabine,SabineAir,
            SabineAirHiAbs,Norris_Eyring], default to Norris_Eyring
    """
    if F_abs is None:
        F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])

    if A_air is None:
        humidity = 50
        A_air = 5.5e-4 * (50 / humidity) * ((F_abs / 1000) ** 1.7)

    V_room = np.prod(room_size)  # Volume of room m^3
    S_wall_all = [room_size[0] * room_size[2],
                  room_size[1] * room_size[2],
                  room_size[0] * room_size[1]]
    S_room = 2. * np.sum(S_wall_all)  # Total area of shoebox room surfaces
    # Effective absorbing area of room surfaces at each frequency
    Se = (S_wall_all[1] * (A[:, 0] + A[:, 1])
          + S_wall_all[0] * (A[:, 2] + A[:, 3])
          + S_wall_all[2] * (A[:, 4] + A[:, 5]))
    A_mean = Se / S_room  # Mean absorption of wall surfaces

    # Reverberation time estimate
    if np.linalg.norm(1 - A_mean) < np.finfo(float).eps:
        # anechoic room, force RT60 all zeros
        RT60 = np.zeros(F_abs.shape)
    else:
        # Select an estimation equation
        if estimator == 'Sabine':
            RT60 = np.divide((55.25 / c) * V_room, Se)  # Sabine equation
        if estimator == 'SabineAir':
            # Sabine equation (SI units) adjusted for air
            RT60 = np.divide((55.25 / c) * V_room, (4 * A_air * V_room + Se))
        if estimator == 'SabineAirHiAbs':
            # % Sabine equation (SI units) adjusted for air and high absorption
            RT60 = np.divide(55.25 / c * V_room,
                             4*A_air*V_room+np.multiply(Se, (1+A_mean/2)))
        if estimator == 'Norris_Eyring':
            # Norris-Eyring estimate adjusted for air absorption
            RT60 = np.divide(55.25 / c * V_room,
                             (4*A_air*V_room
                              - S_room*np.log(1-A_mean+np.finfo(float).eps)))

        return RT60


def RT2Absorb(RT60, room_size, F_abs=None, c=343, A_air=None,
              estimator='Norris_Eyring'):
    if np.max(RT60) < 1e-10:
        return np.ones((6, 6), dtype=np.float32)

    if F_abs is None:
        F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])

    if A_air is None:
        humidity = 50
        A_air = 5.5e-4 * (50 / humidity) * ((F_abs / 1000) ** 1.7)

    V_room = np.prod(room_size)  # Volume of room m^3
    S_wall_all = [room_size[0] * room_size[2],
                  room_size[1] * room_size[2],
                  room_size[0] * room_size[1]]
    S_room = 2. * np.sum(S_wall_all)  # Total area of shoebox room surfaces

    if estimator == 'Sabine':
        A = np.divide(55.25 / c * V_room / S_room, RT60)
    if estimator == 'SabineAir':
        A = (np.divide(55.25 / c * V_room, RT60) - 4 * A_air * V_room) / S_room
    if estimator == 'SabineAirHiAbs':
        A = np.sqrt(
            2*(np.divide(55.25/c*V_room, RT60)-4*A_air*V_room)+1)-1
    if estimator == 'Norris_Eyring':
        A = 1-np.exp((4*A_air*V_room-np.divide(55.25/c*V_room, RT60))/S_room)
    else:
        A = np.ones(6) * np.Inf

    A = np.repeat(A.reshape([1, 6]), 6, axis=0)  # all 6 wall
    # set precision, two digit after the decimal separator
    A = np.round(A*100)/100
    return A


def test_Absorb2RT():

    # A_acoustic_plaster = np.asarray([0.10,0.20,0.50,0.60,0.70,0.70])
    # A_RT0_5 = np.asarray([0.2136,0.2135,0.2132,0.2123,0.2094,0.1999])
    F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])
    A_RT0_5 = np.asarray([0.2214, 0.3051, 0.6030, 0.7770, 0.8643, 0.8627])
    room_size = (5.1, 7.1, 3)
    A = np.repeat(A_RT0_5.reshape((-1, 1)), repeats=6, axis=1)
    RT = Absorb2RT(room_size=room_size, A=A, F_abs=F_abs)
    print(RT)


def test_RT2Absorb():
    F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])
    A_RT0_5 = np.asarray([0.2136, 0.2135, 0.2132, 0.2123, 0.2094, 0.1999])
    room_size = (5.1, 7.1, 3)
    A = np.repeat(A_RT0_5.reshape((-1, 1)), repeats=6, axis=1)
    RT = Absorb2RT(room_size=room_size, A=A, F_abs=F_abs)
    print(RT)
    A = RT2Absorb(RT60=RT, room_size=room_size, F_abs=F_abs)
    print(A)


if __name__ == '__main__':
    test_RT2Absorb()
