from radiator_rl.envs.house_env import TimeManager

timemanager = TimeManager(
    dt=600, 
    owner_schedule=[(10,36), (108, 144)], 
    off_peak_schedule=[(0, 36), (132, 144)],
    start_time="2025-01-01 00:00:00"
    )
for _ in range(150):
    print(f"Time: {timemanager.current_time}, \
          Time before owners come back: {timemanager.time_before_owners_come_back} steps, Time before off-peak {timemanager.time_before_off_peak}")
    timemanager.step()