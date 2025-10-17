from datetime import datetime, timedelta

class TimeManager():
    def __init__(self,
                 dt,
                 owner_schedule=[(0, 36), (108, 144)],
                 off_peak_schedule=[(0, 36), (132, 144)],
                 start_time="2025-01-01 00:00:00"
                 ):
        """
        Manages the simulation time and owner presence schedule. Everythin is in time steps of dt seconds.
        Args:
            dt (int): Time step in seconds.
            owner_schedule (list of tuples): List of (start, end) time steps when owners are present.
            start_time (str): Start time in "YYYY-MM-DD HH:MM:SS" format.
        """
        self.dt = timedelta(seconds=dt)
        self.owner_schedule = owner_schedule
        self.off_peak_schedule = off_peak_schedule
        self.start_time = start_time
        self.current_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.current_step = 0
        self.time_before_owners_come_back = self.owner_schedule[0][0]  # Initial time before owners come back (in time steps)
        self.time_before_off_peak = self.off_peak_schedule[0][0]

        # Calculate the number of time steps per day
        self.steps_per_day = int(timedelta(days=1).total_seconds() // self.dt.total_seconds())

    def _is_off_peak(self):
        """Returns True if the current time step is within an off-peak period."""
        current_step_in_day = self.current_step % self.steps_per_day
        for start, end in self.off_peak_schedule:
            if start <= current_step_in_day < end:
                return True
        return False
    
    def _is_owner_present(self):
        """Returns True if the current time step is within an owner presence period."""
        current_step_in_day = self.current_step % self.steps_per_day
        for start, end in self.owner_schedule:
            if start <= current_step_in_day < end:
                return True
        return False

    def _update_time_before_return(self):
        """Updates time_before_owners_come_back based on the current position."""
        current_step_in_day = self.current_step % self.steps_per_day

        if self._is_owner_present():
            self.time_before_owners_come_back = 0
        else:
            # Find the next presence period
            next_start = None
            for start, end in self.owner_schedule:
                if current_step_in_day < start:
                    next_start = start
                    break
            if next_start is not None:
                self.time_before_owners_come_back = next_start - current_step_in_day
            else:
                # If we are past all periods, wait for the first period the next day
                self.time_before_owners_come_back = self.steps_per_day - current_step_in_day + self.owner_schedule[0][0]

    def _update_time_before_off_peak(self):
        """Updates time_before_off_peak based on the current position."""
        current_step_in_day = self.current_step % self.steps_per_day

        if self._is_off_peak():
            self.time_before_off_peak = 0
        else:
            # Find the next presence period
            next_start = None
            for start, end in self.off_peak_schedule:
                if current_step_in_day < start:
                    next_start = start
                    break
            if next_start is not None:
                self.time_before_off_peak = next_start - current_step_in_day
            else:
                # If we are past all periods, wait for the first period the next day
                self.time_before_off_peak = self.steps_per_day - current_step_in_day + self.off_peak_schedule[0][0]

    def step(self):
        self.current_time += self.dt
        self.current_step += 1
        self._update_time_before_return()
        self._update_time_before_off_peak()

    def reset(self):
        self.current_time = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        self.current_step = 0
        self.time_before_owners_come_back = self.owner_schedule[0][0]
        
    @property
    def hour_of_day(self):
        return self.current_time.hour

    @property
    def weekday(self):
        return self.current_time.weekday()
    
