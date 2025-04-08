from enum import Enum
from copy import copy

"""Possible types of growth."""
GrowthType = Enum("Growth", ["LINEAR", "EXPONENTIAL", "LOGISTIC"])


class GrowthController:
    def __init__(self, current_volume, growth_type, growth_rate):
        pass

    def set_pressure(self, pressure):
        """Set new pressure."""
        pass

    def step_growth(self, current_volume, dt):
        """Step through the growth controller to calculate normal growth.

        Returns new pressure.
        """
        pass

    def step_divided(self, current_volume, dt):
        """Step through the growth controller after division."""
        pass


class PIDController(GrowthController):
    """Class to control growth by a PID controller, in which changes to a cell's
    internal pressure governs how much it grows in the next frame.

    Attributes:
        growth_type (Growth): Type of growth exhibited by cells.
        growth_rate (float): Rate of growth of cells.
        initial_pressure (float): Initial pressure of cells.
        target_volume (float): Target volume of cells.
        Kp (float): P variable of the PID controller.
        Ki (float): I variable of the PID controller.
        Kd (float): D variable of the PID controller.
    """

    def __init__(
        self,
        current_volume,
        growth_type=GrowthType.LINEAR,
        growth_rate=1,
        initial_pressure=0.01,
        target_volume=30,
        Kp=0.05,
        Ki=0.00001,
        Kd=0.5,
    ):
        self.growth_type = growth_type
        self.growth_rate = growth_rate  # in cubic microns per frame
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.PID_scale = 15
        self.initial_pressure = initial_pressure
        self.target_volume = target_volume

        self.integral = 0
        self.previous_error = 0
        self.previous_pressure = self.initial_pressure
        self.next_volume = current_volume

    def copy(self):
        return copy(self)

    def set_pressure(self, pressure):
        self.previous_pressure = pressure

    def step_growth(self, current_volume, dt):
        """Step through the growth controller to calculate normal growth.

        Args:
            current_volume (float): Current volume of the cell.
            dt (float): Time step.

        Returns:
                float: The pressure at the next time step.
        """
        match self.growth_type:
            case GrowthType.LINEAR:
                self.next_volume += self.growth_rate * dt
            case GrowthType.EXPONENTIAL:
                self.next_volume *= 1 + self.growth_rate * dt
            case GrowthType.LOGISTIC:
                self.next_volume = self.next_volume * (
                    1
                    + self.growth_rate
                    * (1 - self.next_volume / self.target_volume)
                    * dt
                )
            case _:
                raise ValueError(
                    "Growth type must be one of LINEAR, EXPONENTIAL, or LOGISTIC."
                )
        self.next_volume = min(self.next_volume, self.target_volume)
        volume_deviation = 1 - current_volume / self.next_volume

        # Update pressure based on PID output
        error = volume_deviation
        integral = self.integral + error
        derivative = error - self.previous_error
        pid = self.Kp * error + self.Ki * integral + self.Kd * derivative

        next_pressure = self.previous_pressure + pid * self.PID_scale

        # Update previous error and pressure for the next iteration
        self.previous_error = error
        self.integral = integral
        self.previous_pressure = next_pressure

        return next_pressure

    def step_divided(self, new_volume):
        # self.previous_pressure = self.initial_pressure
        self.next_volume = new_volume
