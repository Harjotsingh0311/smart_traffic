# signal_time.py
import math


class TrafficSignalController:
    def __init__(self):
        self.minGreen  = 10
        self.maxGreen  = 60
        self.noOfLanes = 2

        # Seconds per vehicle type to clear intersection
        self.carTime      = 2
        self.busTime      = 3
        self.truckTime    = 3
        self.bikeTime     = 1
        self.rickshawTime = 1.5

        self.priority_detected = False
        self.priority_lane     = None
        self.forced_lane       = 'RIGHT'

    def calculate_green_time(self, vehicle_counts):
        """Adaptive green time based on vehicle density."""
        cars      = vehicle_counts.get('car',           0)
        buses     = vehicle_counts.get('bus',           0)
        trucks    = vehicle_counts.get('truck',         0)
        bikes     = vehicle_counts.get('motorcycle',    0)
        rickshaws = vehicle_counts.get('auto-rickshaw', 0)

        t = math.ceil(
            (cars      * self.carTime      +
             buses     * self.busTime      +
             trucks    * self.truckTime    +
             bikes     * self.bikeTime     +
             rickshaws * self.rickshawTime)
            / (self.noOfLanes + 1)
        )
        return max(self.minGreen, min(t, self.maxGreen))

    def set_priority_vehicle(self, lane_id):
        self.priority_detected = True
        self.priority_lane     = lane_id

    def reset_priority_vehicle(self):
        self.priority_detected = False
        self.priority_lane     = None

    def get_green_lane(self, lane_vehicle_counts):
        # Priority vehicle always wins
        if self.priority_detected and self.priority_lane:
            return self.priority_lane
        # Auto-rotate timer
        if self.forced_lane:
            return self.forced_lane
        # Fallback: busiest lane
        totals = {
            lane: sum(
                v for k, v in counts.items()
                if k != 'priority_vehicle'
            )
            for lane, counts in lane_vehicle_counts.items()
        }
        return max(totals, key=totals.get) if totals else 'RIGHT'