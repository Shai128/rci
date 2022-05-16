import abc


class TSCalibration(abc.ABC):
    @abc.abstractmethod
    def __init__(self, desired_coverage_level, **kwargs):
        self.desired_coverage_level = desired_coverage_level
        pass

    @abc.abstractmethod
    def fit(self, x_cal, y_cal, predicted_interval, **kwargs):
        pass

    @abc.abstractmethod
    def calibrate(self, x, y, predicted_interval, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        pass


class DummyTSCalibration(TSCalibration):
    def __init__(self, desired_coverage_level, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, **kwargs)

    def fit(self, x_cal, y_cal, predicted_interval, **kwargs):
        return

    def calibrate(self, x, y, predicted_interval, **kwargs):
        return predicted_interval

