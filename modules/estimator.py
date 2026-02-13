# modules/estimator.py

class DistanceEstimator:
    def __init__(self, focal_length_px, real_width_cm):
        """
        Khởi tạo bộ ước lượng khoảng cách.
        :param focal_length_px: Tiêu cự ảo (tính bằng pixel)
        :param real_width_cm: Chiều rộng thực tế của vật thể (cm)
        """
        self.focal_length = focal_length_px
        self.real_width = real_width_cm

    def estimate(self, width_px):
        """
        Tính khoảng cách dựa trên độ rộng pixel.
        Công thức: D = (W_real * Focal) / W_pixel
        """
        if width_px == 0: return 0
        
        distance_cm = (self.real_width * self.focal_length) / width_px
        return distance_cm / 100.0  # Đổi sang mét (m)