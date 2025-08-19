"""Time utilities cho scheduling"""

from datetime import datetime, timedelta
from typing import List, Tuple, Optional


class TimeUtils:
    """Helper functions cho time calculations"""
    
    @staticmethod
    def is_within_time_window(arrival_time: datetime, tw_start: datetime, tw_end: datetime) -> bool:
        """Kiểm tra arrival time có trong time window không"""
        return tw_start <= arrival_time <= tw_end
    
    @staticmethod
    def calculate_late_minutes(arrival_time: datetime, deadline: datetime) -> int:
        """Tính số phút trễ (0 nếu đúng giờ)"""
        if arrival_time <= deadline:
            return 0
        
        late_duration = arrival_time - deadline
        return int(late_duration.total_seconds() / 60)
    
    @staticmethod
    def calculate_early_minutes(arrival_time: datetime, earliest_time: datetime) -> int:
        """Tính số phút đến sớm (0 nếu đúng giờ)"""
        if arrival_time >= earliest_time:
            return 0
        
        early_duration = earliest_time - arrival_time
        return int(early_duration.total_seconds() / 60)
    
    @staticmethod
    def adjust_for_time_window(planned_time: datetime, tw_start: datetime, tw_end: datetime) -> datetime:
        """Điều chỉnh planned time để fit vào time window"""
        if planned_time < tw_start:
            return tw_start
        elif planned_time > tw_end:
            return tw_end
        else:
            return planned_time
    
    @staticmethod
    def get_next_available_time(current_time: datetime, service_duration_min: int) -> datetime:
        """Tính thời gian available tiếp theo sau khi hoàn thành service"""
        return current_time + timedelta(minutes=service_duration_min)
    
    @staticmethod
    def is_within_shift(time: datetime, shift_start: datetime, shift_end: datetime) -> bool:
        """Kiểm tra time có trong ca làm việc không"""
        return shift_start <= time <= shift_end
    
    @staticmethod
    def calculate_shift_overlap(activity_start: datetime, activity_end: datetime, 
                              shift_start: datetime, shift_end: datetime) -> timedelta:
        """Tính thời gian overlap giữa activity và shift"""
        overlap_start = max(activity_start, shift_start)
        overlap_end = min(activity_end, shift_end)
        
        if overlap_start >= overlap_end:
            return timedelta(0)
        
        return overlap_end - overlap_start
    
    @staticmethod
    def find_best_insertion_time(existing_schedule: List[Tuple[datetime, datetime]], 
                                new_duration: timedelta,
                                preferred_start: Optional[datetime] = None) -> Optional[datetime]:
        """
        Tìm thời điểm tốt nhất để insert activity mới vào schedule
        
        Args:
            existing_schedule: List của (start_time, end_time) đã schedule
            new_duration: Thời gian cần cho activity mới
            preferred_start: Thời gian prefer (optional)
            
        Returns:
            Thời điểm start tốt nhất, None nếu không tìm được
        """
        
        if not existing_schedule:
            return preferred_start or datetime.now()
        
        # Sort schedule by start time
        sorted_schedule = sorted(existing_schedule, key=lambda x: x[0])
        
        # Try to insert at preferred time first
        if preferred_start:
            new_end = preferred_start + new_duration
            
            # Check if this slot is free
            conflicts = any(
                not (new_end <= start or preferred_start >= end)
                for start, end in sorted_schedule
            )
            
            if not conflicts:
                return preferred_start
        
        # Find gaps in schedule
        gaps = []
        
        # Gap before first activity
        first_start = sorted_schedule[0][0]
        if preferred_start and preferred_start + new_duration <= first_start:
            return preferred_start
        
        # Gaps between activities
        for i in range(len(sorted_schedule) - 1):
            current_end = sorted_schedule[i][1]
            next_start = sorted_schedule[i + 1][0]
            
            gap_duration = next_start - current_end
            if gap_duration >= new_duration:
                # Found suitable gap
                return current_end
        
        # Gap after last activity
        last_end = sorted_schedule[-1][1]
        return last_end
    
    @staticmethod
    def round_to_minutes(dt: datetime, minutes: int = 5) -> datetime:
        """Round datetime đến minute gần nhất"""
        minute = (dt.minute // minutes) * minutes
        return dt.replace(minute=minute, second=0, microsecond=0)
    
    @staticmethod
    def get_time_bucket(dt: datetime, bucket_minutes: int = 15) -> datetime:
        """Get time bucket cho caching"""
        minute_bucket = (dt.minute // bucket_minutes) * bucket_minutes
        return dt.replace(minute=minute_bucket, second=0, microsecond=0)
